import random
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import kornia.augmentation as KA
from PIL import Image

TensorOrNone = Optional[torch.Tensor]

from matplotlib import colormaps as cm
import numpy as np
def depth2rgb(x: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    cmap = cm.get_cmap("Spectral")
    x = (np.clip(x, dmin, dmax) - dmin) / (dmax - dmin)
    x = (x * 255).astype(np.uint8)
    return (cmap(x) * 255)[..., :3]

class ImageAug(nn.Module):
    """
    Post-dataloader batch augmentation for a variable number of aligned tensors.

    Each input tensor must have shape:
        (B, C, H, W)

    Key idea:
    - Augmentation parameters are sampled ONCE per batch element.
    - The same spatial transforms and dropout mask are applied to every input tensor.
    - Color jitter is only applied to tensors marked as 'image'.

    Example:
        rgb_aug, depth_aug, mask_aug = aug(
            rgb, depth, mask,
            kinds=["image", "depth", "mask"]
        )

    Supported kinds:
        - "image": bilinear interpolation + color jitter eligible;
                   expected as uint8 in [0, 255]
        - "depth": bilinear interpolation + no color jitter
        - "mask": nearest interpolation + no color jitter
    """

    def __init__(
        self,
        crop_ratio: float = 0.95,
        rotate_deg: float = 5.0,
        brightness: float = 0.2,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
        max_holes: int = 128,
        min_holes: int = 1,
        pixel_dropout_p: float = 0.8,
        max_shift_x_ratio: float = 0.2,
        max_shift_y_ratio: float = 0.2,
        debug_save_dir: str | None = None,
        debug_name: str = "image_aug",
        debug_max_calls: int = 0,
        debug_max_samples: int = 4,
    ):
        super().__init__()

        self.crop_ratio = crop_ratio
        self.rotate_deg = rotate_deg

        self.max_holes = max_holes
        self.min_holes = min_holes
        self.pixel_dropout_p = pixel_dropout_p

        self.max_shift_x_ratio = max_shift_x_ratio
        self.max_shift_y_ratio = max_shift_y_ratio
        self.debug_save_dir = Path(debug_save_dir) if debug_save_dir else None
        self.debug_name = debug_name
        self.debug_max_calls = debug_max_calls
        self.debug_max_samples = debug_max_samples
        self.debug_call_count = 0

        self.color_jitter = KA.ColorJiggle(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0,
            same_on_batch=False,
            keepdim=True,
        )

    def forward(
        self,
        *inputs: torch.Tensor,
        kinds: Optional[Sequence[str]] = None,
        apply_color: bool = True,
        apply_spatial: bool = True,
        apply_pixel_dropout: bool = True,
        apply_random_shift: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            *inputs:
                Variable number of aligned tensors, each of shape (B, C, H, W).

            kinds:
                Sequence of same length as inputs.
                Each entry must be either:
                    - "image" (expected as uint8 in [0, 255])
                    - "depth"
                    - "mask"

                If None, all inputs are treated as "image".

            apply_color:
                Apply color jitter to inputs marked as "image" with 3 channels.

            apply_spatial:
                Apply shared random crop+resize and rotation.

            apply_pixel_dropout:
                Apply shared multi-hole dropout mask.

            apply_random_shift:
                Apply shared random translation.

        Returns:
            If one input is passed: returns one tensor.
            Otherwise: returns tuple of tensors in same order.
        """
        if len(inputs) == 0:
            raise ValueError("At least one input tensor must be provided.")

        if kinds is None:
            kinds = ["image"] * len(inputs)

        if len(kinds) != len(inputs):
            raise ValueError("len(kinds) must match number of input tensors.")

        self._validate_inputs(inputs, kinds)

        original_dtypes = [x.dtype for x in inputs]
        processed = [self._to_float_tensor(x) for x in inputs]

        b, _, h, w = processed[0].shape

        if apply_spatial:
            crop_params = self._sample_crop_params(b, h, w, processed[0].device)
            rot_params = self._sample_rotation_params(b, h, w, processed[0].device, processed[0].dtype)
            processed = [
                self._apply_spatial_to_tensor(x, kind, crop_params, rot_params)
                for x, kind in zip(processed, kinds)
            ]

        if apply_random_shift:
            shift_params = self._sample_shift_params(
                b, h, w, processed[0].device, processed[0].dtype
            )
            processed = [
                self._apply_shift_to_tensor(x, kind, shift_params)
                for x, kind in zip(processed, kinds)
            ]

        if apply_color:
            processed = [
                self._apply_color_to_tensor(x, kind)
                for x, kind in zip(processed, kinds)
            ]

        if apply_pixel_dropout:
            hole_mask = self._sample_dropout_mask(
                b, h, w, processed[0].device, processed[0].dtype
            )
            processed = [x * hole_mask for x in processed]

        outputs = [
            self._restore_dtype(x, dtype, kind)
            for x, dtype, kind in zip(processed, original_dtypes, kinds)
        ]

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_inputs(
        self,
        inputs: Sequence[torch.Tensor],
        kinds: Sequence[str],
    ) -> None:
        valid_kinds = {"image", "depth", "mask"}

        ref_shape = None
        for idx, (x, kind) in enumerate(zip(inputs, kinds)):
            if kind not in valid_kinds:
                raise ValueError(
                    f"Invalid kind '{kind}' at index {idx}. Must be 'image', 'depth', or 'mask'."
                )

            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input {idx} must be a torch.Tensor.")

            if x.ndim != 4:
                raise ValueError(
                    f"Input {idx} must have shape (B, C, H, W), got {tuple(x.shape)}"
                )

            if kind == "image" and x.dtype != torch.uint8:
                raise TypeError(
                    f"Input {idx} with kind 'image' must be torch.uint8 with values in "
                    f"[0, 255], got {x.dtype}."
                )

            if ref_shape is None:
                ref_shape = (x.shape[0], x.shape[2], x.shape[3])
            else:
                cur_shape = (x.shape[0], x.shape[2], x.shape[3])
                if cur_shape != ref_shape:
                    raise ValueError(
                        "All inputs must have matching batch size and spatial size. "
                        f"Expected (B,H,W)={ref_shape}, got {cur_shape} for input {idx}."
                    )

    # -------------------------------------------------------------------------
    # Param sampling
    # -------------------------------------------------------------------------

    def _sample_crop_params(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> dict:
        crop_h = max(1, int(h * self.crop_ratio))
        crop_w = max(1, int(w * self.crop_ratio))

        max_top = h - crop_h
        max_left = w - crop_w

        if max_top > 0:
            tops = torch.randint(0, max_top + 1, (b,), device=device)
        else:
            tops = torch.zeros((b,), device=device, dtype=torch.long)

        if max_left > 0:
            lefts = torch.randint(0, max_left + 1, (b,), device=device)
        else:
            lefts = torch.zeros((b,), device=device, dtype=torch.long)

        return {
            "crop_h": crop_h,
            "crop_w": crop_w,
            "tops": tops,
            "lefts": lefts,
            "out_h": h,
            "out_w": w,
        }

    def _sample_rotation_params(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        angles = (torch.rand(b, device=device, dtype=dtype) * 2.0 - 1.0) * self.rotate_deg

        center = torch.tensor(
            [[(w - 1) / 2.0, (h - 1) / 2.0]],
            device=device,
            dtype=dtype,
        ).repeat(b, 1)

        scale = torch.ones((b, 2), device=device, dtype=dtype)
        translations = torch.zeros((b, 2), device=device, dtype=dtype)

        M = K.geometry.transform.get_affine_matrix2d(
            translations=translations,
            center=center,
            scale=scale,
            angle=angles,
            sx=torch.zeros(b, device=device, dtype=dtype),
            sy=torch.zeros(b, device=device, dtype=dtype),
        )[:, :2, :]

        return {"M": M, "dsize": (h, w)}

    def _sample_shift_params(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        max_shift_x = self.max_shift_x_ratio * w
        max_shift_y = self.max_shift_y_ratio * h

        shift_x = (torch.rand(b, device=device, dtype=dtype) * 2.0 - 1.0) * max_shift_x
        shift_y = (torch.rand(b, device=device, dtype=dtype) * 2.0 - 1.0) * max_shift_y
        translations = torch.stack([shift_x, shift_y], dim=1)

        center = torch.tensor(
            [[(w - 1) / 2.0, (h - 1) / 2.0]],
            device=device,
            dtype=dtype,
        ).repeat(b, 1)

        scale = torch.ones((b, 2), device=device, dtype=dtype)
        angles = torch.zeros((b,), device=device, dtype=dtype)

        M = K.geometry.transform.get_affine_matrix2d(
            translations=translations,
            center=center,
            scale=scale,
            angle=angles,
            sx=torch.zeros(b, device=device, dtype=dtype),
            sy=torch.zeros(b, device=device, dtype=dtype),
        )[:, :2, :]

        return {"M": M, "dsize": (h, w)}

    def _sample_dropout_mask(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.rand(1, device=device).item() > self.pixel_dropout_p:
            return torch.ones((b, 1, h, w), device=device, dtype=dtype)

        min_height = max(1, h // 40)
        min_width = max(1, w // 40)
        max_height = min(max(1, h // 30), h)
        max_width = min(max(1, w // 30), w)

        hole_mask = torch.ones((b, 1, h, w), device=device, dtype=dtype)

        for i in range(b):
            n_holes = random.randint(self.min_holes, self.max_holes)
            for _ in range(n_holes):
                hole_h = random.randint(min_height, max_height)
                hole_w = random.randint(min_width, max_width)

                top_max = max(0, h - hole_h)
                left_max = max(0, w - hole_w)

                top = 0 if top_max == 0 else random.randint(0, top_max)
                left = 0 if left_max == 0 else random.randint(0, left_max)

                hole_mask[i, :, top:top + hole_h, left:left + hole_w] = 0.0

        return hole_mask

    # -------------------------------------------------------------------------
    # Transform application
    # -------------------------------------------------------------------------

    def _apply_spatial_to_tensor(
        self,
        x: torch.Tensor,
        kind: str,
        crop_params: dict,
        rot_params: dict,
    ) -> torch.Tensor:
        b, _, _, _ = x.shape
        crop_h = crop_params["crop_h"]
        crop_w = crop_params["crop_w"]
        out_h = crop_params["out_h"]
        out_w = crop_params["out_w"]
        tops = crop_params["tops"]
        lefts = crop_params["lefts"]

        use_bilinear = kind in {"image", "depth"}
        mode_resize = "bilinear" if use_bilinear else "nearest"
        mode_warp = "bilinear" if use_bilinear else "nearest"

        cropped = []
        for i in range(b):
            top = int(tops[i].item())
            left = int(lefts[i].item())

            xi = x[i:i + 1, :, top:top + crop_h, left:left + crop_w]
            xi = F.interpolate(
                xi,
                size=(out_h, out_w),
                mode=mode_resize,
                align_corners=False if mode_resize != "nearest" else None,
            )
            cropped.append(xi)

        x = torch.cat(cropped, dim=0)

        x = K.geometry.transform.warp_affine(
            x,
            rot_params["M"],
            dsize=rot_params["dsize"],
            mode=mode_warp,
            padding_mode="reflection",
            align_corners=False,
        )
        return x

    def _apply_shift_to_tensor(
        self,
        x: torch.Tensor,
        kind: str,
        shift_params: dict,
    ) -> torch.Tensor:
        mode_warp = "bilinear" if kind in {"image", "depth"} else "nearest"

        x = K.geometry.transform.warp_affine(
            x,
            shift_params["M"],
            dsize=shift_params["dsize"],
            mode=mode_warp,
            padding_mode="reflection",
            align_corners=False,
        )
        return x

    def _apply_color_to_tensor(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "image" and x.shape[1] == 3:
            x = x.clamp(min=0.0)
            x = self.color_jitter(x)
        return x

    # -------------------------------------------------------------------------
    # Dtype helpers
    # -------------------------------------------------------------------------

    def _to_float_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        if not x.is_floating_point():
            return x.float()
        return x

    def _restore_dtype(
        self,
        x: torch.Tensor,
        dtype: torch.dtype,
        kind: str,
    ) -> torch.Tensor:
        if dtype == torch.uint8:
            return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

        if dtype == torch.bool:
            return x > 0.5

        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return x.round().to(dtype)

        return x.to(dtype)
