from __future__ import annotations

from typing import List, Literal

import albumentations as A
import cv2
import numpy as np
from omegaconf import DictConfig
import kornia.augmentation as K
import torch
from torchvision import transforms
import torchvision.transforms as T
from torch.nn import functional as F
from torch import nn

from srth_new.general.utils.lang_encoding import encode_text, initialize_model_and_tokenizer
from .dvrk_policy import DVRKPolicy
from srth_new.low_level_policy.models.detr.models.backbone import build_image_backbone
from srth_new.low_level_policy.models.detr.models.transformer import build_transformer
from srth_new.low_level_policy.models.detr.models.detr_vae import build_encoder
from srth_new.low_level_policy.models.detr.models.detr_vae import DETRVAE

import logging
log = logging.getLogger(__name__)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTPolicy(DVRKPolicy):
    """ACT policy wrapper used by the low-level training and inference code.

    This class layers project-specific behavior around the DETR/ACT backbone:

    - builds the ACT model and optimizer from the DETR configuration
    - caches and applies dataset statistics used to normalize action targets
    - converts between absolute robot actions and the relative policy action
      representation expected by the network
    - optionally encodes language commands for conditioned policies
    - serializes both learned weights and lightweight metadata needed to resume
      training or run inference later

    A subtle but important detail is that the internal ACT model currently
    receives a zero-valued proprioceptive input (`model_qpos`) during the
    forward pass. The externally supplied `current_pose` is still required,
    because it is used to transform dataset actions into the policy's relative
    action space during training and to convert predicted policy actions back
    into absolute robot commands during inference.
    """

    def __init__(
            self,
            lr: float,
            weight_decay: float,
            camera_names: List[str],
            num_queries: int,
            action_dim: int,
            kl_weight: float,
            use_language: bool,
            language_encoder: str,
            action_mode: Literal["hybrid_relative", "ego", "relative_endoscope"],
            norm_scheme: Literal["std", "min_max"],
            img_backbone_cfg: DictConfig,
            transformer_cfg: DictConfig,
            encoder_cfg: DictConfig
        ):
        """Initialize the policy, optimizer, and optional language encoder.

        Args:
            args_override: Mapping of ACT / DETR configuration values. The
                contents are forwarded to `build_ACT_model_and_optimizer`, then
                a few policy-specific keys such as `kl_weight`,
                `action_mode`, `norm_scheme`, and language settings are read
                directly from the same mapping.
        """
        super().__init__(
            action_dim=action_dim,
            action_mode=action_mode,
            norm_scheme=norm_scheme,
        )

        self.kl_weight = kl_weight
        self.state_dim = action_dim
        self.use_language = use_language
        self.language_encoder = language_encoder

        # BUILD MODEL AND OPTIMIZER
        img_backbones = list()
        for _ in range(len(camera_names)):
            img_backbone = build_image_backbone(**img_backbone_cfg) # type:ignore
            img_backbones.append(img_backbone)

        transformer = build_transformer(transformer_cfg)
        encoder = build_encoder(encoder_cfg)

        self.model = DETRVAE(
            img_backbones,
            transformer,
            encoder,
            state_dim=action_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            use_language=use_language,
            use_film="film" in img_backbone_cfg.backbone_type,
        ) 
        self.optimizer = torch.optim.AdamW(
            self._get_param_dict(self.model, img_backbone_cfg), lr=lr, weight_decay=weight_decay
        )

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self._command_embedding_cache = {}
        self.training_text_conditionings: list[str] = []
        self.tokenizer = None
        self.language_model = None
        if self.use_language:
            self.tokenizer, self.language_model = initialize_model_and_tokenizer(
                self.language_encoder
            )
            self.language_model.eval()

        log.info(f"KL Weight {self.kl_weight}")
        self.num_queries = self.model.num_queries # type:ignore

        # self._init_data_aug(data_aug_cfg)

    # def _init_data_aug(self, data_aug_cfg: DictConfig):
    #     self.data_aug = nn.ModuleDict()

    #     for camera_name, aug_cfg in data_aug_cfg.items():
    #         camera_aug = nn.ModuleDict()
    #         targ_shape = aug_cfg.resize_shape   # [H, W]
    #         targ_h, targ_w = int(targ_shape[0]), int(targ_shape[1])
    #         aug_keys = set(str(x) for x in aug_cfg.keys())

    #         if "spatial" in aug_keys:
    #             spatial_cfg = aug_cfg.spatial
    #             crop_h = int(targ_h * spatial_cfg.crop_ratio)
    #             crop_w = int(targ_w * spatial_cfg.crop_ratio)

    #             camera_aug["spatial"] = nn.Sequential(
    #                 K.RandomCrop(
    #                     size=(crop_h, crop_w),
    #                     p=1.0,
    #                     same_on_batch=False,
    #                 ),
    #                 K.Resize(
    #                     size=(targ_h, targ_w),
    #                     antialias=True,
    #                 ),
    #                 K.RandomRotation(
    #                     degrees=5.0,
    #                     p=1.0,
    #                     same_on_batch=False,
    #                     resample="BILINEAR",
    #                     keepdim=True,
    #                 ),
    #             )

    #         if "color_jitter" in aug_keys:
    #             cj_cfg = aug_cfg.color_jitter
    #             camera_aug["color_jitter"] = K.ColorJitter(
    #                 brightness=cj_cfg.brightness,
    #                 contrast=cj_cfg.contrast,
    #                 saturation=cj_cfg.saturation,
    #                 hue=cj_cfg.hue,
    #                 p=1.0,
    #                 same_on_batch=False,
    #             )

    #         if "pixel_dropout" in aug_keys:
    #             pd_cfg = aug_cfg.pixel_dropout
    #             min_height = max(1, targ_h // 40)
    #             min_width = max(1, targ_w // 40)
    #             max_height = min(targ_h // 30, targ_h)
    #             max_width = min(targ_w // 30, targ_w)

    #             camera_aug["pixel_dropout"] = CoarseDropoutTorch(
    #                 min_holes=pd_cfg.min_holes,
    #                 max_holes=pd_cfg.max_holes,
    #                 min_height=min_height,
    #                 max_height=max_height,
    #                 min_width=min_width,
    #                 max_width=max_width,
    #                 fill_value=0.0,
    #                 p=pd_cfg.p,
    #             )

    #         if "random_shift" in aug_keys:
    #             rs_cfg = aug_cfg.random_shift

    #             # Kornia RandomAffine translate expects FRACTIONS of image size.
    #             camera_aug["random_shift"] = K.RandomAffine(
    #                 degrees=0.0,
    #                 translate=(
    #                     float(rs_cfg.max_shift_x_ratio),
    #                     float(rs_cfg.max_shift_y_ratio),
    #                 ),
    #                 scale=None,
    #                 shear=None,
    #                 p=1.0,
    #                 same_on_batch=False,
    #                 resample="BILINEAR",
    #                 padding_mode="border",
    #                 keepdim=True,
    #             )

    #         self.data_aug[camera_name] = camera_aug

    # def _init_data_aug(self, data_aug_cfg: DictConfig):

    #     self.data_aug = dict()

    #     for camera_name, aug_cfg in data_aug_cfg.items():
    #         self.data_aug[camera_name] = dict()
    #         targ_shape = aug_cfg.resize_shape
    #         aug_keys = set(str(x) for x in aug_cfg.keys())

    #         if "spatial" in aug_keys:
    #             spatial_cfg = aug_cfg.spatial
    #             self.data_aug[camera_name]["spatial"] = A.Compose([
    #                     A.RandomCrop(
    #                         height=int(targ_shape[0] * spatial_cfg.crop_ratio), 
    #                         width=int(targ_shape[1] * spatial_cfg.crop_ratio)
    #                     ),
    #                     A.Resize(height=targ_shape[0], width=targ_shape[1]),
    #                     A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101),
    #                 ])
                
    #         if "color_jitter" in aug_keys:
    #             cj_cfg = aug_cfg.color_jitter
    #             self.data_aug[camera_name]["color_jitter"] = T.ColorJitter(
    #                 brightness=cj_cfg.brightness, contrast=cj_cfg.contrast, 
    #                 saturation=cj_cfg.saturation, hue=cj_cfg.hue
    #             )

    #         if "pixel_dropout" in aug_keys:
    #             pd_cfg = aug_cfg.pixel_dropout
    #             min_height = max(1, targ_shape[0] // 40)
    #             min_width = max(1, targ_shape[1] // 40)
    #             max_height = min(targ_shape[0] // 30, targ_shape[0])
    #             max_width = min(targ_shape[1] // 30, targ_shape[1])
    #             self.data_aug[camera_name]["pixel_dropout"] = A.Compose([
    #                 A.CoarseDropout(max_holes=pd_cfg.max_holes, max_height=max_height, max_width=max_width, # type:ignore
    #                                 min_holes=pd_cfg.min_holes, min_height=min_height, min_width=min_width, # type:ignore
    #                                 fill_value=0, p=pd_cfg.p), # type:ignore
    #             ]) # type:ignore

    #         if "random_shift" in aug_keys:
    #             rs_cfg = aug_cfg.random_shift

    #             MAX_SHIFT_X = int(targ_shape[1] * rs_cfg.max_shift_x_ratio)
    #             MAX_SHIFT_Y = int(targ_shape[0] * rs_cfg.max_shift_y_ratio)
    #             def random_shift(img):
    #                 shift_x = np.random.randint(-MAX_SHIFT_X, MAX_SHIFT_X)
    #                 shift_y = np.random.randint(-MAX_SHIFT_Y, MAX_SHIFT_Y)
    #                 img = T.functional.affine(img, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
    #                 return img

    #             self.data_aug[camera_name]["random_shift"] = random_shift

    # def _augment_img(self, img: torch.Tensor, camera_name: str):
    #     img_aug_dict = self.data_aug[camera_name]

    #     aug_types = set(img_aug_dict.keys())
    #     device = img.device
    #     img = img.cpu().numpy()

    #     if "spatial" in aug_types:
    #         img = img_aug_dict["spatial"](img)
    #     if "pixel_dropout" in aug_types:
    #         img = img_aug_dict["pixel_dropout"](image=img)
    #     if "color_jitter" in aug_types:
    #         img = img_aug_dict["color_jitter"](img)
    #     if "random_shift" in aug_types:
    #         img = img_aug_dict["random_shift"](img)
        
    #     return torch.tensor(img).to(device)

    def _get_param_dict(self, model, backbone_cfg: DictConfig):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": backbone_cfg.lr_backbone,
            },
        ]
        return param_dicts

    def _encode_command_text(self, command_text, device: torch.device):
        """Encode one or more command strings for language-conditioned ACT.

        Encoded text embeddings are cached by string value to avoid repeated
        encoder calls during training when the same command appears many times.

        Args:
            command_text: A single string or a batch of strings. Must be
                provided when `use_language=True`.
            device: Device where the returned embedding tensor should live.

        Returns:
            A tensor of shape `(batch, embedding_dim)` or `None` when language
            conditioning is disabled.
        """
        if not self.use_language:
            return None
        if command_text is None:
            raise ValueError("command_text is required when use_language=True")

        texts = self._normalize_command_text(command_text)
        if not texts:
            raise ValueError("command_text must contain at least one string")

        embeddings = []
        for text in texts:
            if text not in self._command_embedding_cache:
                embedding = torch.as_tensor(
                    encode_text(
                        text,
                        self.language_encoder,
                        self.tokenizer,
                        self.language_model,
                    ),
                    dtype=torch.float32,
                ).flatten()
                self._command_embedding_cache[text] = embedding.cpu()
            embeddings.append(self._command_embedding_cache[text])

        return torch.stack(embeddings, dim=0).to(device)

    @staticmethod
    def _normalize_command_text(command_text) -> list[str]:
        if command_text is None:
            return []
        if isinstance(command_text, str):
            texts = [command_text]
        else:
            texts = list(command_text)
        return [
            text if isinstance(text, str) else str(text)
            for text in texts
            if text is not None
        ]

    def _record_training_command_text(self, command_text) -> None:
        self.training_text_conditionings.extend(
            self._normalize_command_text(command_text)
        )

    def _serialize_policy_config(self) -> dict[str, object]:
        policy_config = super()._serialize_policy_config()
        policy_config.update(
            {
                "use_language": self.use_language,
                "language_encoder": self.language_encoder,
            }
        )
        return policy_config

    def _serialize_checkpoint_metadata(self) -> dict[str, object]:
        return {
            "training_text_conditionings": list(self.training_text_conditionings),
        }

    def _restore_checkpoint_metadata(self, model_dict: dict[str, object]) -> None:
        self.training_text_conditionings = list(
            model_dict.get("training_text_conditionings", []) # type:ignore
        )

    def forward(
        self,
        image,
        current_pose,
        actions=None,
        is_pad=None,
        command_text=None,
        return_policy_actions: bool = False,
    ):
        """Run the policy in training or inference mode.

        Training mode is selected by passing `actions`. In that case the method
        converts absolute actions into the policy representation, runs the ACT
        model, and returns a loss dictionary containing L1, KL, and total loss.

        In inference mode, the method samples from the ACT prior, predicts a
        sequence of policy actions, and either returns those raw policy-space
        predictions or converts them into absolute robot commands.

        Args:
            image: Image batch shaped `(B, num_cameras, C, H, W)` in `[0, 1]`.
            current_pose: Current robot pose used for action conversion.
            actions: Optional absolute action targets. Supplying this switches
                the method into training mode.
            is_pad: Optional padding mask aligned with `actions`.
            command_text: Optional command string or batch of strings for
                language-conditioned policies.
            return_policy_actions: In inference mode, return the raw policy
                action tensor instead of absolute robot actions.

        Returns:
            In training mode, a dictionary of loss tensors. In inference mode,
            either a tensor of predicted policy actions or a tensor of absolute
            robot actions.
        """
        env_state = None
        image = self.image_normalize(image)
        batch_size = image.shape[0]
        # since the dVRK is so inaccurate in an absolute setting, we set the absolute
        # qpos to zero so that this will not have an impact on the model
        model_qpos = torch.zeros(
            (batch_size, self.state_dim), dtype=image.dtype, device=image.device
        )
        command_embedding = self._encode_command_text(command_text, image.device)

        if actions is not None:  # training time
            if is_pad is None:
                raise Exception()
            # we keep track of the various commands to send to the robot and
            # save these in the model checkpoint
            self._record_training_command_text(command_text)
            processed_actions = self.prepare_actions_for_training(
                current_pose, actions, is_pad, image.device
            )
            processed_actions = processed_actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                model_qpos,
                image,
                env_state,
                processed_actions,
                is_pad,
                command_embedding=command_embedding,
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(processed_actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                model_qpos, image, env_state, command_embedding=command_embedding
            )  # no action, sample from prior
            if return_policy_actions:
                return a_hat
            return self.postprocess_actions(a_hat, current_pose)

    def configure_optimizers(self):
        """Return the optimizer constructed alongside the ACT model."""
        return self.optimizer
