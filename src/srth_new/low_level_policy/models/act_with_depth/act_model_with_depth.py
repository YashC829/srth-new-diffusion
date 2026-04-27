from __future__ import annotations

import os
from typing import List, Literal

from omegaconf import DictConfig, OmegaConf
import torch
from torchvision import transforms
from torch.nn import functional as F

from srth_new.general.utils.lang_encoding import encode_text, initialize_model_and_tokenizer
from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy
from srth_new.low_level_policy.models.detr.models.backbone import build_image_backbone
from srth_new.low_level_policy.models.detr.models.transformer import build_transformer
from srth_new.low_level_policy.models.detr.models.detr_vae import build_encoder
from srth_new.low_level_policy.models.act_with_depth.detr_vae_depth import (
    DETRVAEDepth,
)
from srth_new.general.third_party.EndoSynth.endosynth.models import load as load_depth_model
from srth_new.low_level_policy.dataset.img_aug import ImageAug

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


class ACTPolicyDepth(DVRKPolicy):
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
            img_resize_cfg: DictConfig,
            img_backbone_cfg: DictConfig,
            transformer_cfg: DictConfig,
            encoder_cfg: DictConfig,
            img_aug_cfg: DictConfig
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

        self.img_resize_cfg = img_resize_cfg
        self.kl_weight = kl_weight
        self.state_dim = action_dim
        self.use_language = use_language
        self.language_encoder = language_encoder

        # build image augmentation pipeline
        self.img_aug_dict = self._build_img_aug_dict(img_aug_cfg)

        # get depth model
        self.MAX_DEPTH_VAL = 0.3
        self.depth_model = load_depth_model("dav2")

        # BUILD MODEL AND OPTIMIZER
        img_backbones = list()
        for _ in range(len(camera_names)):
            img_backbone = build_image_backbone(**img_backbone_cfg) # type:ignore
            img_backbones.append(img_backbone)
        depth_backbone = build_image_backbone(**img_backbone_cfg) # type:ignore

        transformer = build_transformer(transformer_cfg)
        encoder = build_encoder(encoder_cfg)

        self.model = DETRVAEDepth(
            img_backbones,
            depth_backbone,
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

    def _build_img_aug_dict(self, cfg: DictConfig):
        aug_dict = dict()
        for camera_name, camera_aug_cfg in cfg.items():
            aug_dict[camera_name] = ImageAug(**camera_aug_cfg)

        return aug_dict

    @staticmethod
    def _serialize_img_resize_cfg(img_resize_cfg: DictConfig) -> object:
        return OmegaConf.to_container(
            img_resize_cfg,
            resolve=True,
            throw_on_missing=True,
        )

    @staticmethod
    def _deserialize_img_resize_cfg(serialized_img_resize_cfg: object) -> DictConfig:
        if isinstance(serialized_img_resize_cfg, DictConfig):
            return serialized_img_resize_cfg

        restored_cfg = OmegaConf.create(serialized_img_resize_cfg)
        if not isinstance(restored_cfg, DictConfig):
            raise TypeError("Checkpoint img_resize_cfg must deserialize to a mapping.")
        return restored_cfg

    def _move_depth_model_to_device(self, device: torch.device) -> None:
        """Move the wrapped EndoSynth depth model to the requested device."""
        self.depth_model.device = torch.device(device)
        self.depth_model._model = self.depth_model._model.to(device).eval()
        self.depth_model.act = self.depth_model.act.to(device).eval()

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = next(module.buffers()).device
        self._move_depth_model_to_device(device)
        return module

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

        serialized_img_resize_cfg = model_dict.get("img_resize_cfg")
        policy_config = model_dict.get("policy_config")
        if serialized_img_resize_cfg is None and isinstance(
            policy_config, (dict, DictConfig)
        ):
            serialized_img_resize_cfg = policy_config.get("img_resize_cfg")
        if serialized_img_resize_cfg is not None:
            self.img_resize_cfg = self._deserialize_img_resize_cfg(
                serialized_img_resize_cfg
            )

    def _get_depth(self, img: torch.Tensor):
        """Given an rgb image, generate the depth map for the image."""
        return self.depth_model.infer_tensor(img)

    def preprocess_images(
            self, 
            endoscope_img: torch.Tensor, 
            lw_img: torch.Tensor, 
            rw_img: torch.Tensor,
            use_augmentation: bool = False
        ):
        """Resizes images, gets depth image from endoscope image, and performs
        image augmentation."""

        depth_img = self._get_depth(endoscope_img)

        def resize_img(img, new_size: List):
            h_new, w_new = new_size[0], new_size[1]
            return F.interpolate(
                img,
                size=(h_new, w_new),
                mode="bilinear",        # best for images
                align_corners=False
            )
        
        endo_processed = resize_img(endoscope_img.float(), self.img_resize_cfg["left"]).clamp(0, 255.0).to(torch.uint8)
        lw_processed = resize_img(lw_img.float(), self.img_resize_cfg["left_wrist"]).clamp(0, 255.0).to(torch.uint8)
        rw_processed = resize_img(rw_img.float(), self.img_resize_cfg["right_wrist"]).clamp(0, 255.0).to(torch.uint8)

        # resize, clamp, and normalize depth
        depth_processed = resize_img(depth_img.float(), self.img_resize_cfg["left"])

        # AUGMENT IMAGES
        # pass the endo and depth images together to get consistent augmentations
        # across the two images
        if use_augmentation:
            endo_processed, depth_processed = self.img_aug_dict["endoscope_img"](
                endo_processed, depth_processed, kinds=["image", "depth"]
            )

            lw_processed = self.img_aug_dict["lw_img"](lw_processed, apply_random_shift=False)
            rw_processed = self.img_aug_dict["rw_img"](rw_processed, apply_random_shift=False)

        # normalize images with imagenet mean/std
        endo_processed = self.image_normalize(endo_processed / 255.0)
        lw_processed = self.image_normalize(lw_processed / 255.0)
        rw_processed = self.image_normalize(rw_processed / 255.0)

        # normalize depth with min/max normalization
        depth_processed = torch.clamp(depth_processed, min=0.0, max=self.MAX_DEPTH_VAL)
        depth_processed = depth_processed / self.MAX_DEPTH_VAL

        return endo_processed, depth_processed, lw_processed, rw_processed

    def forward(
        self,
        endoscope_img: torch.Tensor, 
        lw_img: torch.Tensor, 
        rw_img: torch.Tensor,
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
        endoscope_img, depth_img, lw_img, rw_img = self.preprocess_images(
            endoscope_img, lw_img, rw_img, use_augmentation=self.training
        )
        # stack the images
        rgb_img_stack = torch.stack([endoscope_img, lw_img, rw_img], dim=1)
        env_state = None
        batch_size = rgb_img_stack.shape[0]
        # since the dVRK is so inaccurate in an absolute setting, we set the absolute
        # qpos to zero so that this will not have an impact on the model
        model_qpos = torch.zeros(
            (batch_size, self.state_dim), dtype=rgb_img_stack.dtype, device=rgb_img_stack.device
        )
        command_embedding = self._encode_command_text(command_text, rgb_img_stack.device)

        if actions is not None: # training or validation
            if is_pad is None:
                raise Exception()
            # we keep track of the various commands to send to the robot and
            # save these in the model checkpoint
            self._record_training_command_text(command_text)
            processed_actions = self.prepare_actions_for_training(
                current_pose, actions, is_pad, rgb_img_stack.device
            )
            processed_actions = processed_actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                model_qpos,
                rgb_img_stack,
                depth_img,
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
        else:  # pure inference time
            a_hat, _, (_, _) = self.model(
                model_qpos,
                rgb_img_stack,
                depth_img,
                env_state,
                command_embedding=command_embedding
            )  # no action, sample from prior
            if return_policy_actions:
                return a_hat
            return self.postprocess_actions(a_hat, current_pose)

    def configure_optimizers(self):
        """Return the optimizer constructed alongside the ACT model."""
        return self.optimizer
