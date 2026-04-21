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
from srth_new.low_level_policy.dataset.img_aug_new import ImageAug

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

    def _build_img_aug_dict(self, cfg: DictConfig):
        aug_dict = dict()
        for camera_name, camera_aug_cfg in cfg.items():
            aug_dict[camera_name] = ImageAug(**camera_aug_cfg)

        return aug_dict

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

    def preprocess_images(
            self, 
            endoscope_img: torch.Tensor, 
            lw_img: torch.Tensor, 
            rw_img: torch.Tensor
        ):
        """Resizes images and performs image augmentation."""

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

        # AUGMENT IMAGES (input images must be [0, 255] uint8)
        # pass the endo and depth images together to get consistent augmentations
        # across the two images
        endo_processed = self.img_aug_dict["endoscope_img"](endo_processed)
        lw_processed = self.img_aug_dict["lw_img"](lw_processed, apply_random_shift=False)
        rw_processed = self.img_aug_dict["rw_img"](rw_processed, apply_random_shift=False)
        # output in the same dtype as original inputs ([0, 255] uint8)

        # Debug show augmented images...
        # from PIL import Image
        # import os
        # os.makedirs("temp", exist_ok=True)
        # for batch_idx in range(endo_processed.shape[0]):
        #     Image.fromarray(endo_processed[batch_idx].cpu().numpy().transpose(1, 2, 0)).save(f"./temp/endo_{batch_idx}.png")
        #     Image.fromarray(lw_processed[batch_idx].cpu().numpy().transpose(1, 2, 0)).save(f"./temp/lw_{batch_idx}.png")
        #     Image.fromarray(rw_processed[batch_idx].cpu().numpy().transpose(1, 2, 0)).save(f"./temp/rw_{batch_idx}.png")

        # normalize images with imagenet mean/std
        endo_processed = self.image_normalize(endo_processed / 255.0)
        lw_processed = self.image_normalize(lw_processed / 255.0)
        rw_processed = self.image_normalize(rw_processed / 255.0)

        return endo_processed, lw_processed, rw_processed

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

        endo_img, lw_img, rw_img = self.preprocess_images(endoscope_img, lw_img, rw_img)
        # stack the images
        image = torch.stack([endo_img, lw_img, rw_img], dim=1)
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
