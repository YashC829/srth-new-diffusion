'''
transformer implementation of diffusion policy using the cholecystectomy data. written with help of claude 4.6.
'''
import logging
import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torchvision import transforms

from srth_new.general.third_party.EndoSynth.endosynth.models import (
    load as load_depth_model,
)
from srth_new.general.utils.lang_encoding import (
    encode_text,
    initialize_model_and_tokenizer,
)
from srth_new.low_level_policy.dataset.img_aug import ImageAug
from srth_new.low_level_policy.models.detr.models.backbone import build_image_backbone
from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy

log = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings for diffusion timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionTransformerPolicy(DVRKPolicy):
    """Transformer-based diffusion policy with cross-attention.

    Each camera's backbone produces a spatial feature map. These are projected,
    flattened to tokens, and encoded by a transformer encoder to form a memory.
    During denoising, noisy action tokens cross-attend to this memory via a
    transformer decoder. This replaces both the 1D UNet noise predictor and the
    VAE-based ACT training objective with a DDPM diffusion loss.

    Note: action history conditioning is not included in this version.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        camera_names: List[str],
        num_queries: int,
        action_dim: int,
        use_language: bool,
        language_encoder: str,
        merge_recovery_phases: bool,
        action_mode: Literal["hybrid_relative", "ego", "relative_endoscope"],
        norm_scheme: Literal["std", "min_max"],
        img_resize_cfg: DictConfig,
        img_backbone_cfg: DictConfig,
        img_aug_cfg: DictConfig,
        use_depth: bool,
        hidden_dim: int,
        nheads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
    ):
        super().__init__(
            action_dim=action_dim,
            action_mode=action_mode,
            norm_scheme=norm_scheme,
        )

        self.camera_names = camera_names
        self.num_queries = num_queries
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.merge_recovery_phases = merge_recovery_phases
        self.use_depth = use_depth
        self.img_resize_cfg = img_resize_cfg
        self.use_film = "film" in img_backbone_cfg.backbone_type

        # ── Image augmentation ───────────────────────────────────────────────
        self.img_aug_dict = self._build_img_aug_dict(img_aug_cfg)

        # ── Depth model ──────────────────────────────────────────────────────
        self.MAX_DEPTH_VAL = 0.3
        self.depth_model = load_depth_model("dav2") if use_depth else None

        # ── Image backbones (one per camera) ─────────────────────────────────
        img_backbones = [build_image_backbone(**img_backbone_cfg) for _ in camera_names]
        self.backbones = nn.ModuleList(img_backbones)

        # Project backbone feature maps to hidden_dim (shared across cameras)
        self.input_proj = nn.Conv2d(img_backbones[0].num_channels, hidden_dim, kernel_size=1)

        # ── Depth backbone and RGB-D fusion ──────────────────────────────────
        if use_depth:
            self.depth_backbone = build_image_backbone(**img_backbone_cfg)
            self.depth_1d_to_3d_proj = nn.Conv2d(1, 3, kernel_size=1)
            self.depth_input_proj = nn.Conv2d(
                self.depth_backbone.num_channels, hidden_dim, kernel_size=1
            )
            # Modality embeddings distinguish RGB vs depth tokens before fusion
            self.aligned_modality_embed = nn.Embedding(2, hidden_dim)
            self.aligned_feature_fusion = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        else:
            self.depth_backbone = None

        # ── Transformer encoder: image tokens → memory ────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ── Transformer decoder: action tokens cross-attend to memory ─────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ── Timestep encoding ────────────────────────────────────────────────
        self.timestep_embed = SinusoidalPosEmb(hidden_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ── Action projections ───────────────────────────────────────────────
        self.action_input_proj = nn.Linear(action_dim, hidden_dim)
        self.action_output_proj = nn.Linear(hidden_dim, action_dim)

        # Learned positional embeddings for the action token sequence
        self.action_pos_embed = nn.Embedding(num_queries, hidden_dim)

        # ── Language ─────────────────────────────────────────────────────────
        # Language embedding is used two ways:
        #   1. FiLM conditioning inside the backbone (if use_film=True)
        #   2. Projected to a token appended to the image memory so action
        #      tokens can attend to it during decoding
        if use_language:
            self.lang_embed_proj = nn.Linear(768, hidden_dim)  # 768 = DistilBERT dim

        # ── Noise scheduler ──────────────────────────────────────────────────
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )
        self.num_inference_steps = num_inference_steps

        # ── Image normalization ──────────────────────────────────────────────
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ── Language model ───────────────────────────────────────────────────
        self._command_embedding_cache = {}
        self.training_text_conditionings: list[str] = []
        self.tokenizer = None
        self.language_model = None
        if use_language:
            self.tokenizer, self.language_model = initialize_model_and_tokenizer(
                language_encoder
            )
            self.language_model.eval()

        log.info(
            f"DiffusionTransformerPolicy: hidden_dim={hidden_dim}, "
            f"encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}, "
            f"num_queries={num_queries}, use_depth={use_depth}, use_language={use_language}"
        )

        # ── Optimizer ────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self._get_param_dict(self, img_backbone_cfg),
            lr=lr,
            weight_decay=weight_decay,
        )

    # ── Backbone helpers (mirrors DETRVAE) ───────────────────────────────────

    def _forward_backbone(self, backbone, image, command_embedding):
        """Forward one image through a backbone with optional FiLM conditioning."""
        if self.use_film:
            features, pos = backbone(image, command_embedding)
        else:
            features, pos = backbone(image)
        return features[0], pos[0]

    def _fuse_aligned_rgbd(
        self, rgb_feature: torch.Tensor, depth_feature: torch.Tensor
    ) -> torch.Tensor:
        """Fuse aligned RGB and depth feature maps with modality embeddings."""
        rgb_mod = self.aligned_modality_embed.weight[0][None, :, None, None]
        depth_mod = self.aligned_modality_embed.weight[1][None, :, None, None]
        fused = torch.cat([rgb_feature + rgb_mod, depth_feature + depth_mod], dim=1)
        return self.aligned_feature_fusion(fused)

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _build_img_aug_dict(self, cfg: DictConfig) -> dict:
        return {name: ImageAug(**cam_cfg) for name, cam_cfg in cfg.items()}

    # ── Optimizer ────────────────────────────────────────────────────────────

    def _get_param_dict(self, model, backbone_cfg: DictConfig) -> list:
        return [
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

    def configure_optimizers(self):
        return self.optimizer

    # ── Depth model device management ────────────────────────────────────────

    def _move_depth_model_to_device(self, device: torch.device) -> None:
        if self.depth_model is None:
            return
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

    # ── Language helpers ──────────────────────────────────────────────────────

    def _encode_command_text(
        self, command_text, device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.use_language:
            return None
        if command_text is None:
            raise ValueError("command_text is required when use_language=True")
        texts = self._normalize_command_text(command_text, self.merge_recovery_phases)
        if not texts:
            raise ValueError("command_text must contain at least one string")
        embeddings = []
        for text in texts:
            if text not in self._command_embedding_cache:
                embedding = torch.as_tensor(
                    encode_text(
                        text, self.language_encoder, self.tokenizer, self.language_model
                    ),
                    dtype=torch.float32,
                ).flatten()
                self._command_embedding_cache[text] = embedding.cpu()
            embeddings.append(self._command_embedding_cache[text])
        return torch.stack(embeddings, dim=0).to(device)

    @staticmethod
    def _normalize_command_text(command_text, merge_recovery_phases: bool) -> list[str]:
        # remove the "_recovery" suffix if merging recovery phases with their
        # respective "standard" phases
        if merge_recovery_phases:
            command_text = command_text.replace(" recovery", "")
        if command_text is None:
            return []
        texts = [command_text] if isinstance(command_text, str) else list(command_text)
        return [t if isinstance(t, str) else str(t) for t in texts if t is not None]

    def _record_training_command_text(self, command_text) -> None:
        self.training_text_conditionings.extend(
            self._normalize_command_text(command_text, self.merge_recovery_phases)
        )

    # ── Image preprocessing ───────────────────────────────────────────────────

    def _get_depth(self, img: torch.Tensor) -> Optional[torch.Tensor]:
        if self.depth_model is None:
            return None
        return self.depth_model.infer_tensor(img)

    def preprocess_images(
        self,
        endoscope_img: torch.Tensor,
        lw_img: torch.Tensor,
        rw_img: torch.Tensor,
        use_augmentation: bool = False,
    ):
        depth_img = self._get_depth(endoscope_img) if self.use_depth else None

        def resize_img(img, new_size):
            h, w = new_size[0], new_size[1]
            return F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)

        endo = resize_img(endoscope_img.float(), self.img_resize_cfg["left"]).clamp(0, 255).to(torch.uint8)
        lw = resize_img(lw_img.float(), self.img_resize_cfg["left_wrist"]).clamp(0, 255).to(torch.uint8)
        rw = resize_img(rw_img.float(), self.img_resize_cfg["right_wrist"]).clamp(0, 255).to(torch.uint8)

        depth = None
        if depth_img is not None:
            depth = resize_img(depth_img.float(), self.img_resize_cfg["left"])

        if use_augmentation:
            if depth is not None:
                endo, depth = self.img_aug_dict["endoscope_img"](
                    endo, depth, kinds=["image", "depth"]
                )
            else:
                endo = self.img_aug_dict["endoscope_img"](endo, apply_random_shift=False)
            lw = self.img_aug_dict["lw_img"](lw, apply_random_shift=False)
            rw = self.img_aug_dict["rw_img"](rw, apply_random_shift=False)

        endo = self.image_normalize(endo / 255.0)
        lw = self.image_normalize(lw / 255.0)
        rw = self.image_normalize(rw / 255.0)

        if depth is not None:
            depth = torch.clamp(depth, 0.0, self.MAX_DEPTH_VAL) / self.MAX_DEPTH_VAL

        return endo, depth, lw, rw

    # ── Observation encoding ──────────────────────────────────────────────────

    def encode_observations(
        self,
        rgb_img_stack: torch.Tensor,
        depth_img: Optional[torch.Tensor],
        command_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode all observations into a token sequence for cross-attention.

               For each camera, the backbone produces (raw_features, pos) both at
               backbone channel dimension (B, C_backbone, H, W). Positional embeddings
               are added to raw_features BEFORE input_proj so both tensors share the
               same channel dimension. The result is projected to hidden_dim and
               flattened to (B, H*W, hidden_dim).

               The first camera's feature (already at hidden_dim) is optionally fused
               with the depth feature (also at hidden_dim) before flattening. All
               camera tokens are concatenated and run through the transformer encoder.

               If use_language=True, the language embedding is projected to a single
               token and appended so action tokens can attend to it during decoding.

               Args:
                   rgb_img_stack: (B, n_cameras, C, H, W)
                   depth_img: (B, 1, H, W) or None
                   command_embedding: (B, 768) or None

               Returns:
                   memory: (B, S_total [+ 1 if language], hidden_dim)
               """
        all_tokens = []
        first_rgb_raw = None  # raw backbone features for camera 0 (before input_proj)

        for cam_id in range(len(self.camera_names)):
            # pos is discarded — it is sized to the DETR transformer's d_model,
            # not our hidden_dim. Our img_encoder handles spatial relationships
            # through self-attention without needing explicit positional encodings.
            raw_features, _ = self._forward_backbone(
                self.backbones[cam_id],
                rgb_img_stack[:, cam_id],
                command_embedding,
            )
            # raw_features: (B, C_backbone, H, W)

            if cam_id == 0:
                first_rgb_raw = raw_features
            else:
                projected = self.input_proj(raw_features)  # (B, hidden_dim, H, W)
                tokens = projected.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
                all_tokens.append(tokens)

        # Fuse depth into first camera feature map if available
        if depth_img is not None and self.depth_backbone is not None:
            depth_3d = self.depth_1d_to_3d_proj(depth_img)
            depth_raw, _ = self._forward_backbone(
                self.depth_backbone, depth_3d, command_embedding
            )
            first_projected = self.input_proj(first_rgb_raw)  # (B, hidden_dim, H, W)
            depth_projected = self.depth_input_proj(depth_raw)  # (B, hidden_dim, H, W)
            first_feature = self._fuse_aligned_rgbd(first_projected, depth_projected)
        else:
            first_feature = self.input_proj(first_rgb_raw)  # (B, hidden_dim, H, W)

        first_tokens = first_feature.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        all_tokens.insert(0, first_tokens)

        # Concatenate all camera tokens: (B, S_total, hidden_dim)
        memory = torch.cat(all_tokens, dim=1)

        # Append language token so action decoder can attend to it during cross-attention
        if command_embedding is not None and self.use_language:
            lang_token = self.lang_embed_proj(command_embedding).unsqueeze(1)  # (B, 1, hidden_dim)
            memory = torch.cat([memory, lang_token], dim=1)

        # Transformer encoder over all image (+ language) tokens
        memory = self.img_encoder(memory)  # (B, S_total, hidden_dim)
        return memory

    # ── Denoising step ────────────────────────────────────────────────────────

    def denoise(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising forward pass through the transformer decoder.

        Noisy action tokens are projected to hidden_dim, positional embeddings
        are added for sequence position, and the timestep embedding is broadcast-
        added so every action token knows the noise level. The transformer decoder
        then performs cross-attention against the image memory.

        Args:
            noisy_actions: (B, T, action_dim)
            timesteps: (B,) integer diffusion timesteps
            memory: (B, S, hidden_dim) encoded image tokens

        Returns:
            noise_pred: (B, T, action_dim)
        """
        B, T, _ = noisy_actions.shape

        # Project noisy actions to transformer width
        action_tokens = self.action_input_proj(noisy_actions)  # (B, T, hidden_dim)

        # Add learned positional embeddings for each action sequence position
        pos_ids = torch.arange(T, device=noisy_actions.device)
        action_tokens = action_tokens + self.action_pos_embed(pos_ids).unsqueeze(0)

        # Add timestep embedding, broadcast across the T dimension
        t_emb = self.timestep_mlp(self.timestep_embed(timesteps.float()))  # (B, hidden_dim)
        action_tokens = action_tokens + t_emb.unsqueeze(1)  # (B, T, hidden_dim)

        # Transformer decoder: action tokens (queries) cross-attend to image memory (keys/values)
        out = self.action_decoder(action_tokens, memory)  # (B, T, hidden_dim)

        return self.action_output_proj(out)  # (B, T, action_dim)

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        endoscope_img: torch.Tensor,
        lw_img: torch.Tensor,
        rw_img: torch.Tensor,
        current_pose,
        action=None,
        action_is_pad=None,
        command_text=None,
        return_policy_actions: bool = False,
        **kwargs,
    ):
        """Run the policy in training or inference mode.

        Training mode is selected by passing `action`. The method processes
        actions into the policy representation, adds DDPM noise, runs the
        transformer denoiser, and returns an MSE loss masked by action_is_pad.

        Inference mode runs the full iterative DDPM denoising loop starting
        from Gaussian noise and returns either raw policy actions or absolute
        robot commands.

        Args:
            endoscope_img: (B, C, H, W)
            lw_img: (B, C, H, W)
            rw_img: (B, C, H, W)
            current_pose: current robot pose for action conversion
            action: (B, T, action_dim) absolute action targets, triggers training mode
            action_is_pad: (B, T) bool mask, required when action is provided
            command_text: string or list of strings for language conditioning
            return_policy_actions: if True, return raw policy-space actions at inference

        Returns:
            Training: {"loss": Tensor, "diffusion_loss": Tensor}
            Inference: (B, action_dim) absolute robot actions or (B, T, action_dim) policy actions
        """
        endoscope_img, depth_img, lw_img, rw_img = self.preprocess_images(
            endoscope_img, lw_img, rw_img, use_augmentation=self.training
        )
        rgb_img_stack = torch.stack([endoscope_img, lw_img, rw_img], dim=1)
        batch_size = rgb_img_stack.shape[0]
        device = rgb_img_stack.device

        command_embedding = self._encode_command_text(command_text, device)

        # Encode observations once — used by both training and inference
        memory = self.encode_observations(rgb_img_stack, depth_img, command_embedding)

        if action is not None:
            # ── Training ──────────────────────────────────────────────────────
            action = action.to(device)
            if action_is_pad is None:
                raise ValueError("action_is_pad is required when actions are provided.")
            self._record_training_command_text(command_text)

            processed_actions = self.prepare_actions_for_training(
                current_pose, action, action_is_pad
            )
            processed_actions = processed_actions[:, : self.num_queries]  # (B, T, action_dim)
            action_is_pad = action_is_pad[:, : self.num_queries]          # (B, T)

            # Sample noise and add to clean actions
            noise = torch.randn_like(processed_actions)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config["num_train_timesteps"],
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_actions = self.noise_scheduler.add_noise(processed_actions, noise, timesteps)

            # Predict the noise
            noise_pred = self.denoise(noisy_actions, timesteps, memory)

            # MSE loss, masked to exclude padded action steps
            loss = F.mse_loss(noise_pred, noise, reduction="none")        # (B, T, action_dim)
            loss = (loss * ~action_is_pad.unsqueeze(-1)).mean()

            return {"loss": loss, "diffusion_loss": loss}

        # ── Inference: iterative DDPM denoising ───────────────────────────────
        actions = torch.randn(batch_size, self.num_queries, self.action_dim, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device)
            noise_pred = self.denoise(actions, t_batch, memory)
            actions = self.noise_scheduler.step(noise_pred, t, actions).prev_sample

        if return_policy_actions:
            return actions
        return self.postprocess_actions(actions, current_pose)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @staticmethod
    def _serialize_img_resize_cfg(img_resize_cfg: DictConfig) -> object:
        return OmegaConf.to_container(img_resize_cfg, resolve=True, throw_on_missing=True)

    @staticmethod
    def _deserialize_img_resize_cfg(serialized: object) -> DictConfig:
        if isinstance(serialized, DictConfig):
            return serialized
        restored = OmegaConf.create(serialized)
        if not isinstance(restored, DictConfig):
            raise TypeError("Checkpoint img_resize_cfg must deserialize to a mapping.")
        return restored

    def _serialize_policy_config(self) -> dict[str, object]:
        policy_config = super()._serialize_policy_config()
        policy_config.update(
            {
                "use_language": self.use_language,
                "language_encoder": self.language_encoder,
                "use_depth": self.use_depth,
                "num_queries": self.num_queries,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        return policy_config

    def _serialize_checkpoint_metadata(self) -> dict[str, object]:
        return {"training_text_conditionings": list(self.training_text_conditionings)}

    def _restore_checkpoint_metadata(self, model_dict: dict[str, object]) -> None:
        self.training_text_conditionings = list(
            model_dict.get("training_text_conditionings", [])
        )
        serialized = model_dict.get("img_resize_cfg")
        policy_config = model_dict.get("policy_config")
        if serialized is None and isinstance(policy_config, (dict, DictConfig)):
            serialized = policy_config.get("img_resize_cfg")
        if serialized is not None:
            self.img_resize_cfg = self._deserialize_img_resize_cfg(serialized)