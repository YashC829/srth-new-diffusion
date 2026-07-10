import logging
from typing import List, Literal, Optional

from collections import deque
from copy import deepcopy
import torch
from matplotlib import colormaps as cm
from ultralytics import YOLO
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torchvision import transforms
from ultralytics.data.augment import LetterBox

from srth_new.general import constants
from srth_new.general.third_party.EndoSynth.endosynth.models import (
    load as load_depth_model,
)
from srth_new.general.utils.lang_encoding import (
    encode_text,
    initialize_model_and_tokenizer,
)
from srth_new.low_level_policy.dataset.img_aug import ImageAug
from srth_new.low_level_policy.models.detr.models.backbone import build_image_backbone
from srth_new.low_level_policy.models.detr.models.detr_vae_utils import build_encoder
from srth_new.low_level_policy.models.detr.models.transformer import build_transformer
from srth_new.low_level_policy.models.detr.models.detr_vae import DETRVAE
from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy

log = logging.getLogger(__name__)

def depth2rgb(x: torch.Tensor, dmin: float, dmax: float) -> torch.Tensor:
    """
    Map depth to RGB using RdYlBu:
    low depth → red, high depth → blue.

    Input:
        x: Tensor of shape [B, 1, H, W]

    Output:
        Tensor of shape [B, 3, H, W]
    """
    cmap = cm.get_cmap("RdYlBu")

    span = max(dmax - dmin, 1e-12)

    # Normalize to [0, 1]
    t = (torch.clamp(x, dmin, dmax) - dmin) / span

    # Remove channel dimension for colormap
    # [B, 1, H, W] -> [B, H, W]
    t_np = t.squeeze(1).detach().cpu().numpy()

    # Apply matplotlib colormap
    # Output shape: [B, H, W, 4]
    rgba = cmap(t_np)

    # Keep RGB only and convert to torch
    # [B, H, W, 3]
    rgb = torch.from_numpy(rgba[..., :3]).to(
        device=x.device,
        dtype=torch.float32,
    )

    # Convert to channel-first
    # [B, H, W, 3] -> [B, 3, H, W]
    rgb = rgb.permute(0, 3, 1, 2)

    return torch.clamp(rgb * 255, 0, 255).to(torch.uint8)

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


class ACTKPPolicy(DVRKPolicy):
    """ACT policy wrapper with optional depth and action-history conditioning.

    This class layers project-specific behavior around the DETR/ACT backbone:

    - builds the ACT model and optimizer from the DETR configuration
    - optionally builds a depth model and depth backbone
    - optionally conditions the policy on action history shaped
      `(B, history_chunk_size, action_dim)`
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
        history_chunk_size: int,
        history_num_tokens: int,
        history_num_layers: int,
        history_num_heads: int,
        action_dim: int,
        kl_weight: float,
        use_language: bool,
        language_encoder: str,
        merge_recovery_phases: bool,
        action_mode: Literal["hybrid_relative", "ego", "relative_endoscope"],
        norm_scheme: Literal["std", "min_max"],
        img_resize_cfg: DictConfig,
        img_backbone_cfg: DictConfig,
        transformer_cfg: DictConfig,
        encoder_cfg: DictConfig,
        img_aug_cfg: DictConfig,
        use_depth: bool,
        use_wrist_cams: bool,

        # these args are specific to using keypoints for conditioning
        kp_predictor_ckpt: str,
        use_kp_pred_model_during_training: bool
    ):
        """Initialize the policy, optimizer, and optional conditioning modules."""
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
        self.use_depth = use_depth
        self.use_wrist_cams = use_wrist_cams
        self.history_chunk_size = history_chunk_size
        self.use_history = history_chunk_size > 0
        self.merge_recovery_phases = merge_recovery_phases

        # create buffers for storing the prediction history
        self.action_history_buffer = deque(
            [torch.zeros(action_dim, device="cuda") for _ in range(history_chunk_size)],
            maxlen=history_chunk_size
        )
        self.action_history_is_pad_buffer = deque(
            [torch.tensor(True, device="cuda") for _ in range(history_chunk_size)],
            maxlen=history_chunk_size
        )

        # Build image augmentation pipeline.
        self.img_aug_dict = self._build_img_aug_dict(img_aug_cfg)

        # Optional depth model.
        self.MAX_DEPTH_VAL = 0.3
        self.depth_model = load_depth_model("dav2") if self.use_depth else None

        # if we don't want to use wrist cameras, remove them from the camera names
        if not use_wrist_cams:
            camera_names = ["left"]

        # Build image backbones.
        img_backbones = []
        for _ in range(len(camera_names)):
            img_backbone = build_image_backbone(**img_backbone_cfg)  # type: ignore
            img_backbones.append(img_backbone)

        depth_backbone = (
            build_image_backbone(**img_backbone_cfg) if self.use_depth else None
        )  # type: ignore

        transformer = build_transformer(transformer_cfg)
        encoder = build_encoder(encoder_cfg)

        self.model = DETRVAE(
            backbones=img_backbones,
            transformer=transformer,
            encoder=encoder,
            state_dim=action_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            depth_backbone=depth_backbone,
            history_chunk_size=history_chunk_size,
            history_num_tokens=history_num_tokens,
            history_num_layers=history_num_layers,
            history_num_heads=history_num_heads,
            use_language=use_language,
            use_film="film" in img_backbone_cfg.backbone_type,
            use_keypoints=True
        )

        self.optimizer = torch.optim.AdamW(
            self._get_param_dict(self.model, img_backbone_cfg),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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
        self.num_queries = self.model.num_queries  # type: ignore

        # we have to store the YOLO model in this way. if we don't, calling
        # ACTKPPolicy.train() will error out because it automatically attempts
        # to call self.kp_model.train(True), which is not compatible
        object.__setattr__(self, "kp_model", YOLO(kp_predictor_ckpt))
        self.use_kp_pred_model_during_training = use_kp_pred_model_during_training

    def _build_img_aug_dict(self, cfg: DictConfig):
        aug_dict = {}

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

        texts = self._normalize_command_text(command_text, self.merge_recovery_phases)

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
    def _normalize_command_text(command_text, merge_recovery_phases: bool) -> list[str]:

        # remove the "_recovery" suffix if merging recovery phases with their
        # respective "standard" phases
        if merge_recovery_phases:
            command_text = command_text.replace(" recovery", "")

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
            self._normalize_command_text(command_text, self.merge_recovery_phases)
        )

    def _serialize_policy_config(self) -> dict[str, object]:
        policy_config = super()._serialize_policy_config()
        policy_config.update(
            {
                "use_language": self.use_language,
                "language_encoder": self.language_encoder,
                "use_depth": self.use_depth,
                "use_history": self.use_history,
                "history_chunk_size": self.history_chunk_size,
            }
        )
        return policy_config

    def _serialize_checkpoint_metadata(self) -> dict[str, object]:
        return {
            "training_text_conditionings": list(self.training_text_conditionings),
        }

    def _restore_checkpoint_metadata(self, model_dict: dict[str, object]) -> None:
        self.training_text_conditionings = list(
            model_dict.get("training_text_conditionings", [])  # type: ignore
        )

        serialized_img_resize_cfg = model_dict.get("img_resize_cfg")
        policy_config = model_dict.get("policy_config")

        if serialized_img_resize_cfg is None and isinstance(
            policy_config,
            (dict, DictConfig),
        ):
            serialized_img_resize_cfg = policy_config.get("img_resize_cfg")

        if serialized_img_resize_cfg is not None:
            self.img_resize_cfg = self._deserialize_img_resize_cfg(
                serialized_img_resize_cfg
            )

    def _get_depth(self, img: torch.Tensor) -> torch.Tensor:
        """Given an RGB image, generate the depth map for the image."""
        if self.depth_model is None:
            return None

        return self.depth_model.infer_tensor(img)

    def preprocess_images(
        self,
        endoscope_img: torch.Tensor,
        lw_img: Optional[torch.Tensor] = None,
        rw_img: Optional[torch.Tensor] = None,
        use_augmentation: bool = False,
    ):
        """Resize images, optionally compute depth, and apply augmentation.

        Returns:
            endo_processed: normalized endoscope RGB image
            depth_processed: normalized depth image or None
            lw_processed: normalized left wrist RGB image
            rw_processed: normalized right wrist RGB image
        """
        if self.use_depth:
            depth_1d = self._get_depth(endoscope_img)
            depth_img = depth2rgb(depth_1d, constants.DEPTH_MIN, constants.DEPTH_MAX).to(torch.uint8)
            # for idx, depth_img_ in enumerate(depth_img):
            #     temp = depth_img_.detach().cpu().numpy()
            #     temp = temp.transpose(1, 2, 0)
            #     Image.fromarray(temp).save(f"depth_img_{idx}.png")
            #     temp = endoscope_img[idx].detach().cpu().numpy()
            #     temp = temp.transpose(1, 2, 0)
            #     Image.fromarray(temp).save(f"endo_img_{idx}.png")

        def resize_img(img, new_size: List):
            h_new, w_new = new_size[0], new_size[1]
            return F.interpolate(
                img,
                size=(h_new, w_new),
                mode="bilinear",
                align_corners=False,
            )

        endo_processed = (
            resize_img(endoscope_img.float(), self.img_resize_cfg["left"])
            .clamp(0, 255.0)
            .to(torch.uint8)
        )
        lw_processed = None; rw_processed = None
        if self.use_wrist_cams:
            if lw_img is None or rw_img is None:
                raise Exception(
                    "If using wrist cameras, must pass both left and right wrist "
                    "images to the policy forward function.")
            lw_processed = (
                resize_img(lw_img.float(), self.img_resize_cfg["left_wrist"])
                .clamp(0, 255.0)
                .to(torch.uint8)
            )
            rw_processed = (
                resize_img(rw_img.float(), self.img_resize_cfg["right_wrist"])
                .clamp(0, 255.0)
                .to(torch.uint8)
            )

        depth_processed = None
        if self.use_depth:
            depth_processed = resize_img(depth_img.float(), self.img_resize_cfg["left"])

        # Augment images.
        # The endoscope RGB image and depth image are augmented together when
        # depth is enabled so spatial alignment is preserved.
        if use_augmentation:
            if depth_processed is not None:
                endo_processed, depth_processed = self.img_aug_dict["endoscope_img"](
                    endo_processed,
                    depth_processed,
                    kinds=["image", "depth"],
                )
            else:
                endo_processed = self.img_aug_dict["endoscope_img"](
                    endo_processed,
                    apply_random_shift=False,
                )

            if self.use_wrist_cams:
                lw_processed = self.img_aug_dict["lw_img"](
                    lw_processed,
                    apply_random_shift=False,
                )
                rw_processed = self.img_aug_dict["rw_img"](
                    rw_processed,
                    apply_random_shift=False,
                )

        # Normalize RGB images with ImageNet mean/std.
        endo_processed = self.image_normalize(endo_processed / 255.0)
        
        if self.use_wrist_cams:
            lw_processed = self.image_normalize(lw_processed / 255.0)
            rw_processed = self.image_normalize(rw_processed / 255.0)

        # # Normalize depth with min/max normalization.
        # if self.use_depth:
        #     depth_processed = torch.clamp(
        #         depth_processed,
        #         min=0.0,
        #         max=self.MAX_DEPTH_VAL,
        #     )
        #     depth_processed = depth_processed / self.MAX_DEPTH_VAL

        return endo_processed, depth_processed, lw_processed, rw_processed
    
    def predict_keypoints(self, img: torch.Tensor):
        if img.dtype != torch.float32:
            raise Exception(
                "The YOLO kp prediction model expects the batched tensor to be normalized [0.0, 1.0]"
            )

        letterbox = LetterBox(new_shape=(640, 640), stride=32, auto=False)

        processed = []
        for im in img:  # im: [3, H, W]
            im_np = im.permute(1, 2, 0).detach().cpu().numpy()  # HWC
            im_lb = letterbox(image=im_np)
            processed.append(torch.from_numpy(im_lb).permute(2, 0, 1))

        img_640 = torch.stack(processed).to(img.device, dtype=img.dtype)
        results = self.kp_model(img_640)
        return results

    def forward(
        self,
        endoscope_img: torch.Tensor,
        lw_img: torch.Tensor,
        rw_img: torch.Tensor,
        current_pose,
        action=None,
        action_is_pad=None,
        command_text=None,
        action_history: Optional[torch.Tensor] = None,
        action_history_is_pad: Optional[torch.Tensor] = None,
        return_policy_actions: bool = False,
        max_inference_chunk_size: int = 100000,
        affordance_kp: Optional[str] = None,
        tool_kp: Optional[str] = None,
        **kwargs
    ):
        """Run the policy in training or inference mode.

        Training mode is selected by passing `actions`. In that case the method
        converts absolute actions into the policy representation, runs the ACT
        model, and returns a loss dictionary containing L1, KL, and total loss.

        In inference mode, the method samples from the ACT prior, predicts a
        sequence of policy actions, and either returns those raw policy-space
        predictions or converts them into absolute robot commands.

        Args:
            endoscope_img: Endoscope RGB image batch.
            lw_img: Left wrist RGB image batch.
            rw_img: Right wrist RGB image batch.
            current_pose: Current robot pose used for action conversion.
            actions: Optional absolute action targets. Supplying this switches
                the method into training mode.
            is_pad: Optional padding mask aligned with `actions`.
            command_text: Optional command string or batch of strings for
                language-conditioned policies.
            action_history: Optional policy-space action history shaped
                `(B, history_chunk_size, action_dim)`.
            action_history_is_pad: Optional padding mask shaped
                `(B, history_chunk_size)`.
            return_policy_actions: In inference mode, return the raw policy
                action tensor instead of absolute robot actions.

        Returns:
            In training mode, a dictionary of loss tensors. In inference mode,
            either a tensor of predicted policy actions or a tensor of absolute
            robot actions.
        """
        endoscope_img_orig = deepcopy(endoscope_img)
        endoscope_img, depth_img, lw_img_, rw_img_ = self.preprocess_images(
            endoscope_img,
            lw_img,
            rw_img,
            use_augmentation=self.training,
        )

        if self.use_wrist_cams:
            assert lw_img_ is not None and rw_img_ is not None
            rgb_img_stack = torch.stack([endoscope_img, lw_img_, rw_img_], dim=1)
        else:
            rgb_img_stack = endoscope_img.unsqueeze(1)

        env_state = None
        batch_size = rgb_img_stack.shape[0]

        # Since the dVRK is inaccurate in an absolute setting, the model qpos is
        # zeroed so absolute pose does not directly influence the network.
        model_qpos = torch.zeros(
            (batch_size, self.state_dim),
            dtype=rgb_img_stack.dtype,
            device=rgb_img_stack.device,
        )

        command_embedding = self._encode_command_text(
            command_text,
            rgb_img_stack.device,
        )

        processed_history = None

        if action is not None:
            action = action.to(rgb_img_stack.device)
            if action_is_pad is None:
                raise ValueError("is_pad is required when actions are provided.")

            # Keep track of training command text strings so they can be saved in
            # checkpoint metadata.
            self._record_training_command_text(command_text)

            processed_actions = self.prepare_actions_for_training(
                current_pose, action, action_is_pad
            )
            processed_actions = processed_actions[:, : self.num_queries]
            action_is_pad = action_is_pad[:, : self.num_queries]

            if self.use_history:
                assert action_history is not None
                assert action_history_is_pad is not None
                processed_history = self.prepare_actions_for_training(
                    current_pose,
                    action_history,
                    action_history_is_pad,
                )
                processed_history = processed_history.to(rgb_img_stack.device)
                action_history_is_pad = action_history_is_pad.to(rgb_img_stack.device)

            if self.use_kp_pred_model_during_training:
                raise NotImplementedError()
                kp_results = self.predict_keypoints(endoscope_img_orig / 255.0)
                processed_keypoints = None
            else:
                processed_keypoints = torch.stack((affordance_kp, tool_kp), dim=1)
            
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos=model_qpos,
                image_stack=rgb_img_stack,
                env_state=env_state,
                actions=processed_actions,
                is_pad=action_is_pad,
                command_embedding=command_embedding,
                depth_image=depth_img,
                history=processed_history,
                history_is_pad=action_history_is_pad,
                keypoints=processed_keypoints
            )

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = {}
            all_l1 = F.l1_loss(processed_actions, a_hat, reduction="none")
            l1 = (all_l1 * ~action_is_pad.unsqueeze(-1)).mean()

            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight

            return loss_dict

        # Inference.

        # build the history tensor from the model prediction history
        if self.use_history:
            action_history = torch.stack(list(self.action_history_buffer), dim=0).unsqueeze(0)
            action_history_is_pad = torch.stack(list(self.action_history_is_pad_buffer), dim=0).unsqueeze(0)

        kp_results = self.predict_keypoints(endoscope_img_orig / 255.0)

        a_hat, _, (_, _) = self.model(
            qpos=model_qpos,
            image_stack=rgb_img_stack,
            env_state=env_state,
            command_embedding=command_embedding,
            depth_image=depth_img,
            history=action_history,
            history_is_pad=action_history_is_pad,
        )

        if max_inference_chunk_size < self.num_queries:
            a_hat = a_hat[:, :max_inference_chunk_size]

        # keep track of action prediction history
        if self.use_history:
            self.action_history_buffer.extend(a_hat.detach().squeeze(0))
            self.action_history_is_pad_buffer.extend(
                torch.zeros(a_hat.shape[1], dtype=torch.bool, device=a_hat.device)
            )

        # this will return the normalized actions in whatever action_mode space
        # was specified in the configuration
        if return_policy_actions:
            return a_hat

        # this will process the model outputs and provide the actions in the format
        # to send to the dVRK robot
        return self.postprocess_actions(a_hat, current_pose)

    def configure_optimizers(self):
        """Return the optimizer constructed alongside the ACT model."""
        return self.optimizer
