from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy


class DiffusionModel(DVRKPolicy):
    """Wraps a basic lerobot diffusion model-based policy."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        camera_names: List[str],
        num_queries: int,
        history_chunk_size: int,
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
        img_aug_cfg: DictConfig,
    ):
        pass
