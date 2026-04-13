from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from srth_new.low_level_policy.inference.inference import LowLevelPolicy
from srth_new.low_level_policy.utils import resolve_device, set_seed


@hydra.main(
    version_base=None,
    config_path="../../../conf/low_level_policy",
    config_name="inference",
)
def main(cfg: DictConfig) -> None:
    device = resolve_device(str(cfg.device))
    checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint_path)))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    policy = instantiate(cfg.policy).to(device)
    policy.load_checkpoint(checkpoint_path, map_location=device)
    policy.eval()

    low_level_policy = LowLevelPolicy(
        policy=policy,
        prediction_frequency_hz=cfg.prediction_frequency_hz,
        sleep_rate=cfg.sleep_rate
    )
    low_level_policy.run()


if __name__ == "__main__":
    main()
