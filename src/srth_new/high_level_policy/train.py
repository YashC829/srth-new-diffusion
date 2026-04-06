import hydra
from omegaconf import DictConfig

def load_dataloaders():
    pass

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pass