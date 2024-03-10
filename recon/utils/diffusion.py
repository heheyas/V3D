import torch
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

from scripts.demo.streamlit_helpers import (
    load_model_from_config,
    get_sampler,
    get_batch,
    do_sample,
)


def load_config_and_model(ckpt: Path):
    if (ckpt.parent.parent / "configs").exists():
        config_path = list((ckpt.parent.parent / "configs").glob("*-project.yaml"))[0]
    else:
        config_path = list(
            (ckpt.parent.parent.parent / "configs").glob("*-project.yaml")
        )[0]

    config = OmegaConf.load(config_path)

    model, msg = load_model_from_config(config, ckpt)

    return config, model


def load_sampler(sampler_cfg):
    return get_sampler(**sampler_cfg)


def load_batch():
    pass


class DiffusionEngine:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def sample(self):
        pass
