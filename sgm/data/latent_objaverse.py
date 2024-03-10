import numpy as np
from pathlib import Path
from PIL import Image
import json
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from pytorch_lightning import LightningDataModule
from einops import rearrange


class LatentObjaverseSpiral(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        random_front=False,
        max_item=None,
        cond_aug_mean=-3.0,
        cond_aug_std=0.5,
        condition_on_elevation=False,
        **unused_kwargs,
    ):
        print("Using LVIS subset with precomputed Latents")
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.transform = transform

        self.latent_dir = Path("/mnt/vepfs/3Ddataset/render_results/latents512")

        self.ids = json.load(open("./assets/lvis_uids.json", "r"))
        self.n_views = 18
        valid_ids = []
        for idx in self.ids:
            if (self.root_dir / idx).exists():
                valid_ids.append(idx)
        self.ids = valid_ids
        print("=" * 30)
        print("Number of valid ids: ", len(self.ids))
        print("=" * 30)

        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        self.condition_on_elevation = condition_on_elevation

        if max_item is not None:
            self.ids = self.ids[:max_item]

            ## debug
            self.ids = self.ids * 10000
