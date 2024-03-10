import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from einops import rearrange
from rembg import remove, new_session

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from datasets.ortho import (
    inv_RT,
    camNormal2worldNormal,
    RT_opengl2opencv,
    normal_opengl2opencv,
)
from utils.dpt import DPT


def blender2midas(img):
    """Blender: rub
    midas: lub
    """
    img[..., 0] = -img[..., 0]
    img[..., 1] = -img[..., 1]
    img[..., -1] = -img[..., -1]
    return img


def midas2blender(img):
    """Blender: rub
    midas: lub
    """
    img[..., 0] = -img[..., 0]
    img[..., 1] = -img[..., 1]
    img[..., -1] = -img[..., -1]
    return img


class BlenderDatasetBase:
    def setup(self, config, split):
        self.config = config
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        dpt = DPT(device=self.rank, mode="normal")

        self.directions = []
        with open(
            os.path.join(self.config.root_dir, self.config.scene, f"transforms.json"),
            "r",
        ) as f:
            meta = json.load(f)

        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        if "img_wh" in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif "img_downscale" in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        # self.near, self.far = self.config.near_plane, self.config.far_plane
        _session = new_session()
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, frame in enumerate(meta["frames"]):
            c2w = torch.from_numpy(np.array(frame["transform_matrix"])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(
                self.config.root_dir,
                self.config.scene,
                f"{frame['file_path']}",
            )
            img = Image.open(img_path)
            img = remove(img, session=_session)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
            fx = frame["fl_x"]
            fy = frame["fl_y"]
            cx = frame["cx"]
            cy = frame["cy"]

            self.all_fg_masks.append(img[..., -1])  # (h, w)
            self.all_images.append(img[..., :3])

            self.directions.append(get_ray_directions(self.w, self.h, fx, fy, cx, cy))

        self.all_c2w, self.all_images, self.all_fg_masks = (
            torch.stack(self.all_c2w, dim=0).float().to(self.rank),
            torch.stack(self.all_images, dim=0).float().to(self.rank),
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank),
        )

        self.normals = dpt(self.all_images)

        self.all_masks = self.all_fg_masks.cpu().numpy() > 0.1

        self.normals = self.normals * 2.0 - 1.0
        self.normals = midas2blender(self.normals).cpu().numpy()
        # self.normals = self.normals.cpu().numpy()
        self.normals[..., 0] *= -1
        self.normals[~self.all_masks] = [0, 0, 0]
        normals = rearrange(self.normals, "b h w c -> b c h w")
        normals = normals * 0.5 + 0.5
        normals = torch.from_numpy(normals)
        save_image(make_grid(normals, nrow=4), "tmp/normals.png")
        # exit(0)

        (
            self.all_poses,
            self.all_normals,
            self.all_normals_world,
            self.all_w2cs,
            self.all_color_masks,
        ) = ([], [], [], [], [])

        for c2w_opengl, normal in zip(self.all_c2w.cpu().numpy(), self.normals):
            RT_opengl = inv_RT(c2w_opengl)
            RT_opencv = RT_opengl2opencv(RT_opengl)
            c2w_opencv = inv_RT(RT_opencv)
            self.all_poses.append(c2w_opencv)
            self.all_w2cs.append(RT_opencv)
            normal = normal_opengl2opencv(normal)
            normal_world = camNormal2worldNormal(inv_RT(RT_opencv)[:3, :3], normal)
            self.all_normals.append(normal)
            self.all_normals_world.append(normal_world)

        self.directions = torch.stack(self.directions).to(self.rank)
        self.origins = self.directions
        self.all_poses = np.stack(self.all_poses)
        self.all_normals = np.stack(self.all_normals)
        self.all_normals_world = np.stack(self.all_normals_world)
        self.all_w2cs = np.stack(self.all_w2cs)

        self.all_c2w = torch.from_numpy(self.all_poses).float().to(self.rank)
        self.all_images = self.all_images.to(self.rank)
        self.all_fg_masks = self.all_fg_masks.to(self.rank)
        self.all_rgb_masks = self.all_fg_masks.to(self.rank)
        self.all_normals_world = (
            torch.from_numpy(self.all_normals_world).float().to(self.rank)
        )

        # normals = rearrange(self.all_normals_world, "b h w c -> b c h w")
        # normals = normals * 0.5 + 0.5
        # # normals = torch.from_numpy(normals)
        # save_image(make_grid(normals, nrow=4), "tmp/normals_world.png")
        # # exit(0)

        # # normals = (normals + 1) / 2.0
        # # for debug
        # index = [0, 9]
        # self.all_poses = self.all_poses[index]
        # self.all_c2w = self.all_c2w[index]
        # self.all_normals_world = self.all_normals_world[index]
        # self.all_w2cs = self.all_w2cs[index]
        # self.rgb_masks = self.all_rgb_masks[index]
        # self.fg_masks = self.all_fg_masks[index]
        # self.all_images = self.all_images[index]
        # self.directions = self.directions[index]
        # self.origins = self.origins[index]

        # images = rearrange(self.all_images, "b h w c -> b c h w")
        # normals = rearrange(normals, "b h w c -> b c h w")
        # save_image(make_grid(images, nrow=4), "tmp/images.png")
        # save_image(make_grid(normals, nrow=4), "tmp/normals.png")
        # breakpoint()

        # self.normals = self.normals * 2.0 - 1.0


class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register("videonvs-scene")
class VideoNVSScene(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = BlenderIterableDataset(
                self.config, self.config.train_split
            )
        if stage in [None, "fit", "validate"]:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, "test"]:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, "predict"]:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
