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
from mediapy import read_video
from pathlib import Path
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


def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    opengl=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    # opencv to opengl
    if opengl:
        c2w[..., 1:3] *= -1

    return c2w


def get_uniform_poses(num_frames, radius, elevation, opengl=False):
    T = num_frames
    azimuths = np.deg2rad(np.linspace(0, 360, T + 1)[:T])
    elevations = np.full_like(azimuths, np.deg2rad(elevation))
    cam_dists = np.full_like(azimuths, radius)

    campos = np.stack(
        [
            cam_dists * np.cos(elevations) * np.cos(azimuths),
            cam_dists * np.cos(elevations) * np.sin(azimuths),
            cam_dists * np.sin(elevations),
        ],
        axis=-1,
    )

    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    poses = []
    for t in range(T):
        poses.append(get_c2w_from_up_and_look_at(up, center, campos[t], opengl=opengl))

    return np.stack(poses, axis=0)


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

        # with open(
        #     os.path.join(
        #         self.config.root_dir, self.config.scene, f"transforms_train.json"
        #     ),
        #     "r",
        # ) as f:
        #     meta = json.load(f)

        # if "w" in meta and "h" in meta:
        #     W, H = int(meta["w"]), int(meta["h"])
        # else:
        #     W, H = 800, 800
        frames = read_video(Path(self.config.root_dir) / f"{self.config.scene}")
        rembg_session = new_session()
        num_frames, H, W = frames.shape[:3]

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

        self.focal = 0.5 * w / math.tan(0.5 * np.deg2rad(60))  # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            self.w, self.h, self.focal, self.focal, self.w // 2, self.h // 2
        ).to(
            self.rank
        )  # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        radius = 2.0
        elevation = 0.0
        poses = get_uniform_poses(num_frames, radius, elevation, opengl=True)
        for i, (c2w, frame) in enumerate(zip(poses, frames)):
            c2w = torch.from_numpy(np.array(c2w)[:3, :4])
            self.all_c2w.append(c2w)

            img = Image.fromarray(frame)
            img = remove(img, session=rembg_session)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1])  # (h, w)
            self.all_images.append(img[..., :3])

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
        # save_image(make_grid(normals, nrow=4), "tmp/normals.png")
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

        self.directions = torch.stack([self.directions] * len(self.all_images))
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


@datasets.register("v3d")
class BlenderDataModule(pl.LightningDataModule):
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
