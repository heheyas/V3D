import json
import torch
from scene import Scene
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from mediapy import write_video
from tqdm import tqdm
from einops import rearrange
from utils.camera_utils import get_uniform_poses
from mediapy import write_image


@torch.no_grad()
def render_spiral(dataset, opt, pipe, model_path):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    views = []
    alphas = []
    for view_cam in tqdm(viewpoint_stack):
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(view_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        views.append(image)
        alphas.append(render_pkg["alpha"])
    views = torch.stack(views)
    alphas = torch.stack(alphas)

    png_images = (
        (torch.cat([views, alphas], dim=1).clamp(0.0, 1.0) * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    png_images = rearrange(png_images, "t c h w -> t h w c")

    poses = get_uniform_poses(
        dataset.num_frames, dataset.radius, dataset.elevation, opengl=True
    )
    camera_angle_x = np.deg2rad(dataset.fov)
    name = Path(dataset.model_path).stem
    meta_dir = Path(f"blenders/{name}")
    meta_dir.mkdir(exist_ok=True, parents=True)
    meta = {}
    meta["camera_angle_x"] = camera_angle_x
    meta["frames"] = []
    for idx, (pose, image) in enumerate(zip(poses, png_images)):
        this_frames = {}
        this_frames["file_path"] = f"{idx:06d}"
        this_frames["transform_matrix"] = pose.tolist()
        meta["frames"].append(this_frames)
        write_image(meta_dir / f"{idx:06d}.png", image)

    with open(meta_dir / "transforms_train.json", "w") as f:
        json.dump(meta, f, indent=4)
    with open(meta_dir / "transforms_val.json", "w") as f:
        json.dump(meta, f, indent=4)
    with open(meta_dir / "transforms_test.json", "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)
    lp = lp.extract(args)
    fake_image = Image.fromarray(np.zeros([512, 512, 3], dtype=np.uint8))
    lp.images = [fake_image] * args.num_frames

    # Initialize system state (RNG)
    render_spiral(
        lp,
        op.extract(args),
        pp.extract(args),
        model_path=args.model_path,
    )
