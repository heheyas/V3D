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


@torch.no_grad()
def render_spiral(dataset, opt, pipe, model_path):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    views = []
    for view_cam in tqdm(viewpoint_stack):
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(view_cam, gaussians, pipe, bg, scaling_modifier=0.1)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        views.append(image)
    views = torch.stack(views)

    write_video(
        f"./paper/specials/{Path(dataset.model_path).stem}.mp4",
        rearrange(views.cpu().numpy(), "t c h w -> t h w c"),
        fps=30,
    )


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
