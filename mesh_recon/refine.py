import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import kiui
from kiui.mesh import Mesh
import json
from pathlib import Path
import tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import trimesh
from mediapy import write_image, write_video
from einops import rearrange

from kiui.op import uv_padding, safe_normalize, inverse_sigmoid
from kiui.cam import orbit_camera, get_perspective

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from mesh import Mesh
import tyro


class Refiner(nn.Module):
    def __init__(self, mesh_filename, scene_dir, num_opt=4, lpips: float = 0.0) -> None:
        super().__init__()
        self.output_size = 512
        znear = 0.1
        zfar = 10
        self.mesh = Mesh.load_obj(mesh_filename)
        # self.mesh.v[..., 1], self.mesh.v[..., 2] = (
        #     self.mesh.v[..., 2],
        #     self.mesh.v[..., 1],
        # )
        self.glctx = dr.RasterizeGLContext()

        self.device = torch.device("cuda")
        self.lpips_meter = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(self.device)
        self.lpips = lpips

        self.scene_dir = Path(scene_dir)

        with open(self.scene_dir / "transforms_train.json", "r") as f:
            meta = json.load(f)

        fov = np.rad2deg(meta["camera_angle_x"])

        poses = []
        image_gt = []
        for f in meta["frames"]:
            poses.append(np.array(f["transform_matrix"]))
            image = Image.open(self.scene_dir / f"{f['file_path']}.png")
            image = to_tensor(image)
            image = image[:3] + 1 - image[3].unsqueeze(0)
            image_gt.append(image)

        self.poses = np.stack(poses)
        self.image_gt = torch.stack(image_gt).to(self.device)

        self.n_frames = len(self.poses)
        self.opt_frames = np.linspace(0, self.n_frames, num_opt + 1)[:num_opt].astype(
            int
        )
        print(self.opt_frames)

        # gs renderer
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(fov))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        self.proj_matrix[3, 2] = -(zfar * znear) / (zfar - znear)
        self.proj_matrix[2, 3] = 1

        self.glctx = dr.RasterizeGLContext()

        self.proj = torch.from_numpy(get_perspective(fov)).float().to(self.device)

        self.v = self.mesh.v.contiguous().float().to(self.device)
        self.f = self.mesh.f.contiguous().int().to(self.device)
        self.vc = self.mesh.vc.contiguous().float().to(self.device)
        # self.vt = self.mesh.vt
        # self.ft = self.mesh.ft

    def render_normal(self, pose):
        h = w = self.output_size

        v = self.v
        f = self.f

        if not hasattr(self.mesh, "vn") or self.mesh.vn is None:
            self.mesh.auto_normal()
        vc = self.mesh.vn.to(self.device)

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        vc = torch.einsum("ij, kj -> ki", pose[:3, :3].T, vc).contiguous()

        # get v_clip and render rgb
        v_cam = (
            torch.matmul(
                F.pad(v, pad=(0, 1), mode="constant", value=1.0), torch.inverse(pose).T
            )
            .float()
            .unsqueeze(0)
        )
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous()  # [1, H, W, 1]
        alpha = (
            dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0)
        )  # [H, W] important to enable gradients!

        # color, texc_db = dr.interpolate(
        #     self.vc.unsqueeze(0), rast, f, rast_db=rast_db, diff_attrs="all"
        # )
        color, texc_db = dr.interpolate(vc.unsqueeze(0), rast, f)
        color = dr.antialias(color, rast, v_clip, f)
        # image = torch.sigmoid(
        #     dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)
        # )  # [1, H, W, 3]

        image = color.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous()  # [3, H, W]
        image = (image + 1) / 2.0
        image = alpha * image + (1 - alpha)

        return image, alpha

    def render_mesh(self, pose, use_sigmoid=True):
        h = w = self.output_size

        v = self.v
        f = self.f
        if use_sigmoid:
            vc = torch.sigmoid(self.vc)
        else:
            vc = self.vc

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = (
            torch.matmul(
                F.pad(v, pad=(0, 1), mode="constant", value=1.0), torch.inverse(pose).T
            )
            .float()
            .unsqueeze(0)
        )
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous()  # [1, H, W, 1]
        alpha = (
            dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0)
        )  # [H, W] important to enable gradients!

        # color, texc_db = dr.interpolate(
        #     self.vc.unsqueeze(0), rast, f, rast_db=rast_db, diff_attrs="all"
        # )
        color, texc_db = dr.interpolate(vc.unsqueeze(0), rast, f)
        color = dr.antialias(color, rast, v_clip, f)
        # image = torch.sigmoid(
        #     dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)
        # )  # [1, H, W, 3]

        image = color.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous()  # [3, H, W]
        image = alpha * image + (1 - alpha)

        return image, alpha

    def refine_texture(self, texture_resolution: int = 512, iters: int = 5000):
        h = w = texture_resolution
        albedo = torch.ones(h * w, 3, device=self.device, dtype=torch.float32) * 0.5
        albedo = albedo.view(h, w, -1)
        vc_original = self.vc.clone()
        self.vc = nn.Parameter(inverse_sigmoid(vc_original)).to(self.device)

        optimizer = torch.optim.Adam(
            [
                {"params": self.vc, "lr": 1e-3},
            ]
        )

        pbar = tqdm.trange(iters)
        for i in pbar:
            index = np.random.choice(self.opt_frames)
            pose = self.poses[index]
            image_gt = self.image_gt[index]

            image_pred, _ = self.render_mesh(pose)

            if i % 1000 == 0:
                save_image(image_pred, f"tmp/image_pred_{i}.png")
                save_image(image_gt, f"tmp/image_gt_{i}.png")

            loss = F.mse_loss(image_pred, image_gt)
            if self.lpips > 0.0:
                loss += (
                    self.lpips_meter(
                        image_gt.clamp(0, 1)[None], image_pred.clamp(0, 1)[None]
                    )
                    * self.lpips
                )
            # * 10.0

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss.item():.6f}")

    @torch.no_grad()
    def render_spiral(self):
        images = []
        for i, pose in enumerate(self.poses):
            image, _ = self.render_mesh(pose, use_sigmoid=False)
            images.append(image)

        images = torch.stack(images)
        images = images.cpu().numpy()
        images = rearrange(images, "b c h w -> b h w c")
        write_video(f"renders/{self.scene_dir.stem}.mp4", images, fps=3)

    @torch.no_grad()
    def render_normal_spiral(self):
        images = []
        for i, pose in enumerate(self.poses):
            image, _ = self.render_normal(pose)
            images.append(image)

        images = torch.stack(images)
        images = images.cpu().numpy()
        images = rearrange(images, "b c h w -> b h w c")
        write_video(f"renders/{self.scene_dir.stem}_normal.mp4", images, fps=3)

    def export(self, filename):
        mesh = trimesh.Trimesh(
            vertices=self.mesh.v.cpu().numpy(),
            faces=self.mesh.f.cpu().numpy(),
            vertex_colors=torch.sigmoid(self.vc.detach()).cpu().numpy(),
        )
        self.vc.data = torch.sigmoid(self.vc.detach())
        trimesh.repair.fix_inversion(mesh)
        mesh.export(filename)


def do_refine(
    mesh: str,
    scene: str,
    num_opt: int = 4,
    iters: int = 2000,
    skip_refine: bool = False,
    render_normal: bool = True,
    lpips: float = 1.0,
):
    refiner = Refiner(
        # "tmp/corgi_size_1.obj",
        mesh,
        scene,
        num_opt=num_opt,
        lpips=lpips,
    )
    if not skip_refine:
        refiner.refine_texture(512, iters)
        save_path = Path("refined") / f"{Path(scene).stem}.obj"
        refiner.export(str(save_path))

    refiner.render_spiral()
    if render_normal:
        refiner.render_normal_spiral()


if __name__ == "__main__":
    tyro.cli(do_refine)
