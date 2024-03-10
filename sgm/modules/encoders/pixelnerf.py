import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import numpy as np
from einops import rearrange, repeat, einsum

from .math_utils import get_ray_limits_box, linspace

from ...modules.diffusionmodules.openaimodel import Timestep


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, image):
        return image


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            # embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = repeat(x, "... C -> ... N C", N=self.num_freqs * 2)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = rearrange(embed, "... N C -> ... (N C)")
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed


class RayGenerator(torch.nn.Module):
    """
    from camera pose and intrinsics to ray origins and directions
    """

    def __init__(self):
        super().__init__()
        (
            self.ray_origins_h,
            self.ray_directions,
            self.depths,
            self.image_coords,
            self.rendering_options,
        ) = (None, None, None, None, None)

    def forward(self, cam2world_matrix, intrinsics, render_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        render_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """

        N, M = cam2world_matrix.shape[0], render_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(
            torch.meshgrid(
                torch.arange(
                    render_size, dtype=torch.float32, device=cam2world_matrix.device
                ),
                torch.arange(
                    render_size, dtype=torch.float32, device=cam2world_matrix.device
                ),
                indexing="ij",
            )
        )
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        y_cam = uv[:, :, 1].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (
            (
                x_cam
                - cx.unsqueeze(-1)
                + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
                - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
            )
            / fx.unsqueeze(-1)
            * z_cam
        )
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack(
            (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1
        )

        # NOTE: this should be named _blender2opencv
        _opencv2blender = (
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=cam2world_matrix.device,
            )
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )

        cam2world_matrix = torch.bmm(cam2world_matrix, _opencv2blender)

        world_rel_points = torch.bmm(
            cam2world_matrix, cam_rel_points.permute(0, 2, 1)
        ).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs


class RaySampler(torch.nn.Module):
    def __init__(
        self,
        num_samples_per_ray,
        bbox_length=1.0,
        near=0.5,
        far=10000.0,
        disparity=False,
    ):
        super().__init__()
        self.num_samples_per_ray = num_samples_per_ray
        self.bbox_length = bbox_length
        self.near = near
        self.far = far
        self.disparity = disparity

    def forward(self, ray_origins, ray_directions):
        if not self.disparity:
            t_start, t_end = get_ray_limits_box(
                ray_origins, ray_directions, 2 * self.bbox_length
            )
        else:
            t_start = torch.full_like(ray_origins, self.near)
            t_end = torch.full_like(ray_origins, self.far)
        is_ray_valid = t_end > t_start
        if torch.any(is_ray_valid).item():
            t_start[~is_ray_valid] = t_start[is_ray_valid].min()
            t_end[~is_ray_valid] = t_start[is_ray_valid].max()

        if not self.disparity:
            depths = linspace(t_start, t_end, self.num_samples_per_ray)
            depths += (
                torch.rand_like(depths)
                * (t_end - t_start)
                / (self.num_samples_per_ray - 1)
            )
        else:
            step = 1.0 / self.num_samples_per_ray
            z_steps = torch.linspace(
                0, 1 - step, self.num_samples_per_ray, device=ray_origins.device
            )
            z_steps += torch.rand_like(z_steps) * step
            depths = 1 / (1 / self.near * (1 - z_steps) + 1 / self.far * z_steps)
            depths = depths[..., None, None, None]

        return ray_origins[None] + ray_directions[None] * depths


class PixelNeRF(torch.nn.Module):
    def __init__(
        self,
        num_samples_per_ray: int = 128,
        feature_dim: int = 64,
        interp: str = "bilinear",
        padding: str = "border",
        disparity: bool = False,
        near: float = 0.5,
        far: float = 10000.0,
        use_feats_std: bool = False,
        use_pos_emb: bool = False,
    ) -> None:
        super().__init__()
        # self.positional_encoder = Timestep(3)  # TODO
        self.num_samples_per_ray = num_samples_per_ray
        self.ray_generator = RayGenerator()
        self.ray_sampler = RaySampler(
            num_samples_per_ray, near=near, far=far, disparity=disparity
        )  # TODO
        self.interp = interp
        self.padding = padding

        self.positional_encoder = PositionalEncoding()

        # self.feature_aggregator = nn.Linear(128, 129)  # TODO
        self.use_feats_std = use_feats_std
        self.use_pos_emb = use_pos_emb
        d_in = feature_dim
        if use_feats_std:
            d_in += feature_dim
        if use_pos_emb:
            d_in += self.positional_encoder.d_out
        self.feature_aggregator = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 129),
        )

        # self.decoder = nn.Linear(128, 131)  # TODO
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 131),
        )

    def project(self, ray_samples, source_c2ws, source_instrincs):
        # TODO: implement
        # S for number of source cameras
        # ray_samples: [B, N, H * W, N_sample, 3]
        # source_c2ws: [B, S, 4, 4]
        # source_intrinsics: [B, S, 3, 3]
        # return [B, S, N, H * W, N_sample, 2]
        S = source_c2ws.shape[1]
        B = ray_samples.shape[0]
        N = ray_samples.shape[1]
        HW = ray_samples.shape[2]
        ray_samples = repeat(
            ray_samples,
            "B N HW N_sample C -> B S N HW N_sample C",
            S=source_c2ws.shape[1],
        )
        padding = torch.ones((B, S, N, HW, self.num_samples_per_ray, 1)).to(ray_samples)
        ray_samples_homo = torch.cat([ray_samples, padding], dim=-1)
        source_c2ws = repeat(source_c2ws, "B S C1 C2 -> B S N 1 1 C1 C2", N=N)
        source_instrincs = repeat(source_instrincs, "B S C1 C2 -> B S N 1 1 C1 C2", N=N)
        source_w2c = source_c2ws.inverse()
        projected_samples = einsum(
            source_w2c, ray_samples_homo, "... i j, ... j -> ... i"
        )[..., :3]
        # NOTE: assumes opengl convention
        projected_samples = -1 * projected_samples[..., :2] / projected_samples[..., 2:]
        # NOTE: intrinsics here are normalized by resolution
        fx = source_instrincs[..., 0, 0]
        fy = source_instrincs[..., 1, 1]
        cx = source_instrincs[..., 0, 2]
        cy = source_instrincs[..., 1, 2]
        x = projected_samples[..., 0] * fx + cx
        # negative sign here is caused by opengl, F.grid_sample is consistent with openCV convention
        y = -projected_samples[..., 1] * fy + cy

        return torch.stack([x, y], dim=-1)

    def forward(
        self, image_feats, source_c2ws, source_intrinsics, c2ws, intrinsics, render_size
    ):
        # image_feats: [B S C H W]
        B = c2ws.shape[0]
        T = c2ws.shape[1]
        ray_origins, ray_directions = self.ray_generator(
            c2ws.reshape(-1, 4, 4), intrinsics.reshape(-1, 3, 3), render_size
        )  # [B * N, H * W, 3]
        # breakpoint()

        ray_samples = self.ray_sampler(
            ray_origins, ray_directions
        )  # [N_sample, B * N, H * W, 3]
        ray_samples = rearrange(ray_samples, "Ns (B N) HW C -> B N HW Ns C", B=B)

        projected_samples = self.project(ray_samples, source_c2ws, source_intrinsics)
        # # debug
        # p = projected_samples[:, :, 0, :, 0, :]
        # p = p.reshape(p.shape[0] * p.shape[1], *p.shape[2:])

        # breakpoint()

        # image_feats = repeat(image_feats, "B S C H W -> (B S N) C H W", N=T)
        image_feats = rearrange(image_feats, "B S C H W -> (B S) C H W")
        projected_samples = rearrange(
            projected_samples, "B S N HW Ns xy -> (B S) (N Ns) HW xy"
        )
        # make sure the projected samples are in the range of [-1, 1], as required by F.grid_sample
        joint = F.grid_sample(
            image_feats,
            projected_samples * 2.0 - 1.0,
            padding_mode=self.padding,
            mode=self.interp,
            align_corners=True,
        )
        # print("image_feats", image_feats.max(), image_feats.min())
        # print("samples", projected_samples.max(), projected_samples.min())
        joint = rearrange(
            joint,
            "(B S) C (N Ns) HW -> B S N HW Ns C",
            B=B,
            Ns=self.num_samples_per_ray,
        )

        reduced = torch.mean(joint, dim=1)  # reduce on source dimension
        if self.use_feats_std:
            if not joint.shape[1] == 1:
                reduced = torch.cat((reduced, joint.std(dim=1)), dim=-1)
            else:
                reduced = torch.cat((reduced, torch.zeros_like(reduced)), dim=-1)

        if self.use_pos_emb:
            reduced = torch.cat((reduced, self.positional_encoder(ray_samples)), dim=-1)
        reduced = self.feature_aggregator(reduced)

        feats, weights = reduced.split([reduced.shape[-1] - 1, 1], dim=-1)
        # feats: [B, N, H * W, N_samples, N_c]
        # weights: [B, N, H * W, N_samples, 1]
        weights = F.softmax(weights, dim=-2)

        feats = torch.sum(feats * weights, dim=-2)

        rgb, feats = self.decoder(feats).split([3, 128], dim=-1)

        rgb = F.sigmoid(rgb)
        rgb = rearrange(rgb, "B N (H W) C -> B N C H W", H=render_size)
        feats = rearrange(feats, "B N (H W) C -> B N C H W", H=render_size)

        # print(rgb.max(), rgb.min())
        # print(feats.max(), feats.min())

        return rgb, feats
