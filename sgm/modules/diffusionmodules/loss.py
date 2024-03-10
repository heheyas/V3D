from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        return_model_output: bool = False,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        # for video diffusion
        if "num_video_frames" in batch:
            num_frames = batch["num_video_frames"]
            for k in ["crossattn", "concat"]:
                cond[k] = repeat(cond[k], "b ... -> b t ...", t=num_frames)
                cond[k] = rearrange(cond[k], "b t ... -> (b t) ...", t=num_frames)
        return self._forward(network, denoiser, cond, input, batch, return_model_output)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        return_model_output: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        if not return_model_output:
            return self.get_loss(model_output, input, w)
        else:
            return self.get_loss(model_output, input, w), model_output

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")


class StandardDiffusionLossWithPixelNeRFLoss(StandardDiffusionLoss):
    def __init__(
        self,
        sigma_sampler_config: Dict,
        loss_weighting_config: Dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0,
        batch2model_keys: str | List[str] | None = None,
        pixelnerf_loss_weight: float = 1.0,
        pixelnerf_loss_type: str = "l2",
    ):
        super().__init__(
            sigma_sampler_config,
            loss_weighting_config,
            loss_type,
            offset_noise_level,
            batch2model_keys,
        )
        self.pixelnerf_loss_weight = pixelnerf_loss_weight
        self.pixelnerf_loss_type = pixelnerf_loss_type

    def get_pixelnerf_loss(self, model_output, target):
        if self.pixelnerf_loss_type == "l2":
            return torch.mean(
                ((model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.pixelnerf_loss_type == "l1":
            return torch.mean(
                ((model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.pixelnerf_loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        return_model_output: bool = False,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch, return_model_output)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        return_model_output: bool = False,
    ) -> Tuple[torch.Tensor | Dict]:
        loss = super()._forward(
            network, denoiser, cond, input, batch, return_model_output
        )
        pixelnerf_loss = self.get_pixelnerf_loss(
            cond["rgb"], batch["pixelnerf_input"]["rgb"]
        )

        if not return_model_output:
            return loss + self.pixelnerf_loss_weight * pixelnerf_loss
        else:
            return loss[0] + self.pixelnerf_loss_weight * pixelnerf_loss, loss[1]
