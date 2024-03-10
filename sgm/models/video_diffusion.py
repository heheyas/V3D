import re
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from einops import rearrange, repeat

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
    video_frames_as_grid,
)


def flatten_for_video(input):
    return input.flatten()


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "frames",  # for video inputs
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        load_last_embedder: bool = False,
        from_scratch: bool = False,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        self.load_last_embedder = load_last_embedder
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, from_scratch)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def _load_last_embedder(self, original_state_dict):
        original_module_name = "conditioner.embedders.3"
        state_dict = dict()
        for k, v in original_state_dict.items():
            m = re.match(rf"^{original_module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict[m.group(1)] = v

        idx = -1
        for i in range(len(self.conditioner.embedders)):
            if isinstance(
                self.conditioner.embedders[i], VideoPredictionEmbedderWithEncoder
            ):
                idx = i

        print(f"Embedder [{idx}] is the frame encoder, make sure this is expected")

        self.conditioner.embedders[idx].load_state_dict(state_dict)

    def init_from_ckpt(
        self,
        path: str,
        from_scratch: bool = False,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        deleted_keys = []
        for k, v in self.state_dict().items():
            # resolve shape dismatch
            if k in sd:
                if v.shape != sd[k].shape:
                    del sd[k]
                    deleted_keys.append(k)

        if from_scratch:
            new_sd = {}
            for k in sd:
                if "first_stage_model" in k:
                    new_sd[k] = sd[k]
            sd = new_sd
            print(sd.keys())

        if len(deleted_keys) > 0:
            print(f"Deleted Keys: {deleted_keys}")

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        if len(deleted_keys) > 0:
            print(f"Deleted Keys: {deleted_keys}")

        if (len(missing) > 0 or len(unexpected) > 0) and self.load_last_embedder:
            # means we are loading from a checkpoint that has the old embedder (motion bucket id and fps id)
            print("Modified embedder to support 3d spiral video inputs")
            self._load_last_embedder(sd)

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        is_video_input = False
        bs = z.shape[0]
        if z.dim() == 5:
            is_video_input = True
            # for video diffusion
            z = rearrange(z, "b t c h w -> (b t) c h w")
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)

        if is_video_input:
            out = rearrange(out, "(b t) c h w -> b t c h w", b=bs)

        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.input_key == "latents":
            return x * self.scale_factor

        bs = x.shape[0]
        is_video_input = False
        if x.dim() == 5:
            is_video_input = True
            # for video diffusion
            x = rearrange(x, "b t c h w -> (b t) c h w")
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z

        # if is_video_input:
        #     z = rearrange(z, "(b t) c h w -> b t c h w", b=bs)

        return z

    def forward(self, x, batch):
        loss, model_output = self.loss_fn(
            self.model,
            self.denoiser,
            self.conditioner,
            x,
            batch,
            return_model_output=True,
        )
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean, "model_output": model_output}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        # TODO: move this shit to collate_fn in dataloader
        # if "fps_id" in batch:
        #     batch["fps_id"] = flatten_for_video(batch["fps_id"])
        # if "motion_bucket_id" in batch:
        #     batch["motion_bucket_id"] = flatten_for_video(batch["motion_bucket_id"])
        # if "cond_aug" in batch:
        #     batch["cond_aug"] = flatten_for_video(batch["cond_aug"])
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        # ## debug
        # x_recon = self.decode_first_stage(x)
        # video_frames_as_grid((batch["frames"][0] + 1.0) / 2.0, "./tmp/origin.jpg")
        # video_frames_as_grid((x_recon[0] + 1.0) / 2.0, "./tmp/recon.jpg")
        # ## debug
        batch["global_step"] = self.global_step
        # breakpoint()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        with torch.no_grad():
            if "model_output" in loss_dict:
                if batch_idx % 100 == 0:
                    if isinstance(self.logger, WandbLogger):
                        model_output = loss_dict["model_output"].detach()[
                            : batch["num_video_frames"]
                        ]
                        recons = (
                            (self.decode_first_stage(model_output) + 1.0) / 2.0
                        ).clamp(0.0, 1.0)
                        recon_grid = make_grid(recons, nrow=4)
                        self.logger.log_image(
                            key=f"train/model_output_recon",
                            images=[recon_grid],
                            step=self.global_step,
                        )
            del loss_dict["model_output"]

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[-2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:
                        # image
                        xc = x
                    else:
                        pass
                        # breakpoint()
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    # for video diffusions will be logging frames of a video
    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 1,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        # # debug
        # return {}
        # # debug
        assert "num_video_frames" in batch, "num_video_frames must be in batch"
        num_video_frames = batch["num_video_frames"]
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {"num_video_frames": num_video_frames}
        n = min(x.shape[0] // num_video_frames, N)
        sampling_kwargs["image_only_indicator"] = torch.cat(
            [batch["image_only_indicator"][:n]] * 2
        )

        N = min(x.shape[0] // num_video_frames, N) * num_video_frames
        x = x.to(self.device)[:N]
        # log["inputs"] = rearrange(x, "(b t) c h w -> b c h (t w)", t=num_video_frames)
        if self.input_key != "latents":
            log["inputs"] = x
        z = self.encode_first_stage(x)
        recon = self.decode_first_stage(z)
        # log["reconstructions"] = rearrange(
        #     recon, "(b t) c h w -> b c h (t w)", t=num_video_frames
        # )
        log["reconstructions"] = recon
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                if k == "vector":
                    end = N
                else:
                    end = n
                c[k], uc[k] = map(lambda y: y[k][:end].to(self.device), (c, uc))

        # for k in c:
        #     print(c[k].shape)

        for k in ["crossattn", "concat"]:
            c[k] = repeat(c[k], "b ... -> b t ...", t=num_video_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_video_frames)
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_video_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_video_frames)

        # for k in c:
        #     print(c[k].shape)
        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log
