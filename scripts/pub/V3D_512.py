import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
import tyro
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from mediapy import write_video
import rembg
from kiui.op import recenter
from safetensors.torch import load_file as load_safetensors
from typing import Any

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    ckpt_path: Optional[str] = None,
    min_cfg: Optional[float] = None,
    max_cfg: Optional[float] = None,
    sigma_max: Optional[float] = None,
):
    config = OmegaConf.load(config)

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if max_cfg is not None:
        config.model.params.sampler_config.params.guider_config.params.max_scale = (
            max_cfg
        )
    if min_cfg is not None:
        config.model.params.sampler_config.params.guider_config.params.min_scale = (
            min_cfg
        )
    if sigma_max is not None:
        print("Overriding sigma_max to ", sigma_max)
        config.model.params.sampler_config.params.discretization_config.params.sigma_max = (
            sigma_max
        )

    config.model.params.from_scratch = False

    if ckpt_path is not None:
        config.model.params.ckpt_path = str(ckpt_path)
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    return model, None


def sample_one(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    checkpoint_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 1,
    motion_bucket_id: int = 300,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 24,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    noise: torch.Tensor = None,
    save: bool = False,
    cached_model: Any = None,
    border_ratio: float = 0.3,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 2.5,
    sigma_max: float = None,
    ignore_alpha: bool = False,
):
    model_config = "scripts/pub/configs/V3D_512.yaml"
    num_frames = OmegaConf.load(
        model_config
    ).model.params.sampler_config.params.guider_config.params.num_frames
    print("Detected num_frames:", num_frames)
    num_steps = default(num_steps, 25)
    output_folder = default(output_folder, f"outputs/V3D_512")
    decoding_t = min(decoding_t, num_frames)

    sd = load_safetensors("./ckpts/svd_xt.safetensors")
    clip_model_config = OmegaConf.load("configs/embedder/clip_image.yaml")
    clip_model = instantiate_from_config(clip_model_config).eval()
    clip_sd = dict()
    for k, v in sd.items():
        if "conditioner.embedders.0" in k:
            clip_sd[k.replace("conditioner.embedders.0.", "")] = v
    clip_model.load_state_dict(clip_sd)
    clip_model = clip_model.to(device)

    ae_model_config = OmegaConf.load("configs/ae/video.yaml")
    ae_model = instantiate_from_config(ae_model_config).eval()
    encoder_sd = dict()
    for k, v in sd.items():
        if "first_stage_model" in k:
            encoder_sd[k.replace("first_stage_model.", "")] = v
    ae_model.load_state_dict(encoder_sd)
    ae_model = ae_model.to(device)

    if cached_model is None:
        model, filter = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
            ckpt_path=checkpoint_path,
            min_cfg=min_guidance_scale,
            max_cfg=max_guidance_scale,
            sigma_max=sigma_max,
        )
    else:
        model = cached_model
    torch.manual_seed(seed)

    need_return = True
    path = Path(input_path)
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        need_return = False
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_path in all_img_paths:
        with Image.open(input_path) as image:
            # if image.mode == "RGBA":
            #     image = image.convert("RGB")
            w, h = image.size

            if border_ratio > 0:
                if image.mode != "RGBA" or ignore_alpha:
                    image = image.convert("RGB")
                    image = np.asarray(image)
                    carved_image = rembg.remove(image)  # [H, W, 4]
                else:
                    image = np.asarray(image)
                    carved_image = image
                mask = carved_image[..., -1] > 0
                image = recenter(carved_image, mask, border_ratio=border_ratio)
                image = image.astype(np.float32) / 255.0
                if image.shape[-1] == 4:
                    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                print("Ignore border ratio")
            image = image.resize((512, 512))

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = clip_model(image)
        value_dict["cond_frames"] = ae_model.encode(image)
        value_dict["cond_frames"] += cond_aug * torch.randn_like(
            value_dict["cond_frames"]
        )
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device) if noise is None else noise
                randn = randn.to(device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                # writer = cv2.VideoWriter(
                #     video_path,
                #     cv2.VideoWriter_fourcc(*"MP4V"),
                #     fps_id + 1,
                #     (samples.shape[-1], samples.shape[-2]),
                # )

                frames = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                if save:
                    write_video(video_path, frames, fps=3)

                images = []
                for frame in frames:
                    images.append(Image.fromarray(frame))

                if need_return:
                    return images, model


if __name__ == "__main__":
    tyro.cli(sample_one)
