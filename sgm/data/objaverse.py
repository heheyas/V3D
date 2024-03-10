import numpy as np
from pathlib import Path
from PIL import Image
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.transforms.functional import to_tensor
from pytorch_lightning import LightningDataModule
from einops import rearrange


def read_camera_matrix_single(json_file):
    # for gobjaverse
    with open(json_file, "r", encoding="utf8") as reader:
        json_content = json.load(reader)

    # negative sign for opencv to opengl
    camera_matrix = torch.zeros(3, 4)
    camera_matrix[:3, 0] = torch.tensor(json_content["x"])
    camera_matrix[:3, 1] = -torch.tensor(json_content["y"])
    camera_matrix[:3, 2] = -torch.tensor(json_content["z"])
    camera_matrix[:3, 3] = torch.tensor(json_content["origin"])
    """
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    """

    return camera_matrix


def read_camera_instrinsics_single(json_file, h: int, w: int, scale: float = 1.0):
    with open(json_file, "r", encoding="utf8") as reader:
        json_content = json.load(reader)

    h = int(h * scale)
    w = int(w * scale)

    y_fov = json_content["y_fov"]
    x_fov = json_content["x_fov"]

    fy = h / 2 / np.tan(y_fov / 2)
    fx = w / 2 / np.tan(x_fov / 2)

    cx = w // 2
    cy = h // 2

    intrinsics = torch.tensor(
        [
            [fx, fy],
            [cx, cy],
            [w, h],
        ],
        dtype=torch.float32,
    )
    return intrinsics


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat(
        [
            RT,
            torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).repeat(
                RT.shape[0], 1, 1
            ),
        ],
        dim=1,
    )


def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack(
        [
            torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
            torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
            torch.tensor([[0, 0, 1]], dtype=torch.float32).repeat(RT.shape[0], 1),
        ],
        dim=1,
    )
    return torch.cat(
        [
            E.reshape(-1, 16),
            I.reshape(-1, 9),
        ],
        dim=-1,
    )


def calc_elevation(c2w):
    ## works for single or batched c2w
    ## assume world up is (0, 0, 1)
    pos = c2w[..., :3, 3]

    return np.arcsin(pos[..., 2] / np.linalg.norm(pos, axis=-1, keepdims=False))


def read_camera_matrix_single(json_file):
    with open(json_file, "r", encoding="utf8") as reader:
        json_content = json.load(reader)

    # negative sign for opencv to opengl
    # camera_matrix = np.zeros([3, 4])
    # camera_matrix[:3, 0] = np.array(json_content["x"])
    # camera_matrix[:3, 1] = -np.array(json_content["y"])
    # camera_matrix[:3, 2] = -np.array(json_content["z"])
    # camera_matrix[:3, 3] = np.array(json_content["origin"])
    camera_matrix = torch.zeros([3, 4])
    camera_matrix[:3, 0] = torch.tensor(json_content["x"])
    camera_matrix[:3, 1] = -torch.tensor(json_content["y"])
    camera_matrix[:3, 2] = -torch.tensor(json_content["z"])
    camera_matrix[:3, 3] = torch.tensor(json_content["origin"])
    """
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    """

    return camera_matrix


def blend_white_bg(image):
    new_image = Image.new("RGB", image.size, (255, 255, 255))
    new_image.paste(image, mask=image.split()[3])

    return new_image


def flatten_for_video(input):
    return input.flatten()


FLATTEN_FIELDS = ["fps_id", "motion_bucket_id", "cond_aug", "elevation"]


def video_collate_fn(batch: list[dict], *args, **kwargs):
    out = {}
    for key in batch[0].keys():
        if key in FLATTEN_FIELDS:
            out[key] = default_collate([item[key] for item in batch])
            out[key] = flatten_for_video(out[key])
        elif key == "num_video_frames":
            out[key] = batch[0][key]
        elif key in ["frames", "latents", "rgb"]:
            out[key] = default_collate([item[key] for item in batch])
            out[key] = rearrange(out[key], "b t c h w -> (b t) c h w")
        else:
            out[key] = default_collate([item[key] for item in batch])

    if "pixelnerf_input" in out:
        out["pixelnerf_input"]["rgb"] = rearrange(
            out["pixelnerf_input"]["rgb"], "b t c h w -> (b t) c h w"
        )

    return out


class GObjaverse(Dataset):
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
        fps_id=0.0,
        motion_bucket_id=300.0,
        use_latents=False,
        load_caps=False,
        front_view_selection="random",
        load_pixelnerf=False,
        debug_base_idx=None,
        scale_pose: bool = False,
        max_n_cond: int = 1,
        **unused_kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.transform = transform
        self.use_latents = use_latents

        self.ids = json.load(open(self.root_dir / "valid_uids.json", "r"))
        self.n_views = 24

        self.load_caps = load_caps
        if self.load_caps:
            self.caps = json.load(open(self.root_dir / "text_captions_cap3d.json", "r"))

        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        self.condition_on_elevation = condition_on_elevation
        self.fps_id = fps_id
        self.motion_bucket_id = motion_bucket_id
        self.load_pixelnerf = load_pixelnerf
        self.scale_pose = scale_pose
        self.max_n_cond = max_n_cond

        if self.use_latents:
            self.latents_dir = self.root_dir / "latents256"
            self.clip_dir = self.root_dir / "clip_emb256"

        self.front_view_selection = front_view_selection
        if self.front_view_selection == "random":
            pass
        elif self.front_view_selection == "fixed":
            pass
        elif self.front_view_selection.startswith("clip_score"):
            self.clip_scores = torch.load(self.root_dir / "clip_score_per_view.pt")
            self.ids = list(self.clip_scores.keys())
        else:
            raise ValueError(
                f"Unknown front view selection method {self.front_view_selection}"
            )

        if max_item is not None:
            self.ids = self.ids[:max_item]
            ## debug
            self.ids = self.ids * 10000

        if debug_base_idx is not None:
            print(f"debug mode with base idx: {debug_base_idx}")
            self.debug_base_idx = debug_base_idx

    def __getitem__(self, idx: int):
        if hasattr(self, "debug_base_idx"):
            idx = (idx + self.debug_base_idx) % len(self.ids)
        data = {}
        idx_list = np.arange(self.n_views)
        # if self.random_front:
        #     roll_idx = np.random.randint(self.n_views)
        #     idx_list = np.roll(idx_list, roll_idx)
        if self.front_view_selection == "random":
            roll_idx = np.random.randint(self.n_views)
            idx_list = np.roll(idx_list, roll_idx)
        elif self.front_view_selection == "fixed":
            pass
        elif self.front_view_selection == "clip_score_softmax":
            this_clip_score = (
                F.softmax(self.clip_scores[self.ids[idx]], dim=-1).cpu().numpy()
            )
            roll_idx = np.random.choice(idx_list, p=this_clip_score)
            idx_list = np.roll(idx_list, roll_idx)
        elif self.front_view_selection == "clip_score_max":
            this_clip_score = (
                F.softmax(self.clip_scores[self.ids[idx]], dim=-1).cpu().numpy()
            )
            roll_idx = np.argmax(this_clip_score)
            idx_list = np.roll(idx_list, roll_idx)
        frames = []
        if not self.use_latents:
            try:
                for view_idx in idx_list:
                    frame = Image.open(
                        self.root_dir
                        / "gobjaverse"
                        / self.ids[idx]
                        / f"{view_idx:05d}/{view_idx:05d}.png"
                    )
                    frames.append(self.transform(frame))
            except:
                idx = 0
                frames = []
                for view_idx in idx_list:
                    frame = Image.open(
                        self.root_dir
                        / "gobjaverse"
                        / self.ids[idx]
                        / f"{view_idx:05d}/{view_idx:05d}.png"
                    )
                    frames.append(self.transform(frame))
                # a workaround for some bugs in gobjaverse
                # use idx=0 and the repeat will be resolved when gathering results, valid number of items can be checked by the len of results
            frames = torch.stack(frames, dim=0)
            cond = frames[0]

            cond_aug = np.exp(
                np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
            )

            data.update(
                {
                    "frames": frames,
                    "cond_frames_without_noise": cond,
                    "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
                    "cond_frames": cond + cond_aug * torch.randn_like(cond),
                    "fps_id": torch.as_tensor([self.fps_id] * self.n_views),
                    "motion_bucket_id": torch.as_tensor(
                        [self.motion_bucket_id] * self.n_views
                    ),
                    "num_video_frames": 24,
                    "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
                }
            )
        else:
            latents = torch.load(self.latents_dir / f"{self.ids[idx]}.pt")[idx_list]
            clip_emb = torch.load(self.clip_dir / f"{self.ids[idx]}.pt")[idx_list][0]

            cond = latents[0]

            cond_aug = np.exp(
                np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
            )

            data.update(
                {
                    "latents": latents,
                    "cond_frames_without_noise": clip_emb,
                    "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
                    "cond_frames": cond + cond_aug * torch.randn_like(cond),
                    "fps_id": torch.as_tensor([self.fps_id] * self.n_views),
                    "motion_bucket_id": torch.as_tensor(
                        [self.motion_bucket_id] * self.n_views
                    ),
                    "num_video_frames": 24,
                    "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
                }
            )

        if self.condition_on_elevation:
            sample_c2w = read_camera_matrix_single(
                self.root_dir / self.ids[idx] / f"00000/00000.json"
            )
            elevation = calc_elevation(sample_c2w)
            data["elevation"] = torch.as_tensor([elevation] * self.n_views)

        if self.load_pixelnerf:
            assert "frames" in data, f"pixelnerf cannot work with latents only mode"
            data["pixelnerf_input"] = {}
            RTs = []
            intrinsics = []
            for view_idx in idx_list:
                meta = (
                    self.root_dir
                    / "gobjaverse"
                    / self.ids[idx]
                    / f"{view_idx:05d}/{view_idx:05d}.json"
                )
                RTs.append(read_camera_matrix_single(meta)[:3])
                intrinsics.append(read_camera_instrinsics_single(meta, 256, 256))
            RTs = torch.stack(RTs, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            cameras = build_camera_standard(RTs, intrinsics)
            data["pixelnerf_input"]["cameras"] = cameras

            downsampled = []
            for view_idx in idx_list:
                frame = Image.open(
                    self.root_dir
                    / "gobjaverse"
                    / self.ids[idx]
                    / f"{view_idx:05d}/{view_idx:05d}.png"
                ).resize((32, 32))
                downsampled.append(to_tensor(blend_white_bg(frame)))
            data["pixelnerf_input"]["rgb"] = torch.stack(downsampled, dim=0)
            data["pixelnerf_input"]["frames"] = data["frames"]
            if self.scale_pose:
                c2ws = cameras[..., :16].reshape(-1, 4, 4)
                center = c2ws[:, :3, 3].mean(0)
                radius = (c2ws[:, :3, 3] - center).norm(dim=-1).max()
                scale = 1.5 / radius
                c2ws[..., :3, 3] = (c2ws[..., :3, 3] - center) * scale
                cameras[..., :16] = c2ws.reshape(-1, 16)

        if self.load_caps:
            data["caption"] = self.caps[self.ids[idx]]
            data["ids"] = self.ids[idx]

        return data

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        if self.max_n_cond > 1:
            n_cond = np.random.randint(1, self.max_n_cond + 1)
            if n_cond > 1:
                for b in batch:
                    source_index = [0] + np.random.choice(
                        np.arange(1, self.n_views),
                        self.max_n_cond - 1,
                        replace=False,
                    ).tolist()
                    b["pixelnerf_input"]["source_index"] = torch.as_tensor(source_index)
                    b["pixelnerf_input"]["n_cond"] = n_cond
                    b["pixelnerf_input"]["source_images"] = b["frames"][source_index]
                    b["pixelnerf_input"]["source_cameras"] = b["pixelnerf_input"][
                        "cameras"
                    ][source_index]

        return video_collate_fn(batch)


class ObjaverseSpiral(Dataset):
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
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.transform = transform

        self.ids = json.load(open(self.root_dir / f"{split}_ids.json", "r"))
        self.n_views = 24
        valid_ids = []
        for idx in self.ids:
            if (self.root_dir / idx).exists():
                valid_ids.append(idx)
        self.ids = valid_ids

        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        self.condition_on_elevation = condition_on_elevation

        if max_item is not None:
            self.ids = self.ids[:max_item]

            ## debug
            self.ids = self.ids * 10000

    def __getitem__(self, idx: int):
        frames = []
        idx_list = np.arange(self.n_views)
        if self.random_front:
            roll_idx = np.random.randint(self.n_views)
            idx_list = np.roll(idx_list, roll_idx)
        for view_idx in idx_list:
            frame = Image.open(
                self.root_dir / self.ids[idx] / f"{view_idx:05d}/{view_idx:05d}.png"
            )
            frames.append(self.transform(frame))

        # data = {"jpg": torch.stack(frames, dim=0)}  # [T, C, H, W]
        frames = torch.stack(frames, dim=0)
        cond = frames[0]

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )

        data = {
            "frames": frames,
            "cond_frames_without_noise": cond,
            "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
            "cond_frames": cond + cond_aug * torch.randn_like(cond),
            "fps_id": torch.as_tensor([1.0] * self.n_views),
            "motion_bucket_id": torch.as_tensor([300.0] * self.n_views),
            "num_video_frames": 24,
            "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
        }

        if self.condition_on_elevation:
            sample_c2w = read_camera_matrix_single(
                self.root_dir / self.ids[idx] / f"00000/00000.json"
            )
            elevation = calc_elevation(sample_c2w)
            data["elevation"] = torch.as_tensor([elevation] * self.n_views)

        return data

    def __len__(self):
        return len(self.ids)


class ObjaverseLVISSpiral(Dataset):
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
        use_precomputed_latents=False,
        **unused_kwargs,
    ):
        print("Using LVIS subset")
        self.root_dir = Path(root_dir)
        self.latent_dir = Path("/mnt/vepfs/3Ddataset/render_results/latents512")
        self.split = split
        self.random_front = random_front
        self.transform = transform
        self.use_precomputed_latents = use_precomputed_latents

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

    def __getitem__(self, idx: int):
        frames = []
        idx_list = np.arange(self.n_views)
        if self.random_front:
            roll_idx = np.random.randint(self.n_views)
            idx_list = np.roll(idx_list, roll_idx)
        for view_idx in idx_list:
            frame = Image.open(
                self.root_dir
                / self.ids[idx]
                / "elevations_0"
                / f"colors_{view_idx * 2}.png"
            )
            frames.append(self.transform(frame))

        frames = torch.stack(frames, dim=0)
        cond = frames[0]

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )

        data = {
            "frames": frames,
            "cond_frames_without_noise": cond,
            "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
            "cond_frames": cond + cond_aug * torch.randn_like(cond),
            "fps_id": torch.as_tensor([0.0] * self.n_views),
            "motion_bucket_id": torch.as_tensor([300.0] * self.n_views),
            "num_video_frames": self.n_views,
            "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
        }

        if self.use_precomputed_latents:
            data["latents"] = torch.load(self.latent_dir / f"{self.ids[idx]}.pt")

        if self.condition_on_elevation:
            # sample_c2w = read_camera_matrix_single(
            #     self.root_dir / self.ids[idx] / f"00000/00000.json"
            # )
            # elevation = calc_elevation(sample_c2w)
            # data["elevation"] = torch.as_tensor([elevation] * self.n_views)
            assert False, "currently assumes elevation 0"

        return data

    def __len__(self):
        return len(self.ids)


class ObjaverseALLSpiral(ObjaverseLVISSpiral):
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
        use_precomputed_latents=False,
        **unused_kwargs,
    ):
        print("Using ALL objects in Objaverse")
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.transform = transform
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dir = Path("/mnt/vepfs/3Ddataset/render_results/latents512")

        self.ids = json.load(open("./assets/all_ids.json", "r"))
        self.n_views = 18
        valid_ids = []
        for idx in self.ids:
            if (self.root_dir / idx).exists() and (self.root_dir / idx).is_dir():
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


class ObjaverseWithPose(Dataset):
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
        use_precomputed_latents=False,
        **unused_kwargs,
    ):
        print("Using Objaverse with poses")
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.transform = transform
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dir = Path("/mnt/vepfs/3Ddataset/render_results/latents512")

        self.ids = json.load(open("./assets/all_ids.json", "r"))
        self.n_views = 18
        valid_ids = []
        for idx in self.ids:
            if (self.root_dir / idx).exists() and (self.root_dir / idx).is_dir():
                valid_ids.append(idx)
        self.ids = valid_ids
        print("=" * 30)
        print("Number of valid ids: ", len(self.ids))
        print("=" * 30)

        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        self.condition_on_elevation = condition_on_elevation

    def __getitem__(self, idx: int):
        frames = []
        idx_list = np.arange(self.n_views)
        if self.random_front:
            roll_idx = np.random.randint(self.n_views)
            idx_list = np.roll(idx_list, roll_idx)
        for view_idx in idx_list:
            frame = Image.open(
                self.root_dir
                / self.ids[idx]
                / "elevations_0"
                / f"colors_{view_idx * 2}.png"
            )
            frames.append(self.transform(frame))

        frames = torch.stack(frames, dim=0)
        cond = frames[0]

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )

        data = {
            "frames": frames,
            "cond_frames_without_noise": cond,
            "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
            "cond_frames": cond + cond_aug * torch.randn_like(cond),
            "fps_id": torch.as_tensor([0.0] * self.n_views),
            "motion_bucket_id": torch.as_tensor([300.0] * self.n_views),
            "num_video_frames": self.n_views,
            "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
        }

        if self.use_precomputed_latents:
            data["latents"] = torch.load(self.latent_dir / f"{self.ids[idx]}.pt")

        if self.condition_on_elevation:
            assert False, "currently assumes elevation 0"

        return data


class LatentObjaverse(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        random_front=False,
        subset="lvis",
        fps_id=1.0,
        motion_bucket_id=300.0,
        cond_aug_mean=-3.0,
        cond_aug_std=0.5,
        **unused_kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.random_front = random_front
        self.ids = json.load(open(Path("./assets") / f"{subset}_ids.json", "r"))
        self.clip_emb_dir = self.root_dir / ".." / "clip_emb512"
        self.n_views = 18
        self.fps_id = fps_id
        self.motion_bucket_id = motion_bucket_id
        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        if self.random_front:
            print("Using a random view as front view")

        valid_ids = []
        for idx in self.ids:
            if (self.root_dir / f"{idx}.pt").exists() and (
                self.clip_emb_dir / f"{idx}.pt"
            ).exists():
                valid_ids.append(idx)
        self.ids = valid_ids
        print("=" * 30)
        print("Number of valid ids: ", len(self.ids))
        print("=" * 30)

    def __getitem__(self, idx: int):
        uid = self.ids[idx]
        idx_list = torch.arange(self.n_views)
        latents = torch.load(self.root_dir / f"{uid}.pt")
        clip_emb = torch.load(self.clip_emb_dir / f"{uid}.pt")
        if self.random_front:
            idx_list = torch.roll(idx_list, np.random.randint(self.n_views))
        latents = latents[idx_list]
        clip_emb = clip_emb[idx_list][0]

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )
        cond = latents[0]

        data = {
            "latents": latents,
            "cond_frames_without_noise": clip_emb,
            "cond_frames": cond + cond_aug * torch.randn_like(cond),
            "fps_id": torch.as_tensor([self.fps_id] * self.n_views),
            "motion_bucket_id": torch.as_tensor([self.motion_bucket_id] * self.n_views),
            "cond_aug": torch.as_tensor([cond_aug] * self.n_views),
            "num_video_frames": self.n_views,
            "image_only_indicator": torch.as_tensor([0.0] * self.n_views),
        }

        return data

    def __len__(self):
        return len(self.ids)


class ObjaverseSpiralDataset(LightningDataModule):
    def __init__(
        self,
        root_dir,
        random_front=False,
        batch_size=2,
        num_workers=10,
        prefetch_factor=2,
        shuffle=True,
        max_item=None,
        dataset_cls="richdreamer",
        reso: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.max_item = max_item

        self.transform = Compose(
            [
                blend_white_bg,
                Resize((reso, reso)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        data_cls = {
            "richdreamer": ObjaverseSpiral,
            "lvis": ObjaverseLVISSpiral,
            "shengshu_all": ObjaverseALLSpiral,
            "latent": LatentObjaverse,
            "gobjaverse": GObjaverse,
        }[dataset_cls]

        self.train_dataset = data_cls(
            root_dir=root_dir,
            split="train",
            random_front=random_front,
            transform=self.transform,
            max_item=self.max_item,
            **kwargs,
        )
        self.test_dataset = data_cls(
            root_dir=root_dir,
            split="val",
            random_front=random_front,
            transform=self.transform,
            max_item=self.max_item,
            **kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=video_collate_fn
            if not hasattr(self.train_dataset, "collate_fn")
            else self.train_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=video_collate_fn
            if not hasattr(self.test_dataset, "collate_fn")
            else self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=video_collate_fn
            if not hasattr(self.test_dataset, "collate_fn")
            else self.train_dataset.collate_fn,
        )
