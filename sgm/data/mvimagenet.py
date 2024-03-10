import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
import rembg
from rembg import remove, new_session
from einops import rearrange

from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.transforms.functional import to_tensor
from pytorch_lightning import LightningDataModule

from sgm.data.colmap import read_cameras_binary, read_images_binary
from sgm.data.objaverse import video_collate_fn, FLATTEN_FIELDS, flatten_for_video


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def qt2c2w(q, t):
    # NOTE: remember to convert to opengl coordinate system
    # rot = Rotation.from_quat(q).as_matrix()
    rot = qvec2rotmat(q)
    c2w = np.eye(4)
    c2w[:3, :3] = np.transpose(rot)
    c2w[:3, 3] = -np.transpose(rot) @ t
    c2w[..., 1:3] *= -1
    return c2w


def random_crop():
    pass


class MVImageNet(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        transform,
        reso: int = 256,
        mask_type: str = "random",
        cond_aug_mean=-3.0,
        cond_aug_std=0.5,
        condition_on_elevation=False,
        fps_id=0.0,
        motion_bucket_id=300.0,
        num_frames: int = 24,
        use_mask: bool = True,
        load_pixelnerf: bool = False,
        scale_pose: bool = False,
        max_n_cond: int = 1,
        min_n_cond: int = 1,
        cond_on_multi: bool = False,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split

        avails = self.root_dir.glob("*/*")
        self.ids = list(
            map(
                lambda x: str(x.relative_to(self.root_dir)),
                filter(lambda x: x.is_dir(), avails),
            )
        )

        self.transform = transform
        self.reso = reso
        self.num_frames = num_frames
        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
        self.condition_on_elevation = condition_on_elevation
        self.fps_id = fps_id
        self.motion_bucket_id = motion_bucket_id
        self.mask_type = mask_type
        self.use_mask = use_mask
        self.load_pixelnerf = load_pixelnerf
        self.scale_pose = scale_pose
        self.max_n_cond = max_n_cond
        self.min_n_cond = min_n_cond
        self.cond_on_multi = cond_on_multi

        if self.cond_on_multi:
            assert self.min_n_cond == self.max_n_cond
        self.session = new_session()

    def __getitem__(self, index: int):
        # mvimgnet starts with idx==1
        idx_list = np.arange(0, self.num_frames)
        this_image_dir = self.root_dir / self.ids[index] / "images"
        this_camera_dir = self.root_dir / self.ids[index] / "sparse/0"

        # while not this_camera_dir.exists():
        #     index = (index + 1) % len(self.ids)
        #     this_image_dir = self.root_dir / self.ids[index] / "images"
        #     this_camera_dir = self.root_dir / self.ids[index] / "sparse/0"
        if not this_camera_dir.exists():
            index = 0
            this_image_dir = self.root_dir / self.ids[index] / "images"
            this_camera_dir = self.root_dir / self.ids[index] / "sparse/0"

        this_images = read_images_binary(this_camera_dir / "images.bin")
        # filenames = list(map(lambda x: f"{x:03d}", this_images.keys()))
        filenames = list(this_images.keys())

        if len(filenames) == 0:
            index = 0
            this_image_dir = self.root_dir / self.ids[index] / "images"
            this_camera_dir = self.root_dir / self.ids[index] / "sparse/0"
            this_images = read_images_binary(this_camera_dir / "images.bin")
            # filenames = list(map(lambda x: f"{x:03d}", this_images.keys()))
            filenames = list(this_images.keys())

        filenames = list(
            filter(lambda x: (this_image_dir / this_images[x].name).exists(), filenames)
        )

        filenames = sorted(filenames, key=lambda x: this_images[x].name)

        # # debug
        # names = []
        # for v in filenames:
        #     names.append(this_images[v].name)
        # breakpoint()

        while len(filenames) < self.num_frames:
            num_surpass = self.num_frames - len(filenames)
            filenames += list(reversed(filenames[-num_surpass:]))

        if len(filenames) < self.num_frames:
            print(f"\n\n{self.ids[index]}\n\n")

        frames = []
        cameras = []
        downsampled_rgb = []
        for view_idx in idx_list:
            this_id = filenames[view_idx]
            frame = Image.open(this_image_dir / this_images[this_id].name)
            w, h = frame.size

            if self.mask_type == "random":
                image_size = min(h, w)
                left = np.random.randint(0, w - image_size + 1)
                right = left + image_size
                top = np.random.randint(0, h - image_size + 1)
                bottom = top + image_size
                ## need to assign left, right, top, bottom, image_size
            elif self.mask_type == "object":
                pass
            elif self.mask_type == "rembg":
                image_size = min(h, w)
                if (
                    cached := this_image_dir
                    / f"{this_images[this_id].name[:-4]}_rembg.png"
                ).exists():
                    try:
                        mask = np.asarray(Image.open(cached, formats=["png"]))[..., 3]
                    except:
                        mask = remove(frame, session=self.session)
                        mask.save(cached)
                        mask = np.asarray(mask)[..., 3]
                else:
                    mask = remove(frame, session=self.session)
                    mask.save(cached)
                    mask = np.asarray(mask)[..., 3]
                # in h,w order
                y, x = np.array(mask.nonzero())
                bbox_cx = x.mean()
                bbox_cy = y.mean()

                if bbox_cy - image_size / 2 < 0:
                    top = 0
                elif bbox_cy + image_size / 2 > h:
                    top = h - image_size
                else:
                    top = int(bbox_cy - image_size / 2)

                if bbox_cx - image_size / 2 < 0:
                    left = 0
                elif bbox_cx + image_size / 2 > w:
                    left = w - image_size
                else:
                    left = int(bbox_cx - image_size / 2)

                # top = max(int(bbox_cy - image_size / 2), 0)
                # left = max(int(bbox_cx - image_size / 2), 0)
                bottom = top + image_size
                right = left + image_size
            else:
                raise ValueError(f"Unknown mask type: {self.mask_type}")

            frame = frame.crop((left, top, right, bottom))
            frame = frame.resize((self.reso, self.reso))
            frames.append(self.transform(frame))

            if self.load_pixelnerf:
                # extrinsics
                extrinsics = this_images[this_id]
                c2w = qt2c2w(extrinsics.qvec, extrinsics.tvec)
                # intrinsics
                intrinsics = read_cameras_binary(this_camera_dir / "cameras.bin")
                assert len(intrinsics) == 1
                intrinsics = intrinsics[1]
                f, cx, cy, _ = intrinsics.params
                f *= 1 / image_size
                cx -= left
                cy -= top
                cx *= 1 / image_size
                cy *= 1 / image_size  # all are relative values
                intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

                this_camera = np.zeros(25)
                this_camera[:16] = c2w.reshape(-1)
                this_camera[16:] = intrinsics.reshape(-1)

                cameras.append(this_camera)
                downsampled = frame.resize((self.reso // 8, self.reso // 8))
                downsampled_rgb.append((self.transform(downsampled) + 1.0) * 0.5)

        data = dict()

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )
        frames = torch.stack(frames)
        cond = frames[0]
        # setting all things in data
        data["frames"] = frames
        data["cond_frames_without_noise"] = cond
        data["cond_aug"] = torch.as_tensor([cond_aug] * self.num_frames)
        data["cond_frames"] = cond + cond_aug * torch.randn_like(cond)
        data["fps_id"] = torch.as_tensor([self.fps_id] * self.num_frames)
        data["motion_bucket_id"] = torch.as_tensor(
            [self.motion_bucket_id] * self.num_frames
        )
        data["num_video_frames"] = self.num_frames
        data["image_only_indicator"] = torch.as_tensor([0.0] * self.num_frames)

        if self.load_pixelnerf:
            # TODO: normalize camera poses
            data["pixelnerf_input"] = dict()
            data["pixelnerf_input"]["frames"] = frames
            data["pixelnerf_input"]["rgb"] = torch.stack(downsampled_rgb)

            cameras = torch.from_numpy(np.stack(cameras)).float()
            if self.scale_pose:
                c2ws = cameras[..., :16].reshape(-1, 4, 4)
                center = c2ws[:, :3, 3].mean(0)
                radius = (c2ws[:, :3, 3] - center).norm(dim=-1).max()
                scale = 1.5 / radius
                c2ws[..., :3, 3] = (c2ws[..., :3, 3] - center) * scale
                cameras[..., :16] = c2ws.reshape(-1, 16)

            # if self.max_n_cond > 1:
            #     # TODO implement this
            #     n_cond = np.random.randint(1, self.max_n_cond + 1)
            #     # debug
            #     source_index = [0]
            #     if n_cond > 1:
            #         source_index += np.random.choice(
            #             np.arange(1, self.num_frames),
            #             self.max_n_cond - 1,
            #             replace=False,
            #         ).tolist()
            #         data["pixelnerf_input"]["source_index"] = torch.as_tensor(
            #             source_index
            #         )
            #         data["pixelnerf_input"]["n_cond"] = n_cond
            #         data["pixelnerf_input"]["source_images"] = frames[source_index]
            #         data["pixelnerf_input"]["source_cameras"] = cameras[source_index]

            data["pixelnerf_input"]["cameras"] = cameras

        return data

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        # a hack to add source index and keep consistent within a batch
        if self.max_n_cond > 1:
            # TODO implement this
            n_cond = np.random.randint(self.min_n_cond, self.max_n_cond + 1)
            # debug
            # source_index = [0]
            if n_cond > 1:
                for b in batch:
                    source_index = [0] + np.random.choice(
                        np.arange(1, self.num_frames),
                        self.max_n_cond - 1,
                        replace=False,
                    ).tolist()
                    b["pixelnerf_input"]["source_index"] = torch.as_tensor(source_index)
                    b["pixelnerf_input"]["n_cond"] = n_cond
                    b["pixelnerf_input"]["source_images"] = b["frames"][source_index]
                    b["pixelnerf_input"]["source_cameras"] = b["pixelnerf_input"][
                        "cameras"
                    ][source_index]

                    if self.cond_on_multi:
                        b["cond_frames_without_noise"] = b["frames"][source_index]

        ret = video_collate_fn(batch)

        if self.cond_on_multi:
            ret["cond_frames_without_noise"] = rearrange(ret["cond_frames_without_noise"], "b t ... -> (b t) ...")

        return ret


class MVImageNetFixedCond(MVImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MVImageNetDataset(LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=2,
        shuffle=True,
        num_workers=10,
        prefetch_factor=2,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle

        self.transform = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.train_dataset = MVImageNet(
            root_dir=root_dir,
            split="train",
            transform=self.transform,
            **kwargs,
        )

        self.test_dataset = MVImageNet(
            root_dir=root_dir,
            split="test",
            transform=self.transform,
            **kwargs,
        )

    def train_dataloader(self):
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0])

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.train_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.test_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=video_collate_fn,
        )
