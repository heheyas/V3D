"""
adopted from SparseFusion
Wrapper for the full CO3Dv2 dataset
#@ Modified from https://github.com/facebookresearch/pytorch3d
"""

import json
import logging
import math
import os
import random
import time
import warnings
from collections import defaultdict
from itertools import islice
from typing import (
    Any,
    ClassVar,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)
from einops import rearrange, repeat

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.dataset_base import DatasetBase
from sgm.data.json_index_dataset import (
    FrameAnnotsEntry,
    _bbox_xywh_to_xyxy,
    _bbox_xyxy_to_xywh,
    _clamp_box_to_image_bounds_and_round,
    _crop_around_box,
    _get_1d_bounds,
    _get_bbox_from_mask,
    _get_clamp_bbox,
    _load_1bit_png_mask,
    _load_16big_png_depth,
    _load_depth,
    _load_depth_mask,
    _load_image,
    _load_mask,
    _load_pointcloud,
    _rescale_bbox,
    _safe_as_tensor,
    _seq_name_to_seed,
)
from sgm.data.objaverse import video_collate_fn
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    get_available_subset_names,
)
from pytorch3d.renderer.cameras import PerspectiveCameras

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field, fields

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import Pointclouds, join_pointclouds_as_batch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

CO3D_ALL_CATEGORIES = list(
    reversed(
        [
            "baseballbat",
            "banana",
            "bicycle",
            "microwave",
            "tv",
            "cellphone",
            "toilet",
            "hairdryer",
            "couch",
            "kite",
            "pizza",
            "umbrella",
            "wineglass",
            "laptop",
            "hotdog",
            "stopsign",
            "frisbee",
            "baseballglove",
            "cup",
            "parkingmeter",
            "backpack",
            "toyplane",
            "toybus",
            "handbag",
            "chair",
            "keyboard",
            "car",
            "motorcycle",
            "carrot",
            "bottle",
            "sandwich",
            "remote",
            "bowl",
            "skateboard",
            "toaster",
            "mouse",
            "toytrain",
            "book",
            "toytruck",
            "orange",
            "broccoli",
            "plant",
            "teddybear",
            "suitcase",
            "bench",
            "ball",
            "cake",
            "vase",
            "hydrant",
            "apple",
            "donut",
        ]
    )
)

CO3D_ALL_TEN = [
    "donut",
    "apple",
    "hydrant",
    "vase",
    "cake",
    "ball",
    "bench",
    "suitcase",
    "teddybear",
    "plant",
]


# @ FROM https://github.com/facebookresearch/pytorch3d
@dataclass
class FrameData(Mapping[str, Any]):
    """
    A type of the elements returned by indexing the dataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.
    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        image_size_hw: The size of the image in pixels; (height, width) tensor
                        of shape (2,).
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box tightly enclosing the foreground object in the
            format (x0, y0, width, height). The convention assumes that
            `x0+width` and `y0+height` includes the boundary of the box.
            I.e., to slice out the corresponding crop from an image tensor `I`
            we execute `crop = I[..., y0:y0+height, x0:x0+width]`
        crop_bbox_xywh: The bounding box denoting the boundaries of `image_rgb`
            in the original image coordinates in the format (x0, y0, width, height).
            The convention is the same as for `bbox_xywh`. `crop_bbox_xywh` differs
            from `bbox_xywh` due to padding (which can happen e.g. due to
            setting `JsonIndexDataset.box_crop_context > 0`)
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: Optional[torch.LongTensor]
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    frame_timestamp: Optional[torch.Tensor] = None
    image_size_hw: Optional[torch.Tensor] = None
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = ""
    depth_map: Optional[torch.Tensor] = torch.zeros(1)
    depth_mask: Optional[torch.Tensor] = torch.zeros(1)
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    crop_bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = ""
    sequence_point_cloud: Optional[Pointclouds] = torch.zeros(1)
    sequence_point_cloud_idx: Optional[torch.Tensor] = torch.zeros(1)
    frame_type: Union[str, List[str], None] = ""  # known | unseen
    meta: dict = field(default_factory=lambda: {})
    valid_region: Optional[torch.Tensor] = None
    category_one_hot: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        new_params = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[f.name] = value.to(*args, **kwargs)
            else:
                new_params[f.name] = value
        return type(self)(**new_params)

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(fields(self))

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)

        elif isinstance(elem, Pointclouds):
            return join_pointclouds_as_batch(batch)

        elif isinstance(elem, CamerasBase):
            # TODO: don't store K; enforce working in NDC space
            return join_cameras_as_batch(batch)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


# @ MODIFIED FROM https://github.com/facebookresearch/pytorch3d
class CO3Dv2Wrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="/drive/datasets/co3d/",
        category="hydrant",
        subset="fewview_train",
        stage="train",
        sample_batch_size=20,
        image_size=256,
        masked=False,
        deprecated_val_region=False,
        return_frame_data_list=False,
        reso: int = 256,
        mask_type: str = "random",
        cond_aug_mean=-3.0,
        cond_aug_std=0.5,
        condition_on_elevation=False,
        fps_id=0.0,
        motion_bucket_id=300.0,
        num_frames: int = 20,
        use_mask: bool = True,
        load_pixelnerf: bool = True,
        scale_pose: bool = True,
        max_n_cond: int = 5,
        min_n_cond: int = 2,
        cond_on_multi: bool = False,
    ):
        root = root_dir
        from typing import List

        from co3d.dataset.data_types import (
            FrameAnnotation,
            SequenceAnnotation,
            load_dataclass_jgzip,
        )

        self.dataset_root = root
        self.path_manager = None
        self.subset = subset
        self.stage = stage
        self.subset_lists_file: List[str] = [
            f"{self.dataset_root}/{category}/set_lists/set_lists_{subset}.json"
        ]
        self.subsets: Optional[List[str]] = [subset]
        self.sample_batch_size = sample_batch_size
        self.limit_to: int = 0
        self.limit_sequences_to: int = 0
        self.pick_sequence: Tuple[str, ...] = ()
        self.exclude_sequence: Tuple[str, ...] = ()
        self.limit_category_to: Tuple[int, ...] = ()
        self.load_images: bool = True
        self.load_depths: bool = False
        self.load_depth_masks: bool = False
        self.load_masks: bool = True
        self.load_point_clouds: bool = False
        self.max_points: int = 0
        self.mask_images: bool = False
        self.mask_depths: bool = False
        self.image_height: Optional[int] = image_size
        self.image_width: Optional[int] = image_size
        self.box_crop: bool = True
        self.box_crop_mask_thr: float = 0.4
        self.box_crop_context: float = 0.3
        self.remove_empty_masks: bool = True
        self.n_frames_per_sequence: int = -1
        self.seed: int = 0
        self.sort_frames: bool = False
        self.eval_batches: Any = None

        self.img_h = self.image_height
        self.img_w = self.image_width
        self.masked = masked
        self.deprecated_val_region = deprecated_val_region
        self.return_frame_data_list = return_frame_data_list

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

        start_time = time.time()
        if "all_" in category or category == "all":
            self.category_frame_annotations = []
            self.category_sequence_annotations = []
            self.subset_lists_file = []

            if category == "all":
                cats = CO3D_ALL_CATEGORIES
            elif category == "all_four":
                cats = ["hydrant", "teddybear", "motorcycle", "bench"]
            elif category == "all_ten":
                cats = [
                    "donut",
                    "apple",
                    "hydrant",
                    "vase",
                    "cake",
                    "ball",
                    "bench",
                    "suitcase",
                    "teddybear",
                    "plant",
                ]
            elif category == "all_15":
                cats = [
                    "hydrant",
                    "teddybear",
                    "motorcycle",
                    "bench",
                    "hotdog",
                    "remote",
                    "suitcase",
                    "donut",
                    "plant",
                    "toaster",
                    "keyboard",
                    "handbag",
                    "toyplane",
                    "tv",
                    "orange",
                ]
            else:
                print("UNSPECIFIED CATEGORY SUBSET")
                cats = ["hydrant", "teddybear"]
            print("loading", cats)
            for cat in cats:
                self.category_frame_annotations.extend(
                    load_dataclass_jgzip(
                        f"{self.dataset_root}/{cat}/frame_annotations.jgz",
                        List[FrameAnnotation],
                    )
                )
                self.category_sequence_annotations.extend(
                    load_dataclass_jgzip(
                        f"{self.dataset_root}/{cat}/sequence_annotations.jgz",
                        List[SequenceAnnotation],
                    )
                )
                self.subset_lists_file.append(
                    f"{self.dataset_root}/{cat}/set_lists/set_lists_{subset}.json"
                )

        else:
            self.category_frame_annotations = load_dataclass_jgzip(
                f"{self.dataset_root}/{category}/frame_annotations.jgz",
                List[FrameAnnotation],
            )
            self.category_sequence_annotations = load_dataclass_jgzip(
                f"{self.dataset_root}/{category}/sequence_annotations.jgz",
                List[SequenceAnnotation],
            )

        self.subset_to_image_path = None
        self._load_frames()
        self._load_sequences()
        self._sort_frames()
        self._load_subset_lists()
        self._filter_db()  # also computes sequence indices
        # self._extract_and_set_eval_batches()
        # print(self.eval_batches)
        logger.info(str(self))

        self.seq_to_frames = {}
        for fi, item in enumerate(self.frame_annots):
            if item["frame_annotation"].sequence_name in self.seq_to_frames:
                self.seq_to_frames[item["frame_annotation"].sequence_name].append(fi)
            else:
                self.seq_to_frames[item["frame_annotation"].sequence_name] = [fi]

        if self.stage != "test" or self.subset != "fewview_test":
            count = 0
            new_seq_to_frames = {}
            for item in self.seq_to_frames:
                if len(self.seq_to_frames[item]) > 10:
                    count += 1
                    new_seq_to_frames[item] = self.seq_to_frames[item]
            self.seq_to_frames = new_seq_to_frames

        self.seq_list = list(self.seq_to_frames.keys())

        # @ REMOVE A FEW TRAINING SEQ THAT CAUSES BUG
        remove_list = ["411_55952_107659", "376_42884_85882"]
        for remove_idx in remove_list:
            if remove_idx in self.seq_to_frames:
                self.seq_list.remove(remove_idx)
                print("removing", remove_idx)

        print("total training seq", len(self.seq_to_frames))
        print("data loading took", time.time() - start_time, "seconds")

        self.all_category_list = list(CO3D_ALL_CATEGORIES)
        self.all_category_list.sort()
        self.cat_to_idx = {}
        for ci, cname in enumerate(self.all_category_list):
            self.cat_to_idx[cname] = ci

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq_index = self.seq_list[index]

        if self.subset == "fewview_test" and self.stage == "test":
            batch_idx = torch.arange(len(self.seq_to_frames[seq_index]))

        elif self.stage == "test":
            batch_idx = (
                torch.linspace(
                    0, len(self.seq_to_frames[seq_index]) - 1, self.sample_batch_size
                )
                .long()
                .tolist()
            )
        else:
            rand = torch.randperm(len(self.seq_to_frames[seq_index]))
            batch_idx = rand[: min(len(rand), self.sample_batch_size)]

        frame_data_list = []
        idx_list = []
        timestamp_list = []
        for idx in batch_idx:
            idx_list.append(self.seq_to_frames[seq_index][idx])
            timestamp_list.append(
                self.frame_annots[self.seq_to_frames[seq_index][idx]][
                    "frame_annotation"
                ].frame_timestamp
            )
            frame_data_list.append(
                self._get_frame(int(self.seq_to_frames[seq_index][idx]))
            )

        time_order = torch.argsort(torch.tensor(timestamp_list))
        frame_data_list = [frame_data_list[i] for i in time_order]

        frame_data = FrameData.collate(frame_data_list)
        image_size = torch.Tensor([self.image_height]).repeat(
            frame_data.camera.R.shape[0], 2
        )
        frame_dict = {
            "R": frame_data.camera.R,
            "T": frame_data.camera.T,
            "f": frame_data.camera.focal_length,
            "c": frame_data.camera.principal_point,
            "images": frame_data.image_rgb * frame_data.fg_probability
            + (1 - frame_data.fg_probability),
            "valid_region": frame_data.mask_crop,
            "bbox": frame_data.valid_region,
            "image_size": image_size,
            "frame_type": frame_data.frame_type,
            "idx": seq_index,
            "category": frame_data.category_one_hot,
        }
        if not self.masked:
            frame_dict["images_full"] = frame_data.image_rgb
            frame_dict["masks"] = frame_data.fg_probability
            frame_dict["mask_crop"] = frame_data.mask_crop

        cond_aug = np.exp(
            np.random.randn(1)[0] * self.cond_aug_std + self.cond_aug_mean
        )

        def _pad(input):
            return torch.cat([input, torch.flip(input, dims=[0])], dim=0)[
                : self.num_frames
            ]

        if len(frame_dict["images"]) < self.num_frames:
            for k in frame_dict:
                if isinstance(frame_dict[k], torch.Tensor):
                    frame_dict[k] = _pad(frame_dict[k])

        data = dict()
        if "images_full" in frame_dict:
            frames = frame_dict["images_full"] * 2 - 1
        else:
            frames = frame_dict["images"] * 2 - 1
        data["frames"] = frames
        cond = frames[0]
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
            data["pixelnerf_input"] = dict()
            # Rs = frame_dict["R"].transpose(-1, -2)
            # Ts = frame_dict["T"]
            # Rs[:, :, 2] *= -1
            # Rs[:, :, 0] *= -1
            # Ts[:, 2] *= -1
            # Ts[:, 0] *= -1
            # c2ws = torch.zeros(Rs.shape[0], 4, 4)
            # c2ws[:, :3, :3] = Rs
            # c2ws[:, :3, 3] = Ts
            # c2ws[:, 3, 3] = 1
            # c2ws = c2ws.inverse()
            # # c2ws[..., 0] *= -1
            # # c2ws[..., 2] *= -1
            # cx = frame_dict["c"][:, 0]
            # cy = frame_dict["c"][:, 1]
            # fx = frame_dict["f"][:, 0]
            # fy = frame_dict["f"][:, 1]
            # intrinsics = torch.zeros(cx.shape[0], 3, 3)
            # intrinsics[:, 2, 2] = 1
            # intrinsics[:, 0, 0] = fx
            # intrinsics[:, 1, 1] = fy
            # intrinsics[:, 0, 2] = cx
            # intrinsics[:, 1, 2] = cy

            scene_cameras = PerspectiveCameras(
                R=frame_dict["R"],
                T=frame_dict["T"],
                focal_length=frame_dict["f"],
                principal_point=frame_dict["c"],
                image_size=frame_dict["image_size"],
            )
            R, T, intrinsics = opencv_from_cameras_projection(
                scene_cameras, frame_dict["image_size"]
            )
            c2ws = torch.zeros(R.shape[0], 4, 4)
            c2ws[:, :3, :3] = R
            c2ws[:, :3, 3] = T
            c2ws[:, 3, 3] = 1.0
            c2ws = c2ws.inverse()
            c2ws[..., 1:3] *= -1
            intrinsics[:, :2] /= 256

            cameras = torch.zeros(c2ws.shape[0], 25)
            cameras[..., :16] = c2ws.reshape(-1, 16)
            cameras[..., 16:] = intrinsics.reshape(-1, 9)
            if self.scale_pose:
                c2ws = cameras[..., :16].reshape(-1, 4, 4)
                center = c2ws[:, :3, 3].mean(0)
                radius = (c2ws[:, :3, 3] - center).norm(dim=-1).max()
                scale = 1.5 / radius
                c2ws[..., :3, 3] = (c2ws[..., :3, 3] - center) * scale
                cameras[..., :16] = c2ws.reshape(-1, 16)

            data["pixelnerf_input"]["frames"] = frames
            data["pixelnerf_input"]["cameras"] = cameras
            data["pixelnerf_input"]["rgb"] = (
                F.interpolate(
                    frames,
                    (self.image_width // 8, self.image_height // 8),
                    mode="bilinear",
                    align_corners=False,
                )
                + 1
            ) * 0.5

        return data
        # if self.return_frame_data_list:
        #     return (frame_dict, frame_data_list)
        # return frame_dict

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
            ret["cond_frames_without_noise"] = rearrange(
                ret["cond_frames_without_noise"], "b t ... -> (b t) ..."
            )

        return ret

    def _get_frame(self, index):
        # if index >= len(self.frame_annots):
        #     raise IndexError(f"index {index} out of range {len(self.frame_annots)}")

        entry = self.frame_annots[index]["frame_annotation"]
        # pyre-ignore[16]
        point_cloud = self.seq_annots[entry.sequence_name].point_cloud
        frame_data = FrameData(
            frame_number=_safe_as_tensor(entry.frame_number, torch.long),
            frame_timestamp=_safe_as_tensor(entry.frame_timestamp, torch.float),
            sequence_name=entry.sequence_name,
            sequence_category=self.seq_annots[entry.sequence_name].category,
            camera_quality_score=_safe_as_tensor(
                self.seq_annots[entry.sequence_name].viewpoint_quality_score,
                torch.float,
            ),
            point_cloud_quality_score=_safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(self.frame_annots[index])

        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
            clamp_bbox_xyxy,
            frame_data.crop_bbox_xywh,
        ) = self._load_crop_fg_probability(entry)

        scale = 1.0
        if self.load_images and entry.image is not None:
            # original image size
            frame_data.image_size_hw = _safe_as_tensor(entry.image.size, torch.long)

            (
                frame_data.image_rgb,
                frame_data.image_path,
                frame_data.mask_crop,
                scale,
            ) = self._load_crop_images(
                entry, frame_data.fg_probability, clamp_bbox_xyxy
            )
            # print(frame_data.fg_probability.sum())
            # print('scale', scale)

        #! INSERT
        if self.deprecated_val_region:
            # print(frame_data.crop_bbox_xywh)
            valid_bbox = _bbox_xywh_to_xyxy(frame_data.crop_bbox_xywh).float()
            # print(valid_bbox, frame_data.image_size_hw)
            valid_bbox[0] = torch.clip(
                (
                    valid_bbox[0]
                    - torch.div(frame_data.image_size_hw[1], 2, rounding_mode="floor")
                )
                / torch.div(frame_data.image_size_hw[1], 2, rounding_mode="floor"),
                -1.0,
                1.0,
            )
            valid_bbox[1] = torch.clip(
                (
                    valid_bbox[1]
                    - torch.div(frame_data.image_size_hw[0], 2, rounding_mode="floor")
                )
                / torch.div(frame_data.image_size_hw[0], 2, rounding_mode="floor"),
                -1.0,
                1.0,
            )
            valid_bbox[2] = torch.clip(
                (
                    valid_bbox[2]
                    - torch.div(frame_data.image_size_hw[1], 2, rounding_mode="floor")
                )
                / torch.div(frame_data.image_size_hw[1], 2, rounding_mode="floor"),
                -1.0,
                1.0,
            )
            valid_bbox[3] = torch.clip(
                (
                    valid_bbox[3]
                    - torch.div(frame_data.image_size_hw[0], 2, rounding_mode="floor")
                )
                / torch.div(frame_data.image_size_hw[0], 2, rounding_mode="floor"),
                -1.0,
                1.0,
            )
            # print(valid_bbox)
            frame_data.valid_region = valid_bbox
        else:
            #! UPDATED VALID BBOX
            if self.stage == "train":
                assert self.image_height == 256 and self.image_width == 256
                valid = torch.nonzero(frame_data.mask_crop[0])
                min_y = valid[:, 0].min()
                min_x = valid[:, 1].min()
                max_y = valid[:, 0].max()
                max_x = valid[:, 1].max()
                valid_bbox = torch.tensor(
                    [min_y, min_x, max_y, max_x], device=frame_data.image_rgb.device
                ).unsqueeze(0)
                valid_bbox = torch.clip(
                    (valid_bbox - (256 // 2)) / (256 // 2), -1.0, 1.0
                )
                frame_data.valid_region = valid_bbox[0]
            else:
                valid = torch.nonzero(frame_data.mask_crop[0])
                min_y = valid[:, 0].min()
                min_x = valid[:, 1].min()
                max_y = valid[:, 0].max()
                max_x = valid[:, 1].max()
                valid_bbox = torch.tensor(
                    [min_y, min_x, max_y, max_x], device=frame_data.image_rgb.device
                ).unsqueeze(0)
                valid_bbox = torch.clip(
                    (valid_bbox - (self.image_height // 2)) / (self.image_height // 2),
                    -1.0,
                    1.0,
                )
                frame_data.valid_region = valid_bbox[0]

        #! SET CLASS ONEHOT
        frame_data.category_one_hot = torch.zeros(
            (len(self.all_category_list)), device=frame_data.image_rgb.device
        )
        frame_data.category_one_hot[self.cat_to_idx[frame_data.sequence_category]] = 1

        if self.load_depths and entry.depth is not None:
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(entry, clamp_bbox_xyxy, frame_data.fg_probability)

        if entry.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(
                entry,
                scale,
                clamp_bbox_xyxy,
            )

        if self.load_point_clouds and point_cloud is not None:
            frame_data.sequence_point_cloud_path = pcl_path = os.path.join(
                self.dataset_root, point_cloud.path
            )
            frame_data.sequence_point_cloud = _load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )

        # for key in frame_data:
        #     if frame_data[key] == None:
        #         print(key)
        return frame_data

    def _extract_and_set_eval_batches(self):
        """
        Sets eval_batches based on input eval_batch_index.
        """
        if self.eval_batch_index is not None:
            if self.eval_batches is not None:
                raise ValueError(
                    "Cannot define both eval_batch_index and eval_batches."
                )
            self.eval_batches = self.seq_frame_index_to_dataset_index(
                self.eval_batch_index
            )

    def _load_crop_fg_probability(
        self, entry: types.FrameAnnotation
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[str],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        crop_box_xywh = None

        if (self.load_masks or self.box_crop) and entry.mask is not None:
            full_path = os.path.join(self.dataset_root, entry.mask.path)
            mask = _load_mask(self._local_path(full_path))

            if mask.shape[-2:] != entry.image.size:
                raise ValueError(
                    f"bad mask size: {mask.shape[-2:]} vs {entry.image.size}!"
                )

            bbox_xywh = torch.tensor(_get_bbox_from_mask(mask, self.box_crop_mask_thr))

            if self.box_crop:
                clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                    _get_clamp_bbox(
                        bbox_xywh,
                        image_path=entry.image.path,
                        box_crop_context=self.box_crop_context,
                    ),
                    image_size_hw=tuple(mask.shape[-2:]),
                )
                crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)

                mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)

            fg_probability, _, _ = self._resize_image(mask, mode="nearest")

        return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy, crop_box_xywh

    def _load_crop_images(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor, float]:
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(self._local_path(path))

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            image_rgb = _crop_around_box(image_rgb, clamp_bbox_xyxy, path)

        image_rgb, scale, mask_crop = self._resize_image(image_rgb)

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path, mask_crop, scale

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        clamp_bbox_xyxy: Optional[torch.Tensor],
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        assert entry_depth is not None
        path = os.path.join(self.dataset_root, entry_depth.path)
        depth_map = _load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            depth_bbox_xyxy = _rescale_bbox(
                clamp_bbox_xyxy, entry.image.size, depth_map.shape[-2:]
            )
            depth_map = _crop_around_box(depth_map, depth_bbox_xyxy, path)

        depth_map, _, _ = self._resize_image(depth_map, mode="nearest")

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry_depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
            depth_mask = _load_depth_mask(self._local_path(mask_path))

            if self.box_crop:
                assert clamp_bbox_xyxy is not None
                depth_mask_bbox_xyxy = _rescale_bbox(
                    clamp_bbox_xyxy, entry.image.size, depth_mask.shape[-2:]
                )
                depth_mask = _crop_around_box(
                    depth_mask, depth_mask_bbox_xyxy, mask_path
                )

            depth_mask, _, _ = self._resize_image(depth_mask, mode="nearest")
        else:
            depth_mask = torch.ones_like(depth_map)

        return depth_map, path, depth_mask

    def _get_pytorch3d_camera(
        self,
        entry: types.FrameAnnotation,
        scale: float,
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> PerspectiveCameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale
        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            principal_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        if self.image_height is None or self.image_width is None:
            out_size = list(reversed(entry.image.size))
        else:
            out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    def _load_frames(self) -> None:
        self.frame_annots = [
            FrameAnnotsEntry(frame_annotation=a, subset=None)
            for a in self.category_frame_annotations
        ]

    def _load_sequences(self) -> None:
        self.seq_annots = {
            entry.sequence_name: entry for entry in self.category_sequence_annotations
        }

    def _load_subset_lists(self) -> None:
        logger.info(f"Loading Co3D subset lists from {self.subset_lists_file}.")
        if not self.subset_lists_file:
            return

        frame_path_to_subset = {}

        for subset_list_file in self.subset_lists_file:
            with open(self._local_path(subset_list_file), "r") as f:
                subset_to_seq_frame = json.load(f)

            #! PRINT SUBSET_LIST STATS
            # if len(self.subset_lists_file) == 1:
            #     print('train frames', len(subset_to_seq_frame['train']))
            #     print('val frames', len(subset_to_seq_frame['val']))
            #     print('test frames', len(subset_to_seq_frame['test']))

            for set_ in subset_to_seq_frame:
                for _, _, path in subset_to_seq_frame[set_]:
                    if path in frame_path_to_subset:
                        frame_path_to_subset[path].add(set_)
                    else:
                        frame_path_to_subset[path] = {set_}

        # pyre-ignore[16]
        for frame in self.frame_annots:
            frame["subset"] = frame_path_to_subset.get(
                frame["frame_annotation"].image.path, None
            )

            if frame["subset"] is None:
                continue
                warnings.warn(
                    "Subset lists are given but don't include "
                    + frame["frame_annotation"].image.path
                )

    def _sort_frames(self) -> None:
        # Sort frames to have them grouped by sequence, ordered by timestamp
        # pyre-ignore[16]
        self.frame_annots = sorted(
            self.frame_annots,
            key=lambda f: (
                f["frame_annotation"].sequence_name,
                f["frame_annotation"].frame_timestamp or 0,
            ),
        )

    def _filter_db(self) -> None:
        if self.remove_empty_masks:
            logger.info("Removing images with empty masks.")
            # pyre-ignore[16]
            old_len = len(self.frame_annots)

            msg = "remove_empty_masks needs every MaskAnnotation.mass to be set."

            def positive_mass(frame_annot: types.FrameAnnotation) -> bool:
                mask = frame_annot.mask
                if mask is None:
                    return False
                if mask.mass is None:
                    raise ValueError(msg)
                return mask.mass > 1

            self.frame_annots = [
                frame
                for frame in self.frame_annots
                if positive_mass(frame["frame_annotation"])
            ]
            logger.info("... filtered %d -> %d" % (old_len, len(self.frame_annots)))

        # this has to be called after joining with categories!!
        subsets = self.subsets
        if subsets:
            if not self.subset_lists_file:
                raise ValueError(
                    "Subset filter is on but subset_lists_file was not given"
                )

            logger.info(f"Limiting Co3D dataset to the '{subsets}' subsets.")

            # truncate the list of subsets to the valid one
            self.frame_annots = [
                entry
                for entry in self.frame_annots
                if (entry["subset"] is not None and self.stage in entry["subset"])
            ]

            if len(self.frame_annots) == 0:
                raise ValueError(f"There are no frames in the '{subsets}' subsets!")

            self._invalidate_indexes(filter_seq_annots=True)

        if len(self.limit_category_to) > 0:
            logger.info(f"Limiting dataset to categories: {self.limit_category_to}")
            # pyre-ignore[16]
            self.seq_annots = {
                name: entry
                for name, entry in self.seq_annots.items()
                if entry.category in self.limit_category_to
            }

        # sequence filters
        for prefix in ("pick", "exclude"):
            orig_len = len(self.seq_annots)
            attr = f"{prefix}_sequence"
            arr = getattr(self, attr)
            if len(arr) > 0:
                logger.info(f"{attr}: {str(arr)}")
                self.seq_annots = {
                    name: entry
                    for name, entry in self.seq_annots.items()
                    if (name in arr) == (prefix == "pick")
                }
                logger.info("... filtered %d -> %d" % (orig_len, len(self.seq_annots)))

        if self.limit_sequences_to > 0:
            self.seq_annots = dict(
                islice(self.seq_annots.items(), self.limit_sequences_to)
            )

        # retain only frames from retained sequences
        self.frame_annots = [
            f
            for f in self.frame_annots
            if f["frame_annotation"].sequence_name in self.seq_annots
        ]

        self._invalidate_indexes()

        if self.n_frames_per_sequence > 0:
            logger.info(f"Taking max {self.n_frames_per_sequence} per sequence.")
            keep_idx = []
            # pyre-ignore[16]
            for seq, seq_indices in self._seq_to_idx.items():
                # infer the seed from the sequence name, this is reproducible
                # and makes the selection differ for different sequences
                seed = _seq_name_to_seed(seq) + self.seed
                seq_idx_shuffled = random.Random(seed).sample(
                    sorted(seq_indices), len(seq_indices)
                )
                keep_idx.extend(seq_idx_shuffled[: self.n_frames_per_sequence])

            logger.info(
                "... filtered %d -> %d" % (len(self.frame_annots), len(keep_idx))
            )
            self.frame_annots = [self.frame_annots[i] for i in keep_idx]
            self._invalidate_indexes(filter_seq_annots=False)
            # sequences are not decimated, so self.seq_annots is valid

        if self.limit_to > 0 and self.limit_to < len(self.frame_annots):
            logger.info(
                "limit_to: filtered %d -> %d" % (len(self.frame_annots), self.limit_to)
            )
            self.frame_annots = self.frame_annots[: self.limit_to]
            self._invalidate_indexes(filter_seq_annots=True)

    def _invalidate_indexes(self, filter_seq_annots: bool = False) -> None:
        # update _seq_to_idx and filter seq_meta according to frame_annots change
        # if filter_seq_annots, also uldates seq_annots based on the changed _seq_to_idx
        self._invalidate_seq_to_idx()

        if filter_seq_annots:
            # pyre-ignore[16]
            self.seq_annots = {
                k: v
                for k, v in self.seq_annots.items()
                # pyre-ignore[16]
                if k in self._seq_to_idx
            }

    def _invalidate_seq_to_idx(self) -> None:
        seq_to_idx = defaultdict(list)
        # pyre-ignore[16]
        for idx, entry in enumerate(self.frame_annots):
            seq_to_idx[entry["frame_annotation"].sequence_name].append(idx)
        # pyre-ignore[16]
        self._seq_to_idx = seq_to_idx

    def _resize_image(
        self, image, mode="bilinear"
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        image_height, image_width = self.image_height, self.image_width
        if image_height is None or image_width is None:
            # skip the resizing
            imre_ = torch.from_numpy(image)
            return imre_, 1.0, torch.ones_like(imre_[:1])
        # takes numpy array, returns pytorch tensor
        minscale = min(
            image_height / image.shape[-2],
            image_width / image.shape[-1],
        )
        imre = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None],
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]
        # pyre-fixme[19]: Expected 1 positional argument.
        imre_ = torch.zeros(image.shape[0], self.image_height, self.image_width)
        imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
        # pyre-fixme[6]: For 2nd param expected `int` but got `Optional[int]`.
        # pyre-fixme[6]: For 3rd param expected `int` but got `Optional[int]`.
        mask = torch.zeros(1, self.image_height, self.image_width)
        mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
        return imre_, minscale, mask

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int]
    ) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in idxs:
            # pyre-ignore[16]
            frame_annotation = self.frame_annots[idx]["frame_annotation"]
            out.append(
                (frame_annotation.frame_number, frame_annotation.frame_timestamp)
            )
        return out

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return self.eval_batches

    def _get_frame_type(self, entry: FrameAnnotsEntry) -> Optional[str]:
        return entry["frame_annotation"].meta["frame_type"]


class CO3DDataset(LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=2,
        shuffle=True,
        num_workers=10,
        prefetch_factor=2,
        category="hydrant",
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle

        self.train_dataset = CO3Dv2Wrapper(
            root_dir=root_dir,
            stage="train",
            category=category,
            **kwargs,
        )

        self.test_dataset = CO3Dv2Wrapper(
            root_dir=root_dir,
            stage="test",
            subset="fewview_dev",
            category=category,
            **kwargs,
        )

    def train_dataloader(self):
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
