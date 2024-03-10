import torch
from torch.utils.data import Dataset

default_sub_data_config = {}


class Joint3D(Dataset):
    def __init__(self, sub_data_config: dict) -> None:
        super().__init__()
        self.sub_data_config = sub_data_config
