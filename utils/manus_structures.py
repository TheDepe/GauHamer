from dataclasses import dataclass
import numpy as np
import torch

from utils.graphics_utils import getProjectionMatrix
@dataclass
class Cameras:
    cam_name: np.ndarray
    K: np.ndarray
    extr: np.ndarray
    fovx: float
    fovy: float
    width: int
    height: int
    world_view_transform: np.ndarray
    projection_matrix: np.ndarray
    full_proj_transform: np.ndarray
    camera_center: np.ndarray

    def __getitem__(self, idx):
        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = value[idx]
        return Cameras(**new_dict)