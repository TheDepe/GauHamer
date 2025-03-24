import cv2
import os
import numpy as np
import torch
import json
from natsort import natsorted
import h5py
import joblib
import math
from utils.manus_cam_utils import *
from utils.manus_transforms import *
from utils.manus_extra import *
from utils import manus_params as param_utils
from typing import DefaultDict
from utils.manus_structures import Cameras

class Manus(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(self, root_dir, split="train"):
        self.resize_factor = 1.0 # opts.resize_factor
        self.split = split
        self.near = 0.01 #opts.near
        self.far = 100 #opts.far
        self.training = split == "train"
        self.bg_color = "white" #opts.bg_color
        self.subject_id = 'angela' #opts.subject
        self.width = 1280 #opts.width
        self.height = 720 #opts.height

        root_fp = root_dir

        self.dtype = torch.get_default_dtype()
        self.root_dir = root_fp
        self.actions, self.index_list, self.metadata_dict, to_choose = (
            self.dataset_index_list(
                self.root_dir,
                split,
                1, #self.opts.num_time_steps,
                0.99, #self.opts.split_ratio,
                -1 #self.opts.rand_views_per_timestep,
            )
        )

        # Compute the cameras once
        self.get_all_cameras(to_choose)
        # self.print_data_stats()

        super().__init__()

    def dataset_index_list(
        self, root_dir, split, num_time_steps, split_ratio, rand_views_per_timestep
    ):
        actions = natsorted([fp for fp in os.listdir(root_dir)])
        if False: #self.opts.sequences != "all":
            chosen_actions = []
            for action in self.opts.sequences:
                action = f"{action}.hdf5"
                if action in actions:
                    chosen_actions.append(action)
            actions = chosen_actions

        if len(actions) == 1 and False: #self.opts.split_by_action:
            split_ratio = -1

        if (split_ratio > 0) and False: #self.opts.split_by_action:
            if split == "train":
                actions = actions[: int(split_ratio * len(actions))]
            else:
                actions = actions[int(split_ratio * len(actions)) :]

        index_list = []
        metadata_dict = {}
        for idx, action_path in enumerate(actions):
            action = action_path.split(".")[0]
            metadata_dict[action] = {}
            h5_path = os.path.join(root_dir, action_path)
            with h5py.File(
                h5_path,
                "r",
            ) as file:
                frame_nos = list(file["frames"].keys())
                Ks = file.get("K")
                cam_names = list(Ks.keys())

                # Make a metadata dict here only

                # for fno in frame_nos:
                #     metadata = file["frames"][fno]["metadata"]
                #     metadata = self.fetch_metadata(metadata)
                #     metadata["frame_id"] = fno
                #     metadata["action"] = action
                #     metadata_dict[action][fno] = metadata

            frame_nos = natsorted(frame_nos)

            if (num_time_steps < 0) or (num_time_steps > len(frame_nos)):
                to_choose = frame_nos
            else:
                to_choose = frame_nos[:: (len(frame_nos) // num_time_steps)]

            for fno in to_choose:
                if rand_views_per_timestep < 0:
                    for view in cam_names:
                        index_list.extend([(action, fno, view)])
                else:
                    index_list.extend([(action, fno, None)])

        if not False: #self.opts.split_by_action:
            if split_ratio > 0:
                if split == "train":
                    index_list = index_list[: int(split_ratio * len(index_list))]
                else:
                    index_list = index_list[int(split_ratio * len(index_list)) :]

            with open(f"./{split}_split.json", "w") as f:
                json.dump(index_list, f)

        return actions, index_list, metadata_dict, to_choose

    def get_all_cameras(self, to_choose):
        ## Cameras are not changing for the actions
        action = self.index_list[0][0]

        h5_path = os.path.join(self.root_dir, f"{action}.hdf5")
        d_dict = DefaultDict(list)

        mano_data = {}
        with h5py.File(
            h5_path,
            "r",
        ) as file:
            mano = file.get("mano_rest")
            for k, v in mano.items():
                mano_data[k] = v[:]

            frames = file.get("frames")
            Ks = file.get("K")
            self.cam_names = list(Ks.keys())
            self.cam2idx = {
                cam_name: idx for idx, cam_name in enumerate(self.cam_names)
            }
            extrs = file.get("extr")
            frames_list = list(frames.keys())
            self.all_cameras = []

            ## Loading all the choosen timesteps
            for _ in to_choose:
                for cam_name in self.cam_names:
                    K = Ks[cam_name][:]
                    extr = extrs[cam_name][:]
                    extr = np.concatenate([extr, np.array([[0, 0, 0, 1]])], axis=0)
                    attr_dict = get_opengl_camera_attributes(
                        K,
                        extr,
                        self.width,
                        self.height,
                        resize_factor=self.resize_factor,
                    )
                    for k, v in attr_dict.items():
                        d_dict[k].append(v)
                    d_dict["cam_name"].append(cam_name)

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)
        self.extent = get_scene_extent(self.all_cameras.camera_center)
        self.mano_data = mano_data

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        data = self.fetch_data(idx)
        return data

    def fetch_data_by_frame(self, action, frame_id, cam_name):
        try:
            idx = self.index_list.index((action, frame_id, cam_name))
            data = self.fetch_data(idx)
        except:
            data = None
        return data

    def get_roi_mask(self, img, roi):
        roi_mask = np.zeros_like(img[..., 3])
        roi_mask[roi[1] : roi[3], roi[0] : roi[2]] = 255
        return roi_mask

    def get_bg_color(self):
        if self.bg_color == "random":
            color_bkgd = np.random.rand(3).astype(np.float32)
        elif self.bg_color == "white":
            color_bkgd = np.ones(3).astype(np.float32)
        elif self.bg_color == "black":
            color_bkgd = np.zeros(3).astype(np.float32)
        return color_bkgd

    def fetch_images(self, data, cam_name):
        images = data["images"]
        bboxes = data["bbox"]
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        crop_img = images[cam_name][:]
        bbox = bboxes[cam_name][:]
        xmin, ymin, xmax, ymax = bbox
        try:
            img[ymin:ymax, xmin:xmax, :] = crop_img
        except:
            breakpoint()
        # roi_mask = self.get_roi_mask(img, self.rois[idx])
        # modify_mask = cv2.bitwise_and(roi_mask, img[..., 3])
        # img[..., 3] = modify_mask.astype(np.uint8)
        img = cv2.resize(
            img,
            (0, 0),
            fx=self.resize_factor,
            fy=self.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        img = img / 255.0

        color_bkgd = self.get_bg_color()

        if img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:]
            img[..., :3] = rgb * alpha + color_bkgd * (1.0 - alpha)

        return img

    def get_data_from_h5(self, index):
        action, frame_id, cam_name = self.index_list[index]

        ## Randomly Choose cameras for each timestep
        if cam_name is None:
            cam_name = np.random.default_rng().choice(
                self.cam_names, size=-1, replace=False #self.opts.rand_views_per_timestep, replace=False
            )
        else:
            cam_name = [cam_name]

        h5_path = os.path.join(self.root_dir, f"{action}.hdf5")
        with h5py.File(
            h5_path,
            "r",
        ) as file:
            frames = file.get("frames")
            data = frames[str(frame_id)]
            rgba_list = []
            camidx = []
            for cam in cam_name:
                rgba = self.fetch_images(data, cam)
                rgba_list.append(rgba)
                camidx.append(self.cam2idx[cam])
                
        rgba_list = np.array(rgba_list)
        cameras = self.all_cameras[camidx]
        info = [self.subject_id, action, frame_id, cam_name]
        return rgba_list, cameras, info

    def fetch_data(self, index):
        rgba, camera, info = self.get_data_from_h5(index)
        rgba = np.stack(rgba, axis=0)
        image = rgba[..., :3]
        mask = rgba[..., 3:]
        color_bkgd = self.get_bg_color()

        data_dict = {
            "info": info,
            "rgb": to_tensor(image),
            "mask": to_tensor(mask),
            "camera": to_tensor(camera),
            "scaling_modifier": 1.0,
            "bg_color": to_tensor(color_bkgd),
        }
        focal_pixels = torch.tensor([fov2focal(camera.fovx, 256),
                                                        fov2focal(camera.fovy, 256)])

        # has extra dim for some reason
        cx = camera.K[0, 0, 2] 
        cy = camera.K[0, 1, 2]
        pps_pixels = torch.tensor([cx * 256 / 2,
                                                        cy * 256 / 2])
        #crop_infos = torch.tensor(cam_info.crop_info)
        ret = {
            "gt_images": to_tensor(image),
            "world_view_transforms": camera.world_view_transform,
            "view_to_world_transforms": np.linalg.inv(camera.world_view_transform),
            "full_proj_transforms": camera.full_proj_transform,
            "camera_centers": camera.camera_center,
            #"points_3d": torch.tensor(points_3d),
            "cam_ids": camera.cam_name, 
            "background_color": color_bkgd,
            "gt_masks": np.zeros(3),
            #"subject": torch.tensor(subject), #object ID
            "focals_pixels": focal_pixels,
            "pps_pixels": pps_pixels,
            "crops_info": np.zeros(3),
        }
        print(type(val) for k,val in ret.items())
        return ret

    def get_all_images_per_frame(self, action, frame_id):
        images = []
        for cam in self.cam_names:
            idx = self.index_list.index((action, frame_id, cam))
            rgba, camera, metadata_dict, info = self.get_data_from_h5(idx)
            rgba = rgba / 255.0
            images.append(rgba)
        images = np.array(images)
        return images

    def get_prune_mask(self, points):
        points_mask = np.zeros(points.shape[0])

        action, frame_id = "grasp_mug", "390"
        images = self.get_all_images_per_frame(action, frame_id)

        for idx, cam in enumerate(self.all_cameras):
            l_mask = np.zeros_like(points_mask)
            cam = to_tensor(cam)
            K, R, T = cam.K, cam.R, cam.T
            extrin = torch.cat([R, T[..., None]], dim=1)
            p2d = project_points(points[None], K, extrin)[0]
            mask = images[idx][..., -1]

            x, y = p2d[..., 0], p2d[..., 1]
            mask_x = torch.logical_and(x >= 0, x < self.width)
            mask_y = torch.logical_and(y >= 0, y < self.height)
            mask_xy = torch.logical_and(mask_x, mask_y)
            masked_p2d = p2d[mask_xy]
            masked_p2d = to_numpy(masked_p2d)
            masked_p2d = masked_p2d.astype(np.int32)
            nmask = mask[masked_p2d[..., 1], masked_p2d[..., 0]] == 1
            indices = np.where(mask_xy)[0][nmask]

            l_mask[indices] = 1
            points_mask += l_mask

            ## Visualize 2D points
            # vis = np.zeros((self.height, self.width, 3))
            # vis[masked_p2d[..., 1], masked_p2d[..., 0]] = [1, 1, 1]
            # os.makedirs('./vis', exist_ok=True)
            # cv2.imwrite(f'./vis/{idx}.png', vis * 255)
            # cv2.imwrite(f'./vis/{idx}_mask.png', mask * 255)

        ## Prune all points which are not visible most of the cameras
        points_mask = points_mask < 35
        dump_points(points[points_mask == 0], "./prune_points.ply")
        # breakpoint()
        return points_mask

    @classmethod
    def encode_meta_id(cls, action, frame_id):
        return "%s___%05d" % (action, int(frame_id))

    @classmethod
    def decode_meta_id(cls, meta_id: str):
        action, frame_id = meta_id.split("___")
        return action, int(frame_id)

