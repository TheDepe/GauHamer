import imageio
import glob
import os
from tqdm import tqdm
import json
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import copy
import smplx
import yaml
import cv2
from PIL import Image
import imageio

from .dataset_readers import readCamerasInfosMIRAGE, readCamerasInfosHumman
from utils.general_utils import PILtoTorch, PILtoTorchHMR, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World, fov2focal
from utils.manus_cam_utils import read_params, get_intr, get_extr, get_undistort_params, get_opengl_camera_attributes

MIRAGE_DATASET_ROOT = '/graphics/scratch2/students/perrettde/DATA/mirage_renders/newest_renders' # Change this to your data directory
assert MIRAGE_DATASET_ROOT is not None, "Update the location of the MIRAGE Dataset"
MANO_DIR='/graphics/scratch2/students/perrettde/DATA/SMLP_MANO/models/'


@dataclass
class Camera:
    R: torch.tensor
    T: torch.tensor


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def compute_3d_bounds(vertices):
    # obtain the original bounds for point sampling
    min_xyz = np.min(vertices, axis=0)
    max_xyz = np.max(vertices, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    world_bounds = np.stack([min_xyz, max_xyz], axis=0)
    return world_bounds

def get_mask(vertices, H, W, K, R, T):
    world_bounds = compute_3d_bounds(vertices)
    
    pose = np.concatenate([R, T[:, None]], axis=1)
    
    bound_mask = get_bound_2d_mask(world_bounds, K, pose, H, W)
    return bound_mask


class MIRAGE(Dataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.cam_data_path = cfg.data.cam_data_path
        self.base_path = os.path.join(MIRAGE_DATASET_ROOT)

        if hasattr(self.cfg.data, "overfit"):
            self.overfit = self.cfg.data.overfit 
        else:
            self.overfit = False

        if hasattr(self.cfg.data, "num_training_images"):
            self.num_training_images = cfg.data.num_training_images
        else:
            self.num_training_images = 4

        if self.overfit:
            self.dataset_name = "overfit"

        
        filename = "train.txt" if self.dataset_name == "train" else "test.txt"

        filename = "train_overfit.txt" if self.dataset_name == "overfit" else filename 
        
        with open(os.path.join(self.base_path, filename)) as file:
            self.folders = [line.rstrip() for line in file]
            
        if self.dataset_name != "train":
            self.num_training_images = cfg.data.num_test_val_images
        
        if self.dataset_name == "overfit":
            self.num_training_images = cfg.data.num_training_images
            
        if hasattr(self.cfg.data, "undistort"):
            self.undistort = self.cfg.data.undistort
        else:
            self.undistort = False

        self.cameras_jsons = {}
        self.projection_matrix = getProjectionMatrix(
            znear=cfg.data.znear, zfar=cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        #self.mano = smplx.create(
        #    model_path=os.path.join(MANO_DIR, "MANO_RIGHT.pkl"),
        #    model_type='mano',
        #    gender='neutral',
        #    hand='right',
        #    use_pca=False)
        
        self.items = self.process_folders()
        
    def __len__(self):
        return len(self.items)
    
    @staticmethod
    def get_numpy(hfile, key):
        val = hfile.get(key)
        return np.array(val)
    
    def load_joints(self, folder_name):
        joints_path = os.path.join(self.base_path, folder_name, "processed_joints.npy")
        joints_array = np.load(joints_path)
        return joints_array

    def load_annots(self, folder_name):
        annots_path = os.path.join(self.base_path, folder_name, "annots.npy")
        annots = np.load(annots_path, allow_pickle=True).item()
        cams = annots["cams"]
        ims = annots["ims"]
        return cams, ims

    def parse_cameras(self, cam_data_path):
        cam_data = read_params(cam_data_path) # returns dict of params for each camera
        
        # reformat into to work with this method
        parsed_cameras = {}
        for camera_id, params in cam_data.items():
        
            extr = get_extr(params) # c2w
            intr, dist = get_intr(params)
            undistort = True
            if undistort:
                new_intr, roi = get_undistort_params(
                    intr, dist, (params['width'], params['height'])
                )
                intr = new_intr
            
            # extr is c2w
            attr_dict = get_opengl_camera_attributes(
                new_intr,
                extr,
                1280,
                720,
                resize_factor=1
            )
            # Match rotation/coord system of cameras with rotation of mano
            cam_rotation = np.array([[-1., -0., -0.,  0.],
                            [ 0., -0.,  1.,  0.],
                            [-0.,  1.,  0.,  0.],
                            [ 0.,  0.,  0.,  1.]])
            
            extr = np.linalg.inv(cam_rotation @ np.linalg.inv(extr))
            parsed_cameras[camera_id] = {
                "extrins": extr,
                "intrins": intr,
                "ogl_atrs": attr_dict
            }
                    
        return parsed_cameras

    def process_folders(self):
        items = []
        for i, folder in enumerate(self.folders):
            object_id = folder.split(os.sep)[0]
            object_id = i
            

            parsed_cameras = self.parse_cameras(self.cam_data_path)
            cam_folders = sorted([f for f in os.listdir(os.path.join(self.base_path, folder)) 
                      if os.path.isdir(os.path.join(self.base_path, folder, f))])
            cam_folders = sorted([os.path.join(self.base_path, folder, cam_folder) for cam_folder in cam_folders])

            images_info = []
            for i, perspective in enumerate(cam_folders):
                cam_id = perspective.split(os.sep)[-1]
                metadata = True
                if metadata:
                    mano_params_path = os.path.join(self.base_path, folder, 'metadata.json')
                    with open(mano_params_path, 'r') as f:
                        metadata = json.load(f)
                    mano_params = metadata['grasp_metadata']['MANO']
                    global_orient = np.array(mano_params['rotation'])
                    body_pose = np.array(mano_params['adjusted_pose'][0])
                    betas = np.zeros(10)
                    transl = np.array(mano_params['trans'])
                    
                    global_orient = torch.Tensor(global_orient)
                    body_pose = torch.Tensor(body_pose)
                    betas = torch.Tensor(betas)
                    transl = torch.Tensor(transl)
                
                    mano_params = {
                        "global_orient": global_orient,
                        "pose": body_pose,
                        "betas": betas,
                        "transl": transl
                    }
                else:
                    mano_params = None
                
                ## compute SMPL vertices in the world coordinate system
                #output = self.mano(
                #    betas=torch.Tensor(betas).view(1, 10),
                #    hand_pose=torch.Tensor(body_pose).view(1, 45),
                #    global_orient=torch.Tensor(global_orient).view(1, 3),
                #    transl=torch.Tensor(transl).view(1, 3),
                #    return_verts=True
                #)
                #joints_3d = output.vertices.detach().numpy().squeeze()
                
                
                frame_path = os.path.join(self.base_path, folder, perspective, "hand_rgb.webp")
                camera_idx = i
                info_dict = {
                    "camera_id": camera_idx,
                    "camera_pose": parsed_cameras[cam_id]["extrins"], #c2w
                    "intrinsics": parsed_cameras[cam_id]["intrins"],
                    "frame_path": frame_path
                }
                
                info_dict.update(parsed_cameras[cam_id]["ogl_atrs"])
                images_info.append(info_dict)
                if i == 0:
                    images_info.append(copy.deepcopy(info_dict)) # copy input image to the end, so that it is used as context too?
            
            # Append each scene.
            #items.append((images_info, joints_3d, object_id, mano_params)) 
            items.append((images_info, None, object_id, mano_params))
            #items.append((images_info, None, object_id, None))

        return items

    def load_example_id(self, index,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        
        views, points_3d, subject, mano_params = self.items[index]

                
        all_rgbs = []
        all_world_view_transforms = []
        all_full_proj_transforms = []
        all_camera_centers = []
        all_view_to_world_transforms = []
        cam_ids = []
        masks = []
        background_colors = []
        all_focals_pixels = []
        pps_pixels = []
        crop_infos = []
        sherf_masks = []

        cam_ids = []
        cam_poses = []
        rgb_paths = []
        intrinsics = []
        
        # Select which camera views to take
        if self.overfit:
            indices = torch.randperm(len(views)-1)
            indices = [0, 0] + indices.tolist()
        else:
            if self.dataset_name == "train":
                indices = torch.randperm(len(views))
            else:
                indices = torch.arange(len(views))
            indices = torch.concatenate((indices[:self.cfg.data.input_images], indices)) # add GT to beginning
            indices = indices.tolist()
        
        
        # VERIFIED VALUES
        cam_perspectives = [views[i] for i in indices]
        cam_centers = []
        full_proj_transforms = []
        cam_world_view_transforms = []
        cam_extrins = []
        cam_projection_matrices = []

        
        # Add desired cams to list.
        for cam in cam_perspectives:
            cam_ids.append(cam["camera_id"]) 
            cam_poses.append(cam["camera_pose"]) # c2w
            rgb_paths.append(cam["frame_path"])
            intrinsics.append(cam["intrinsics"])
            cam_centers.append(cam['camera_center'])
            cam_world_view_transforms.append(cam["world_view_transform"])
            full_proj_transforms.append(cam['full_proj_transform'])
            cam_extrins.append(np.linalg.inv(cam['world_view_transform']))
            cam_projection_matrices.append(cam['projection_matrix'])


        random_background_color = self.cfg.data.random_background_color if hasattr(self.cfg.data, "random_background_color") else False
        background_color = 255 if hasattr(self.cfg.data, "white_background") and self.cfg.data.white_background == True else 0
        tight_crop = self.cfg.data.get("cropped", False) # False
        
        cam_infos = []
        i = 0
        total_images = self.cfg.data.input_images + self.num_training_images
        accepted_cam_ids = []
        while len(cam_infos) < total_images:
            try:
                cam_infos += readCamerasInfosMIRAGE([rgb_paths[i]], [cam_poses[i]], [intrinsics[i]], None, [cam_ids[i]],
                                                    random_background_color=random_background_color, 
                                                    bg_color=background_color, tight_crop=tight_crop)
                accepted_cam_ids.append(cam_ids[i])
            except ValueError as e:
                # Can't load something. Move to next image view point
                pass
                
            i += 1
            if i == len(rgb_paths):
                # Reached all avaiable images
                #print("Reached Total Available images")
                break
            
        if len(cam_infos)==0:
            raise ValueError("Couldn't find any images")

               
               
        for i, cam_info in enumerate(cam_infos):
            R = cam_info.R
            T = cam_info.T

            # Image size here is 720px x 720px
            if self.cfg.data.get("hmr_preprocessing", False) and i==0:
                all_rgbs.append(PILtoTorchHMR(cam_info.image, 
                (self.cfg.data.training_resolution, self.cfg.data.training_resolution))[:3, :, :])
            else:
                all_rgbs.append(PILtoTorch(cam_info.image, 
                (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])
            # Here it is now 256x256px

            h_min, w_min, crop_size, h, w = cam_info.crop_info
            #sherf_mask = get_mask(points_3d, h, w, intrinsics[i], np.transpose(R), T)
            
            #sherf_mask = sherf_mask[h_min:h_min+crop_size, w_min:w_min+crop_size]
            #sherf_mask = Image.fromarray(np.repeat(sherf_mask[:, :, None], 3, axis=-1)*(255//np.max(sherf_mask)))
            #sherf_masks.append(PILtoTorch(sherf_mask, (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

            world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1) # CORRECT
            view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

            projection_matrix = getProjectionMatrix(
                znear=self.cfg.data.znear,
                zfar=self.cfg.data.zfar,
                fovX=cam_info.FovX,
                fovY=cam_info.FovY,
                pX=cam_info.px,
                pY=cam_info.py
            ).transpose(0, 1)

            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3] 
            
            all_world_view_transforms.append(world_view_transform) 
            all_view_to_world_transforms.append(view_world_transform) 
            all_full_proj_transforms.append(full_proj_transform) 
            all_camera_centers.append(camera_center)
            background_colors.append(torch.tensor(cam_info.color/255, dtype=torch.float32)) 
            masks.append(PILtoTorch(cam_info.mask, 
                    (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :][0, ...])
            all_focals_pixels.append(torch.tensor([fov2focal(cam_info.FovX, self.cfg.data.training_resolution),
                                                                fov2focal(cam_info.FovY, self.cfg.data.training_resolution)]))
            pps_pixels.append(torch.tensor([cam_info.px * self.cfg.data.training_resolution / 2,
                                                        cam_info.py * self.cfg.data.training_resolution / 2]))
            crop_infos.append(torch.tensor(cam_info.crop_info))
            


        all_world_view_transforms = torch.stack(all_world_view_transforms)
        all_view_to_world_transforms = torch.stack(all_view_to_world_transforms)
        all_full_proj_transforms = torch.stack(all_full_proj_transforms)
        all_camera_centers = torch.stack(all_camera_centers)
        all_rgbs = torch.stack(all_rgbs)
        cam_ids = torch.tensor(accepted_cam_ids)
        background_colors = torch.stack(background_colors)
        masks = torch.stack(masks)
        all_focals_pixels = torch.stack(all_focals_pixels)
        pps_pixels = torch.stack(pps_pixels)
        crop_infos = torch.stack(crop_infos)
        #sherf_masks =  torch.stack(sherf_masks)

        # for cam, loc in zip(cam_ids, all_focals_pixels):
        #     print(f"Cam {cam} is at {loc}")
        ret = {
            "gt_images": all_rgbs,
            "world_view_transforms": all_world_view_transforms,
            "view_to_world_transforms": all_view_to_world_transforms,
            "full_proj_transforms": all_full_proj_transforms,
            "camera_centers": all_camera_centers,
            #"points_3d": torch.tensor(points_3d),
            "cam_ids": cam_ids, 
            "background_color": background_colors,
            "gt_masks": masks,
            #"subject": torch.tensor(subject), #object ID
            "focals_pixels": all_focals_pixels,
            "pps_pixels": pps_pixels,
            "crops_info": crop_infos,
        }
        #ret["sherf_masks"] = sherf_masks
        if mano_params is not None:
            ret["global_orient"] = mano_params["global_orient"]
            ret["pose"] = mano_params["pose"]
            ret["betas"] = mano_params["betas"]
            ret["transl"] = mano_params["transl"]
        return ret

    def get_example_id(self, index):
        return str(index)

    
    def get_source_cw2wT(self, source_cameras_view_to_world):
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)

    def __getitem__(self, index):
        images_and_camera_poses = self.load_example_id(index)

        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses