import os
from HaMeR.hamer.models import HAMER
from HaMeR.hamer.models.heads.mano_head import MANOTransformerDecoderHead
from HaMeR.hamer.configs import get_config
import torch.nn as nn
import torch
import einops
from utils.general_utils import quaternion_raw_multiply
import math
from torchvision.transforms.functional import resize
from utils.general_utils import batch_rotmat_to_aa


#CACHE_DIR = os.path.join('/path/to/unzipped/weights/', ".cache") # CHANGE TO YOUR PATH WITH PRETRAINED HAMER WEIGHTS
#CACHE_DIR_HAMER = os.path.join(CACHE_DIR, "HaMeR") # RENAME
#DEFAULT_CHECKPOINT = f'{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt'

CACHE_DIR = os.path.join('/graphics/scratch2/students/perrettde/MODELS/GausHamer/', ".cache")
CACHE_DIR_HAMER = os.path.join(CACHE_DIR, "HaMeR") # RENAME
DEFAULT_CHECKPOINT = f'{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt'

class GaussiansHead(MANOTransformerDecoderHead):
    """ This just extends the based MANOTransformerDecoderHead from HAMER.

        Input is Transformer Input
        Standard Output: 
            Mano params (pose, betas)
            Cam
        Extended Output:
            Gaussian Features    
    """
    def __init__(self, cfg, mano_cfg):
        super().__init__(mano_cfg)

        self.gaussians_per_vertex = cfg.model.get("gaussians_per_vertex", 1)
        self.num_vertices = 778 * self.gaussians_per_vertex
        print("\n--------------------------------------------------------------\n")
        print(f"Initialising model with {self.num_vertices} output gaussians.")
        print("\n--------------------------------------------------------------\n")
        dim = self.decpose.in_features
        self.num_tasks = 5
        self.num_vertices_chunks = 2
        self.num_vertices_per_token = self.num_vertices // self.num_vertices_chunks
        self.num_tokens = self.num_tasks * self.num_vertices_chunks
        assert self.num_vertices % self.num_vertices_chunks == 0

        self.decfeatures_out = nn.Linear(dim, self.num_vertices_per_token * 3)

        self.decrot_out = nn.Linear(dim, self.num_vertices_per_token * 4)

        self.decopacity_out = nn.Linear(dim, self.num_vertices_per_token)

        self.decscale_out = nn.Linear(dim, self.num_vertices_per_token * 3)

        self.decoffset_out = nn.Linear(dim, self.num_vertices_per_token * 3)

        self.vertices_positional_encoding = nn.Parameter(torch.rand(1, 1, 44, 1))

        self.tokens = nn.Parameter(torch.rand(self.num_tokens, dim))

        if cfg.model.get('init_gaussians_head', True):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decfeatures_out.weight, cfg.model.get("features_scale", 0.01))
            nn.init.constant_(self.decfeatures_out.bias, cfg.model.get("features_bias", 0.0))
            nn.init.xavier_uniform_(self.decrot_out.weight, cfg.model.get("rotation_scale", 0.01))
            nn.init.constant_(self.decrot_out.bias, cfg.model.get("rotation_bias", 0.0))
            nn.init.xavier_uniform_(self.decopacity_out.weight, cfg.model.get("opacity_scale", 0.01))
            nn.init.constant_(self.decopacity_out.bias, cfg.model.get("opacity_bias", 0.0))
            nn.init.xavier_uniform_(self.decscale_out.weight, cfg.model.get("scale_scale", 0.01))
            nn.init.constant_(self.decscale_out.bias, math.log(cfg.model.get("scale_bias", 0.0)))
            nn.init.xavier_uniform_(self.decoffset_out.weight, cfg.model.get("xyz_scale", 0.01))
            nn.init.constant_(self.decoffset_out.bias, cfg.model.get("xyz_bias", 0.0))
        
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if cfg.model.get("pred_opacity", True):
            self.pred_opacity = True
        else:
            self.pred_opacity = False

    def forward(self, x):
        batch_size = x.shape[0]
    
        pred_mano_params, pred_cam, _ = super().forward(x)

        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        token = self.tokens.repeat(batch_size, 1, 1).to(x.device)
        token_out = self.transformer.transformer(token, context=x)

        pred_features_dc = token_out[:, :1*self.num_vertices_chunks, :]
        pred_features_dc = self.decfeatures_out(pred_features_dc)
        pred_features_dc = pred_features_dc.reshape(batch_size, self.num_vertices_chunks, self.num_vertices_per_token * 3)
        pred_features_dc = pred_features_dc.reshape(batch_size, self.num_vertices, 3)

        pred_rotation = token_out[:, 1*self.num_vertices_chunks:2*self.num_vertices_chunks, :]
        pred_rotation = self.decrot_out(pred_rotation)
        pred_rotation = pred_rotation.reshape(batch_size, self.num_vertices_chunks, self.num_vertices_per_token * 4)
        pred_rotation = pred_rotation.reshape(batch_size, self.num_vertices, 4)

        pred_opacity = token_out[:, 2*self.num_vertices_chunks:3*self.num_vertices_chunks, :]
        pred_opacity = self.decopacity_out(pred_opacity)
        pred_opacity = pred_opacity.reshape(batch_size, self.num_vertices_chunks, self.num_vertices_per_token * 1)
        pred_opacity = pred_opacity.reshape(batch_size, self.num_vertices, 1)

        pred_scale = token_out[:, 3*self.num_vertices_chunks:4*self.num_vertices_chunks, :]
        pred_scale = self.decscale_out(pred_scale)
        pred_scale = pred_scale.reshape(batch_size, self.num_vertices_chunks, self.num_vertices_per_token * 3)
        pred_scale = pred_scale.reshape(batch_size, self.num_vertices, 3)

        pred_offset = token_out[:, 4*self.num_vertices_chunks:, :]
        pred_offset = self.decoffset_out(pred_offset)
        pred_offset = pred_offset.reshape(batch_size, self.num_vertices_chunks, self.num_vertices_per_token * 3)
        pred_offset = pred_offset.reshape(batch_size, self.num_vertices, 3)

        if self.pred_opacity:
            pred_opacity = self.opacity_activation(pred_opacity)
        else:
            pred_opacity = torch.ones_like(pred_opacity)
        return {
            "xyz": pred_offset, 
            "opacity": pred_opacity,
            "scaling": self.scaling_activation(pred_scale),
            "rotation": self.rotation_activation(pred_rotation),
            "features_dc": pred_features_dc.unsqueeze(-2),
            "pred_mano_params": pred_mano_params,
            "pred_cam": pred_cam
        }


class GaussianHaMeR(HAMER):
    def __init__(self, cfg, mano_cfg):
        super().__init__(mano_cfg, init_renderer=False)
        self.mano_head = GaussiansHead(cfg, mano_cfg)
        self.cfg = cfg

    def forward_step(self, batch: torch.ModuleDict, train: bool = False) -> torch.ModuleDict:
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]
        img_size = x.shape[-1]

        if img_size != 256:
            x = resize(x, 256)

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        if self.cfg.opt.get("freeze_vit", True):
            with torch.no_grad():
                conditioning_feats = self.backbone(x[:,:,:,32:-32])
        else:
            conditioning_feats = self.backbone(x[:,:,:,32:-32])
        
        output = self.mano_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output['pred_mano_params'] = {k: v.clone() for k,v in output["pred_mano_params"].items()}
        # Mano params are pose, global orient and betas. NO TRANSLATION
        mano_output = self.mano(**{k: v.float() for k,v in output["pred_mano_params"].items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        return output


class GaussianHaMeRPredictor(nn.Module):
    """ 
    THIS IS WHERE THE POINTS GET MOVE INTO 3D.
    MAIN MODEL WRAPPER. THIS IS CALLED IN INFERENCE/TRAINING
    """
    def __init__(self, cfg, mano_cfg):
        super().__init__()
        self.cfg = cfg
        self.network = GaussianHaMeR(cfg, mano_cfg)
        self.openpose_vertices = [1, 0, 8, 5, 6, 7, 12, 13, 14, 2, 3, 4, 9, 10, 11, 16, 18, 15, 17]
    
    def forward(self, x, 
                source_cameras_view_to_world=None, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                pps_pixels=None):
        
        B = x.shape[0]
        N_views = x.shape[1]

        # includes pred_mano_params as dict
        predictions = self.network({"img": x.squeeze(1)})

        focals = focals_pixels.clone()[:, 0, 0]

        vertices = predictions["pred_vertices"].clone() # These key points are located correctly in 3D space (as per MANO output)
        out_cam = predictions["pred_cam"].clone()
        keypoints_3d = predictions["pred_keypoints_3d"].clone()
        
        # DIFFERENT [12.6655, -0.0399,  0.0318]
        #print("outcam", out_cam)
        pred_cam_t = torch.stack(
            [
                out_cam[:, 1], # x axis
                out_cam[:, 2], # y axis
                2*focals/(self.cfg.data.training_resolution * out_cam[:, 0] + 1e-9) # depth / z axis
            ],dim=-1)
        # DIFFERENT: [-0.0399,  0.0318,  0.6916]
        #print(f"DEBUG || cam out {pred_cam_t}")
        vertices = vertices + pred_cam_t.unsqueeze(1) # shift to location infront of camera
        
        # adjust for principal points and normalise by focal length
        vertices[:, :, 0] = vertices[:, :, 0] - (vertices[:, :, 2, None] * (pps_pixels[:, 0, 0] / focals)[:, None, None])[:, :, 0] 
        vertices[:, :, 1] = vertices[:, :, 1] - (vertices[:, :, 2, None] * (pps_pixels[:, 0, 1] / focals)[:, None, None])[:, :, 0]
        keypoints_3d = keypoints_3d + pred_cam_t.unsqueeze(1)
        keypoints_3d[:, :, 0] = keypoints_3d[:, :, 0] - (keypoints_3d[:, :, 2, None] * (pps_pixels[:, 0, 0] / focals)[:, None, None])[:, :, 0]
        keypoints_3d[:, :, 1] = keypoints_3d[:, :, 1] - (keypoints_3d[:, :, 2, None] * (pps_pixels[:, 0, 1] / focals)[:, None, None])[:, :, 0]

        # transform into world coordinates.
        vertices = torch.cat((vertices, torch.ones_like(vertices[:, :, 0:1])), dim=-1)
        #vertices = torch.bmm(vertices, source_cameras_view_to_world.squeeze(1)) # Unnecessary
        # MANO verts are already in 3d space
        
        if source_cameras_view_to_world is not None:
            vertices = torch.bmm(vertices, source_cameras_view_to_world.squeeze(1))
        else:
            vertices = torch.bmm(vertices, torch.eye(4).unsqueeze(0).to(vertices.device))
            
        vertices = vertices[:, :, :3] 
        vertices = vertices.repeat_interleave(self.network.mano_head.gaussians_per_vertex, dim=1)
        #vertices = vertices - vertices.mean(1) + torch.tensor([[-0.5,0.5,0.5]]).to(vertices.device)
        
        keypoints_3d = torch.cat((keypoints_3d, torch.ones_like(keypoints_3d[:, :, 0:1])), dim=-1)
        # MANO verts are already in 3d space
        if source_cameras_view_to_world is not None:
            keypoints_3d = torch.bmm(keypoints_3d, source_cameras_view_to_world.squeeze(1))
        else:
            keypoints_3d = torch.bmm(keypoints_3d, torch.eye(4).unsqueeze(0).to(keypoints_3d.device))
        keypoints_3d = keypoints_3d[:, :, :3] 

        gaussians_output = {}
        source_cv2wT_quat = source_cv2wT_quat.reshape(B*N_views, *source_cv2wT_quat.shape[2:])
        gaussians_output["xyz_offsets"] = predictions["xyz"].clone()
        gaussians_output["xyz"] = vertices + predictions["xyz"]   #*0.error here
        gaussians_output["rotation"] = self.transform_rotations(predictions["rotation"], 
                                                                source_cv2wT_quat=source_cv2wT_quat)
        gaussians_output["opacity"] = predictions["opacity"]
        gaussians_output["scaling"] = predictions["scaling"]
        gaussians_output["features_dc"] = predictions["features_dc"]

        mano_out = {}
        mano_out["pred_vertices"] = predictions["pred_vertices"]
        mano_out["pred_cam"] = pred_cam_t
        mano_out["pred_keypoints_3d"] = predictions["pred_keypoints_3d"]
        mano_out["pred_mano_params"] = predictions["pred_mano_params"]
        mano_out["world_vertices"] = vertices
        mano_out["world_keypoints_3d"] = keypoints_3d
        return gaussians_output, mano_out
    
    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        Applies a transform that rotates the predicted rotations from 
        camera space to world space.
        Args:
            rotations: predicted in-camera rotation quaternions (B x N x 4)
            source_cameras_to_world: transformation quaternions from 
                camera-to-world matrices transposed(B x 4)
        Retures:
            rotations with appropriately applied transform to world space
        """
        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)

        rotations = quaternion_raw_multiply(Mq, rotations) 
        
        return rotations

    @staticmethod
    def transform_from_3d_points(points_from, points_to):
        points_from_b = torch.cat((points_from, torch.ones_like(points_from[:, :, 0:1])), dim=-1)
        points_to_b = torch.cat((points_to, torch.ones_like(points_to[:, :, 0:1])), dim=-1)
        transform = torch.linalg.lstsq(points_from_b, points_to_b.float()) 
        return transform.solution.transpose(1, 2)
    

def load_hamer_predictor(cfg, checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    print(f"model_cfg loaded from {model_cfg}")
    model_cfg = get_config(model_cfg, update_cachedir=True)
    
    assert cfg.data.training_resolution == model_cfg.MODEL.IMAGE_SIZE

    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    model = GaussianHaMeRPredictor(cfg, model_cfg)
    ckpts = torch.load(checkpoint_path)["state_dict"]
    model.network.load_state_dict(ckpts, strict=False)
    return model
    
    