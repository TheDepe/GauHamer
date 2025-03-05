#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import hydra
import os
import resource
resource.setrlimit(
    resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
)

import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from PIL import Image, ImageDraw, ImageFont

from omegaconf import DictConfig, OmegaConf

from scene.hamer_extension import load_hamer_predictor
from scene.dataset_factory import get_dataset

from utils.general_utils import safe_state, batch_rotmat_to_aa, batch_rodrigues, save_ply
from utils.loss_utils import l1_loss, l2_loss, dice_loss
from utils.camera_utils import get_extra_cameras_for_batch
import lpips as lpips_lib

#from eval import evaluate_dataset
from gaussian_renderer import render_predicted



class MyIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except OSError:
            return self.__next__()    
        except AttributeError:
            return self.__next__()
        except ValueError as e:
            return self.__next__()




def full_loss_fn():
    return

@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):

    vis_dir = os.getcwd()
    output_dir = os.path.join(vis_dir, "rendered_images")
    os.makedirs(output_dir, exist_ok=True)
    
    
    WARMUP_PHASE = 0
    
    
    dict_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    if os.path.isdir(os.path.join(vis_dir, "wandb")):
        run_name_path = glob.glob(os.path.join(vis_dir, "wandb", "latest-run", "run-*"))[0]
        print("Got run name path {}".format(run_name_path))
        run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
        print("Resuming run with id {}".format(run_id))
        wandb_run = wandb.init(project=cfg.wandb.project, resume=True,
                        id = run_id, config=dict_cfg)

    else:
        wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                        config=dict_cfg)

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = load_hamer_predictor(cfg)
    gaussian_predictor.to(device)
    
    l = []
    l.append({'params': gaussian_predictor.parameters(), 
        'lr': cfg.opt.base_lr})

    optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15, 
                            betas=cfg.opt.betas)

    

    if cfg.opt.step_lr_at != -1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=cfg.opt.step_lr_at,
                                                    gamma=0.1)

    # Resuming training
    if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
        print('Loading an existing model from ', os.path.join(vis_dir, "model_latest.pth"))
        checkpoint = torch.load(os.path.join(vis_dir, "model_latest.pth"),
                                map_location=device) 
        try:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                               strict=False)
            print("Warning, model mismatch - was this expected?")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        first_iter = checkpoint["iteration"]
        best_PSNR = checkpoint["best_PSNR"] 
        print('Loaded model')
        
    # Resuming from checkpoint
    elif cfg.opt.pretrained_ckpt is not None:
        pretrained_ckpt_dir = os.path.join(cfg.opt.pretrained_ckpt, "model_latest.pth")
        checkpoint = torch.load(pretrained_ckpt_dir,
                                map_location=device) 
        try:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                               strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # first_iter = checkpoint["iteration"]
        best_PSNR = checkpoint["best_PSNR"] 
        print('Loaded model from a pretrained checkpoint')

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    if cfg.opt.start_lpips_after == 0:
        lambda_lpips = cfg.opt.lambda_lpips
    else:
        lambda_lpips = 0.0
    lambda_l12 = 1.0 - lambda_lpips

    loss_fn_alpha = dice_loss

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    dataset = get_dataset(cfg, "train")
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=cfg.opt.get("num_workers_train", 8))

    val_dataset = get_dataset(cfg, "val")
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.opt.get("num_workers_val", 8),
                                persistent_workers=True,
                                pin_memory=True)

    test_dataset = get_dataset(cfg, "test")
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1,
                                 shuffle=True)
    gaussian_predictor.train()

    print("Beginning training")
    first_iter += 1
    best_PSNR = 0.0
    dataloader_iterator = MyIterator(iter(dataloader))
    
    # ===============================
    # Training Loop
    # ===============================
    for iteration in range(first_iter, cfg.opt.iterations + 1):             
        
        if iteration == cfg.opt.start_lpips_after:
            lambda_lpips = cfg.opt.lambda_lpips
            lambda_l12 = 1.0 - lambda_lpips

        try:
            data = next(dataloader_iterator)
           
        except ValueError as e:
            print(f"Iteration {iteration} failed: {e}. Skipping.")
            continue
         
        except StopIteration:
            dataloader_iterator = MyIterator(iter(dataloader))
            data = next(dataloader_iterator)
                
        # Move data to the specified device
        data = {k: v.to(device) for k, v in data.items()}

        # Extract relevant data from the dataset
        rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]
        focals_pixels_pred = data["focals_pixels"][:, :cfg.data.input_images, ...]
        pps_pixels_pred = data["pps_pixels"][:, :cfg.data.input_images, ...]
        input_images = data["gt_images"][:, :cfg.data.input_images, ...]

        # ALL CORRECT
        # print(f"DEBUG || focals {focals_pixels_pred[0,0]}")
        # print(f"DEBUG || rot_transform_quats {rot_transform_quats[0,0]}")
        # print(f"DEBUG || pps_pixels_pred {pps_pixels_pred[0,0]}")
        # print(f"DEBUG || world_view_transforms {data['world_view_transforms'][0,0]}")
        # print(f"DEBUG || full_proj_transforms {data['full_proj_transforms'][0,0]}")
        # print(f"DEBUG || camera_centers {data['camera_centers'][0,0]}")
        
        gaussian_splats, mano_params = gaussian_predictor(
            x = input_images,
            source_cameras_view_to_world = data["view_to_world_transforms"][:, :cfg.data.input_images, ...], # shape 1,1,4,4
            #source_cameras_view_to_world = None,
            source_cv2wT_quat = rot_transform_quats,
            focals_pixels = focals_pixels_pred,
            pps_pixels = pps_pixels_pred
            )
        # DIFFERENT .16
        #print(f"DEBUG || mano mean {gaussian_splats['xyz'].mean()}")
        # can be used as loss
        #print(f"DEBUG || GT GO: {data['global_orient']}, PRED GO: {batch_rotmat_to_aa(ctw3x3.T @ mano_params['pred_mano_params']['global_orient'].squeeze(0))}")
        # regularize very big gaussians
        if len(torch.where(gaussian_splats["scaling"] > .5)[0]) > 0:
            big_gaussian_reg_loss = torch.mean(
                gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] > .5)] * 0.1)
            print('Regularising {} big Gaussians on iteration {}'.format(
                len(torch.where(gaussian_splats["scaling"] > .5)[0]), iteration))
        else:
            big_gaussian_reg_loss = 0.0
        # regularize very small Gaussians
        if len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]) > 0:
            small_gaussian_reg_loss = torch.mean(
                -torch.log(gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] < 1e-5)]) * 0.1)
            print('Regularising {} small Gaussians on iteration {}'.format(
                len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]), iteration))
        else:
            small_gaussian_reg_loss = 0.0
        # Render
        
        l12_loss_sum = 0.0
        lpips_loss_sum = 0.0
        rendered_images_train = []
        rendered_alphas_train = []
        gt_images_train = []
        gt_masks_train = []
        cam_ids_train = []
        
        
        
        
        for b_idx in range(data["gt_images"].shape[0]):
            # image at index 0 is training, remaining images are targets
            # Rendering is done sequentially because gaussian rasterization code
            # does not support batching
            gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_splats.items()}
            for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):
                if "focals_pixels" in data.keys():
                    focals_pixels_render = data["focals_pixels"][b_idx, r_idx].cpu()
                else:
                    focals_pixels_render = None
                if "background_color" in data.keys():
                        background_color = data["background_color"][b_idx, r_idx]
                else:
                    background_color = background
                render_out = render_predicted(gaussian_splat_batch, 
                                    data["world_view_transforms"][b_idx, r_idx],
                                    data["full_proj_transforms"][b_idx, r_idx],
                                    data["camera_centers"][b_idx, r_idx],
                                    background_color,
                                    cfg,
                                    focals_pixels=focals_pixels_render)
                image = render_out["render"]
                alpha = render_out["alpha"]
                # Put in a list for a later loss computation
                rendered_images_train.append(image)
                rendered_alphas_train.append(alpha)
                gt_image = data["gt_images"][b_idx, r_idx]
                cam_id = data['cam_ids'][b_idx, r_idx]
                cam_ids_train.append(cam_id)
                gt_images_train.append(gt_image)
                gt_mask = data["gt_masks"][b_idx, r_idx]
                gt_masks_train.append(gt_mask)
        rendered_images_train = torch.stack(rendered_images_train, dim=0)
        rendered_alphas_train = torch.stack(rendered_alphas_train, dim=0)
        gt_images_train = torch.stack(gt_images_train, dim=0)
        gt_masks_train = torch.stack(gt_masks_train, dim=0)
        
        
        #save_ply(gaussians=gaussian_splat_batch, path= os.path.join(vis_dir, f"pcd_{iteration}.ply"))
        #print("Saved ply to ", os.path.join(vis_dir, f"pcd_{iteration}.ply"))
        
        # ===============================
        # Loss calculation
        # ===============================
            
        # Add location loss
        actual_location = gaussian_splats['xyz'].mean(1)
        target_location = torch.tensor([[-0.5,0.5,0.5]]).to(actual_location.device)
        location_loss = l2_loss(target_location, actual_location)
        warmup_lambda = 0.5 if iteration < (WARMUP_PHASE) else 0.05
        total_loss = location_loss * warmup_lambda     

        if cfg.opt.get("offset_penalty", True):
                offsets = gaussian_splats["xyz_offsets"]
                norms = torch.linalg.norm(offsets, dim=-1)
                offset_penalty = torch.mean(norms)
                offset_lambda = 0.2 if iteration < WARMUP_PHASE else cfg.opt.get("offsets_coeff", 0.1)
                total_loss += offset_penalty * offset_lambda
            
        
        # Add remaining loss functions
        if iteration > WARMUP_PHASE:
            l12_loss_sum = loss_fn(rendered_images_train, gt_images_train) 
        
            if cfg.opt.lambda_lpips != 0 and iteration > cfg.opt.start_lpips_after:
                lpips_loss_sum = torch.mean(
                    lpips_fn(rendered_images_train * 2 - 1, gt_images_train * 2 - 1),
                    )
        
            total_loss += l12_loss_sum * lambda_l12
            total_loss += lpips_loss_sum * lambda_lpips
        
            if cfg.opt.alpha_loss and iteration:
                alpha_loss = loss_fn_alpha(rendered_alphas_train.squeeze(), gt_masks_train.squeeze())
                total_loss += cfg.opt.alpha_loss_coefficient * alpha_loss 
            
            total_loss += big_gaussian_reg_loss + small_gaussian_reg_loss

        total_loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if cfg.opt.step_lr_at != -1:
            scheduler.step()

        gaussian_predictor.eval()
        
        
        # ========= Save Images ==========
        
        #if iteration % 50 == 0 and iteration < 500 or iteration % 500 == 0:
        if iteration % cfg.logging.render_log == 0 or iteration == 1:
            print("Iteration: ", iteration)
            combined_images = []
            rendered = []
            gts = []
            
            # Loop through each image and camera ID to add the ID to the image
            for i in range(rendered_images_train.size(0)):
                cam_id = torch.stack(cam_ids_train, dim=0)[i].item()  # Assuming cam_ids_train is a tensor of size [N_images]
                                
                # Add camera ID to the image
                rendered_image_with_id = add_cam_id_to_image(rendered_images_train[i], cam_id)
                gt_image_with_id = add_cam_id_to_image(gt_images_train[i], cam_id)

                if rendered_image_with_id.shape[0] != 3:
                    rendered_image_with_id = rendered_image_with_id.repeat(3, 1, 1)
                
                if gt_image_with_id.shape[0] != 3:
                    gt_image_with_id = gt_image_with_id.repeat(3, 1, 1)

                rendered.append(rendered_image_with_id)
                gts.append(gt_image_with_id)

            separator = torch.zeros((3, 256, 25))  # Create black separator (3 channels for RGB)
            
            if len(rendered) > 1:
                rendered[1] = torch.cat([separator, rendered[1]], dim=2)
                gts[1] = torch.cat([separator, gts[1]], dim=2)
            
            rendered = torch.cat(rendered, dim=2)  
            gts = torch.cat(gts, dim=2)

            combined_images = torch.cat([rendered, gts], dim=1)

            print(f"Saved Image on iteration {iteration}.")

            # Save the combined image grid
            save_path = os.path.join(output_dir, f"rendering_iteration_{iteration}.png")
            save_image(
                combined_images, 
                save_path, 
                nrow=rendered_images_train.size(0),  # Arrange each row as per the number of rendered images
                normalize=False
            )
        

        # ========= Logging =============
        with torch.no_grad():
            if iteration % cfg.logging.loss_log == 0:
                wandb.log({"training_loss": total_loss.item()}, step=iteration)
                wandb.log({"location_loss": location_loss.item()}, step=iteration)
                wandb.log({"offset_loss": offset_penalty.item()}, step=iteration)
                if iteration > WARMUP_PHASE:
                    if cfg.opt.lambda_lpips != 0:
                        wandb.log({"training_l12_loss": np.log10(l12_loss_sum.item() + 1e-8)}, step=iteration)
                        if iteration > cfg.opt.start_lpips_after:
                            wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum.item() + 1e-8)}, step=iteration)
                        else:
                            wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum + 1e-8)}, step=iteration)
                
                if type(big_gaussian_reg_loss) == float:
                    brl_for_log = big_gaussian_reg_loss
                else:
                    brl_for_log = big_gaussian_reg_loss.item()
                if type(small_gaussian_reg_loss) == float:
                    srl_for_log = small_gaussian_reg_loss
                else:
                    srl_for_log = small_gaussian_reg_loss.item()
                wandb.log({"reg_loss_big": np.log10(brl_for_log + 1e-8)}, step=iteration)
                wandb.log({"reg_loss_small": np.log10(srl_for_log + 1e-8)}, step=iteration)

            if iteration % cfg.logging.render_log == 0 or iteration == 1:
                render_all = np.asarray([(np.clip(im.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8) for im in rendered_images_train])
                gt_all = np.asarray([(np.clip(im.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8) for im in gt_images_train])
                wandb.log({"render_all": wandb.Video(render_all, fps=1, format="mp4")}, step=iteration)
                wandb.log({"gt_all": wandb.Video(gt_all, fps=1, format="mp4")}, step=iteration)
                render_mask = np.asarray([(np.clip(im.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8) for im in rendered_alphas_train.repeat(1, 3, 1, 1)])
                gt_alpha = np.asarray([(np.clip(im.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8) for im in gt_masks_train.unsqueeze(1).repeat(1, 3, 1, 1)])
                wandb.log({"render_alpha": wandb.Video(render_mask, fps=1, format="mp4")}, step=iteration)
                wandb.log({"gt_alpha": wandb.Video(gt_alpha, fps=1, format="mp4")}, step=iteration)
            


        fnames_to_save = []
        # Find out which models to save
        if (iteration + 1) % cfg.logging.ckpt_iterations == 0:
            fnames_to_save.append(f"model_latest_it{iteration}.pth")
        
        # ============ Model saving =================
        for fname_to_save in fnames_to_save:
            ckpt_save_dict = {
                            "iteration": iteration,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": total_loss.item(),
                            "best_PSNR": best_PSNR
                            }
            ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict() 
            torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))

        gaussian_predictor.train()

    wandb_run.finish()


def add_cam_id_to_image(image_tensor, cam_id, text_color=(0, 0, 0), font_size=20):
    """
    Adds camera ID text to the top-left corner of an image.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (3, H, W) with values in [0, 1].
        cam_id (int): Camera ID to add to the image.
        text_color (tuple): RGB color for the text (default: white).
        font_size (int): Font size for the text.
    
    Returns:
        torch.Tensor: Image tensor with the camera ID text added.
    """
    # Convert the tensor to a PIL image (convert to numpy, scale to 0-255, and convert to uint8)
    image_np = (image_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    # Draw text on the image
    draw = ImageDraw.Draw(image_pil)
    try:
        # Load a default font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback font if the default isn't available
        font = ImageFont.load_default()
    
    text = f"Cam ID: {cam_id}"
    draw.text((10, 10), text, fill=text_color, font=font)

    # Convert the image back to a tensor
    image_with_text = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
    return image_with_text

def training_report(gaussian_splats, data):
    raise NotImplementedError
    torch.cuda.empty_cache()

if __name__ == "__main__":
        main()