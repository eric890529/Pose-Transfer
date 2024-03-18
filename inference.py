from share import *
import config

import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
# from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import data as deepfashion_data
from dataConfig.dataconfig import Config as DataConfig
import argparse
import debugpy
import os

import torchvision
from PIL import Image


import os
def log_local(save_dir, split, images, global_step, current_epoch, batch_idx):
    root = os.path.join(save_dir, "image_log", split)
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=4) #log_images N要調整
        if k != 'control':
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        if k == 'control':
            # #grid, pose, pose_dist = torch.split(grid, [3,3,17], dim = 0)
            # pose, pose_dist = torch.split(grid, [3,20], dim = 0)
            
            # ##如果controlnet有給原圖才需要
            # # import torchvision.transforms as transforms
            # # mean = (0.5, 0.5, 0.5)
            # # std = (0.5, 0.5, 0.5)
            # # m = -0.5/0.5
            # # s = 1.0/0.5
            # # denorm = transforms.Normalize((m, m, m),
            # #                             (s, s, s))
            # # #denorm = transforms.Normalize((-mean/std).tolist(), (1.0/std).tolist())
            
            
            # # grid = denorm(grid)
            
            # import torchvision.transforms as transforms
            # mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            # std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            # denorm = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
            #                                 std=[1.0 / s for s in std])
            # pose = denorm(pose)
            # # grid = pose
            # grids = [grid, pose]
            grid, pose, pose_dist = torch.split(grid, [3,3,17], dim = 0)
            
            import torchvision.transforms as transforms
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            denorm = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                            std=[1.0 / s for s in std])
            grid = denorm(grid)
            
            # import torchvision.transforms as transforms
            # mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            # std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            # denorm = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
            #                                 std=[1.0 / s for s in std])
            # pose = denorm(pose)
            
            grids = [grid, pose]

            for idx, grid in enumerate(grids):
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = os.path.join(root,"pose_" + filename)
                if idx == 0:
                    path = os.path.join(root, filename)
                else:
                    path = os.path.join(root,"pose_" + filename)

                os.makedirs(os.path.split(path)[0], exist_ok=True)
            
                Image.fromarray(grid).save(path)

            # grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            # grid = grid.numpy()
            # grid = (grid * 255).astype(np.uint8)
            # filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            # path = os.path.join(root, "pose_" + filename)
            # os.makedirs(os.path.split(path)[0], exist_ok=True)
            
            # Image.fromarray(grid).save(path)
        else:    
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            
            Image.fromarray(grid).save(path)


# debugpy.listen(("0.0.0.0", 7777))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()


parser = argparse.ArgumentParser(description='help')
parser.add_argument('--dataset_path', type=str, default='/workspace/dataset/dataset/deepfashion')
parser.add_argument('--DataConfigPath', type=str, default='./dataConfig/data.yaml')
parser.add_argument('--batch_size', type=int, default=25)

args = parser.parse_args()

DataConf = DataConfig(args.DataConfigPath)
DataConf.data.path = args.dataset_path

# DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> s
batch_size = 16
DataConf.data.val.batch_size = batch_size

val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)

model = create_model('./models/idea4.yaml').cpu()
model.load_state_dict(load_state_dict('./checkpoint_for_idea4_aff/new_exp_sd21_epoch=161_step=598500.ckpt', location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

if config.save_memory:
    model.low_vram_shift(is_diffusing=True)
index = 0
for x in val_dataset:
    # x = x.to("cuda")
    # for k in x:
    #     x[k] = x[k].to("cuda")
    with torch.no_grad():
        images = model.log_images(x, batch_size, is_inference = True)

    for k in images:
        N = min(images[k].shape[0], batch_size)
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1., 1.)

    name = "idea4_aff"
    log_local("", "train/inferenceLog_" + name, images,
                0, 0, index)
    index += 1