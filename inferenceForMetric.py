from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import data as deepfashion_data
from dataConfig.dataconfig import Config as DataConfig
import argparse
import debugpy
import os

import torchvision
from PIL import Image


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
batch_size = 32
DataConf.data.val.batch_size = batch_size

val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)

ckpt_list = ["new_exp_sd21_epoch=100_step=372000.ckpt"]

path = "/workspace/ControlNet_idea1_2/checkpoint_for_idea4_all_attnFliter"

# dir_list = os.listdir(path)
# print("Files and directories in '", path, "' :")
# prints all files
# print(dir_list)

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.cuda.set_device(1)
for ckpt in ckpt_list:
    eIdx = ckpt.find("epoch")
    epoch = ckpt[eIdx:eIdx+9]
    epoch = epoch.replace("=", "_")

    model = create_model('./models/idea4.yaml').cpu()
    model.load_state_dict(load_state_dict('./checkpoint_for_idea4_all_attnFliter/' + ckpt, location='cpu'))
    model = model.cuda(1)
    model.eval()
    ddim_sampler = DDIMSampler(model)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)
    index = 0

    ddim_steps = 50
    strength = 1.0
    guess_mode = False
    count = 0
    batch_size = 32

    for x in val_dataset:
        count += batch_size
        if count > len(val_dataset.dataset):
            batch_size = batch_size - (count - len(val_dataset.dataset))
        with torch.no_grad():
            z, c = model.get_input(x, "nothing", bs=batch_size, is_inference = True)
            control =  c["c_concat"][0][:batch_size]
            c_style = c["c_style"][0][:batch_size] #
            cond = {"c_concat": [control], "c_style" : [c_style]}
            uc_full = {"c_concat": [control],  "c_style" : [c_style]}
            shape = (4, DataConf.data.resolution // 8, DataConf.data.resolution // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                            shape, cond, verbose=False, eta=0.0,
                                                            unconditional_guidance_scale=9.0,
                                                            unconditional_conditioning=uc_full)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(batch_size)]
            path = './inferenceValDataset_idea4_all_attnFliter_' + epoch 
            if not os.path.exists(path):
                os.makedirs(path)
            index = 0
            for result in results:
                path = './inferenceValDataset_idea4_all_attnFliter_' + epoch 
                path = path + '/' + x["path"][index]
                Image.fromarray(result).save(path)
                index += 1
        