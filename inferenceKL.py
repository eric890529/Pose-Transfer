from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import data as deepfashion_data

from dataConfig.dataconfig import Config as DataConfig
import debugpy
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import data as deepfashion_data

from PIL import Image  
import PIL  
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
import numpy as np
import torchvision
import torchvision.transforms as transforms

# debugpy.listen(("0.0.0.0", 7777))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

import argparse
from io import BytesIO

def process(grid):
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid

import torchvision.transforms as transforms
def denorm(grid):
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    denorm = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                    std=[1.0 / s for s in std])
    grid = denorm(grid)
    return grid
parser = argparse.ArgumentParser(description='help')
parser.add_argument('--dataset_path', type=str, default='/workspace/dataset/dataset/deepfashion')
parser.add_argument('--DataConfigPath', type=str, default='./dataConfig/data.yaml')
parser.add_argument('--batch_size', type=int, default=25)

args = parser.parse_args()

DataConf = DataConfig(args.DataConfigPath)
DataConf.data.path = args.dataset_path

batch_size = 1
DataConf.data.val.batch_size = batch_size

model = create_model('./models/test.yaml').cpu()

resume_path =  './VaeModel/model.ckpt'
# resume_path = './models/control_sd15_ini.ckpt'

gpu = 1
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
torch.cuda.set_device(gpu)

model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.cuda(gpu)

val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)
index=0
import torch
# x = image()
model.eval()

for image in val_dataset:
    with torch.no_grad():
        path = './VAE_image/'
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        image = image['source_image'].to(device="cuda")
        encoder_posterior = model.encode(image)
        # latent = model.get_first_stage_encoding(encoder_posterior).detach()# latent code
        latent = encoder_posterior.sample()

        print(latent.shape)
        
        latent = 0.18215 * latent
        latent = 1. / 0.18215 * latent
        rec  = model.decode(latent)
        
        rec = rec.detach().cpu()
        grid = torchvision.utils.make_grid(rec, nrow=8)
        grid = torch.clamp(grid, -1., 1.) #去除雜訊
        grid = (grid + 1.0) / 2.0
        # grid = denorm(grid)
# 
        grid = process(grid)
        path = path + '/' + str(index) + ".png"

        # print(grid.shape)
        # import cv2
        # grid = cv2.resize(grid, dsize=(176, 256), interpolation=cv2.INTER_CUBIC)
        # Image.fromarray(grid).save(path)
        print(index)
        index += 1




# logger = ImageLogger(batch_frequency=2)

# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])#  accumulate_grad_batches=4 , resume_from_checkpoint = './checkpoint/last.ckpt' , resume_from_checkpoint = './checkpoint_for_diffusion/last.ckpt'

# predictions = trainer.predict(model, val_dataset)
