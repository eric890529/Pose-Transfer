from share import *

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

# debugpy.listen(("0.0.0.0", 7777))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()


import argparse
parser = argparse.ArgumentParser(description='help')
parser.add_argument('--dataset_path', type=str, default='/workspace/dataset/dataset/deepfashion')
parser.add_argument('--DataConfigPath', type=str, default='./dataConfig/data.yaml')
parser.add_argument('--batch_size', type=int, default=10)

args = parser.parse_args()

DataConf = DataConfig(args.DataConfigPath)
DataConf.data.path = args.dataset_path

DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src
    
val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)

# Configs
resume_path = './models/idea4_aff.ckpt'
resume_path = './checkpoint_for_idea4_aff/new_exp_sd21_epoch=91_step=337500.ckpt'
#batch_size = 2
logger_freq = 3000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/idea4_aff.yaml').cpu()


# import torch.nn as nn
# dims = 2
# model_channels = 320
# model.control_model.input_hint_block = TimestepEmbedSequential(
#             conv_nd(dims, 23, 16, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 16, 16, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 16, 32, 3, padding=1, stride=2),
#             nn.SiLU(),
#             conv_nd(dims, 32, 32, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 32, 96, 3, padding=1, stride=2),
#             nn.SiLU(),
#             conv_nd(dims, 96, 96, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, 96, 256, 3, padding=1, stride=2),
#             nn.SiLU(),
#             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         )

model.load_state_dict(load_state_dict(resume_path, location='cpu'))#可以在load dict+一個參數 硬load ## attention這樣好像不會更新到? , strict=False


#model.load_state_dict(load_state_dict(resume_path, location='cpu'))#可以在load dict+一個參數 硬load
# model.first_stage_model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)#可以在load dict+一個參數 硬load
# model.cond_stage_model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


from pytorch_lightning.callbacks import ModelCheckpoint

import os
directory = "checkpoint_for_idea4_aff"
if not os.path.exists(directory):
    os.makedirs(directory)
acc_size = 2
checkpoint_callback = ModelCheckpoint(dirpath = directory,
                                      save_top_k = -1,
                                      every_n_train_steps=9000/acc_size, save_last=True, #4000/1000
                                      save_weights_only=False,
                                      filename='new_exp_sd21_{epoch:02d}_{step:06d}')

# Misc
# dataset = MyDataset()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], accumulate_grad_batches=acc_size
                     , resume_from_checkpoint = resume_path)# , resume_from_checkpoint = './checkpoint/last.ckpt' , resume_from_checkpoint = './checkpoint_for_diffusion/last.ckpt'
#, resume_from_checkpoint = resume_path
# Train!
trainer.fit(model, train_dataset)

# , resume_from_checkpoint = resume_path
# 存checkpoint地點
# 改路徑logger
# losscurve
# model checkpoint
# 如果要單獨finetune 記得改opt 以及control參與計算那邊
