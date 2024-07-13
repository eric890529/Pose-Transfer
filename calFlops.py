# Transformers Model, such as bert.
# from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer
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
# debugpy.listen(("0.0.0.0", 7689))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()
# Configs

resume_path = './models/idea4.ckpt'
# resume_path = './checkpoint_for_idea4_all_attnFliter_only_Attn/new_exp_sd21_epoch=103_step=384000.ckpt'
#batch_size = 2
logger_freq = 4000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

model = create_model('./models/idea4.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

batch_size = 10
# model = AutoModel.from_pretrained(model_save)
import torch
temp = torch.randn(10, 3, 256, 256)

model = model.cuda()

# model.eval()

# from torchvision.models import resnet50
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table


input = torch.randn(1, 4, 32, 32).cuda()
temp1 = torch.randn(1, 20, 256, 256).cuda()
temp2 = torch.randn(1, 3, 256, 256).cuda()

c = {"c_concat" : [temp1], "c_style" : [temp2]}
tensor = (input, c,)
# flops = FlopCountAnalysis(model, tensor)
# params = parameter_count_table(model)
flops, params = profile(model, inputs=(input,c))
print(f"FLOPs: {flops}")
print(f"Params: {params}")

