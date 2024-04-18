import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from cldm.model import create_model

import debugpy

# debugpy.listen(("0.0.0.0", 7777))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

# def get_node_name(name, parent_name):
#     if len(name) <= len(parent_name):
#         return False, ''
#     p = name[:len(parent_name)]
#     if p != parent_name:
#         return False, ''
#     return True, name[len(parent_name):]


# model = create_model(config_path='./models/ControlNetWirhDream.yaml')

# pretrained_weights = torch.load(input_path)
# if 'state_dict' in pretrained_weights:
#     pretrained_weights = pretrained_weights['state_dict']

# scratch_dict = model.state_dict()
# import re
# target_dict = {}
# for k in scratch_dict.keys():
#     is_control, name = get_node_name(k, 'control_')
#     is_style, name2 = get_node_name(k, 'style_encoder')
#     if is_style:
#         copy_k = 'model.diffusion_model' + name2
#     elif is_control :
#         copy_k = 'model.diffusion_' + name
#     else:
#         copy_k = k
#     if copy_k in pretrained_weights:
#         if (re.search('.*transformer_blocks.0.attn2.to_k.*', copy_k) or re.search('.*transformer_blocks.0.attn2.to_v.*', copy_k)) and is_control:
#         #if  re.search('.*transformer.*', copy_k) :
#             target_dict[k] = scratch_dict[k].clone()
#             print(f'These weights are newly added: {k}')
#         else:
#             if re.search('style_encoder.input_blocks.0.0.weight', k):
#                 target_dict[k] = scratch_dict[k].clone()
#                 print(f'These weights are newly added: {k}')
#             else:
#                 # if re.search('style_encoder.*', k):
#                 #     print("Style ",k)
#                 target_dict[k] = pretrained_weights[copy_k].clone()
#             #print(f'These weights are old added: {k}')
#     else:
#         target_dict[k] = scratch_dict[k].clone()
#         print(f'These weights are newly added: {k}')

# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
# print('Done.')

import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'
import re
import torch
from share import *
from cldm.model import create_model

# import debugpy

# debugpy.listen(("0.0.0.0", 7777))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/idea4.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

vae_pretrained_weights = torch.load('./VaeModel/vae-ft-mse-840000-ema-pruned.ckpt')
if 'state_dict' in vae_pretrained_weights:
    vae_pretrained_weights = vae_pretrained_weights['state_dict']
# print(vae_pretrained_weights.keys())
scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name #cldm的diffusion是這樣命名的
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        if re.search('.*first_stage_model.*', copy_k) :
            copy_k = copy_k[18:]
            target_dict[k] = vae_pretrained_weights[copy_k].clone()
            print(f'VAE weight: {k}')
        # elif re.search('model.diffusion_model.input_blocks.0.0.weight', copy_k):
        #     target_dict[k] = scratch_dict[k].clone()
        #     print(f'First Layer not Load: {k}')
        # elif re.search('model.diffusion_model.out.2.*', copy_k):
        #     target_dict[k] = scratch_dict[k].clone()
        #     print(f'last Layer not Load: {k}')
        else:
            target_dict[k] = pretrained_weights[copy_k].clone()
    else: 
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')
    # if re.search('clip_processor.*', k) :
    #     print("here ",k)


model.load_state_dict(target_dict, strict=True)

# model.first_stage_key.load_state_dict()
torch.save(model.state_dict(), output_path)
print('Done.')