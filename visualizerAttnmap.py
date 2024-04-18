
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
# from cldm.ddim_hacked import DDIMSampler
import data as deepfashion_data
from dataConfig.dataconfig import Config as DataConfig
import argparse
import debugpy
import os

import torchvision
from PIL import Image
from attnVisualizer.visualizer import get_local

"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# global ddpm_num_timesteps
global model

global ddim_alphas
global ddim_alphas_prev
global ddim_sigmas 
global attn_map_cache
attn_map_cache = []


def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    

def visualize_grid_to_grid(att_map, grid_index, target, source, filepath, index, grid_size=32, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    H,W = att_map.shape
    with_cls_token = False
      
    grid_target = highlight_grid(target, [grid_index], grid_size)
    grid_source = highlight_grid(source, [grid_index], grid_size, False)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((target.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_target)
    ax[0].axis('off')
    
    ax[1].imshow(grid_source)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')

    save_path =  filepath + '/attnmap_'+str(index)+'.png'

    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # 關閉圖表，以防止它在後台顯示
    
def highlight_grid(image, grid_indexes, grid_size=14, bool = True):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    if bool:
        for grid_index in grid_indexes:
            x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
            a= ImageDraw.ImageDraw(image)
            a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image


def init(modelFromckpt, schedule="linear", **kwargs):
    global model
    model = modelFromckpt
    global ddpm_num_timesteps
    ddpm_num_timesteps = model.num_timesteps
    schedule = schedule

def register_buffer(name, attr):
    if type(attr) == torch.Tensor:
        if attr.device != torch.device("cuda"):
            attr = attr.to(torch.device("cuda"))
    setattr(name, attr)

def make_schedule(ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
    global ddim_timesteps
    ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                num_ddpm_timesteps=ddpm_num_timesteps,verbose=verbose)
    # alphas_cumprod = model.alphas_cumprod

    # global ddpm_num_timesteps

    assert model.alphas_cumprod.shape[0] == ddpm_num_timesteps, 'alphas have to be defined for each timestep'
    to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)

    # global model

    # register_buffer('betas', to_torch(model.betas))
    global betas
    betas = to_torch(model.betas)
    # register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    global alphas_cumprod
    alphas_cumprod = to_torch(model.alphas_cumprod)
    # register_buffer('alphas_cumprod_prev', to_torch(model.alphas_cumprod_prev))
    global alphas_cumprod_prev
    alphas_cumprod_prev = to_torch(model.alphas_cumprod_prev)
    # calculations for diffusion q(x_t | x_{t-1}) and others
    # register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
    global sqrt_alphas_cumprod
    sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod.cpu()))
    # register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
    global sqrt_one_minus_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod.cpu()))
    # register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
    global log_one_minus_alphas_cumprod
    log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod.cpu()))
    # register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
    global sqrt_recip_alphas_cumprod
    sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod.cpu()))
    # register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))
    global  sqrt_recipm1_alphas_cumprod
    sqrt_recipm1_alphas_cumprod =  to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1))

    global ddim_alphas
    global ddim_sigmas
    global ddim_alphas_prev
    # ddim sampling parameters
    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                ddim_timesteps=ddim_timesteps,
                                                                                eta=ddim_eta,verbose=verbose)
    # register_buffer('ddim_sigmas', ddim_sigmas)
    # register_buffer('ddim_alphas', ddim_alphas)
    # register_buffer('ddim_alphas_prev', ddim_alphas_prev)
    global ddim_sqrt_one_minus_alphas
    ddim_sqrt_one_minus_alphas =  np.sqrt(1. - ddim_alphas)
    # register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
    sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
        (1 - model.alphas_cumprod_prev) / (1 - alphas_cumprod) * (
                    1 - alphas_cumprod / model.alphas_cumprod_prev))
    # register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    global ddim_sigmas_for_original_num_steps
    ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps

@torch.no_grad()
def sample( S,
            batch_size,
            shape,
            conditioning=None,
            callback=None,
            normals_sequence=None,
            img_callback=None,
            quantize_x0=False,
            eta=0.,
            mask=None,
            x0=None,
            temperature=1.,
            noise_dropout=0.,
            score_corrector=None,
            corrector_kwargs=None,
            verbose=True,
            x_T=None,
            log_every_t=100,
            unconditional_guidance_scale=1.,
            unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
            dynamic_threshold=None,
            ucg_schedule=None,
            **kwargs
            ):
    if conditioning is not None:
        if isinstance(conditioning, dict):
            ctmp = conditioning[list(conditioning.keys())[0]]
            while isinstance(ctmp, list): ctmp = ctmp[0]
            cbs = ctmp.shape[0]
            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        elif isinstance(conditioning, list):
            for ctmp in conditioning:
                if ctmp.shape[0] != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        else:
            if conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

    make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
    # sampling
    C, H, W = shape
    size = (batch_size, C, H, W)
    print(f'Data shape for DDIM sampling is {size}, eta {eta}')

    samples, intermediates = ddim_sampling(conditioning, size,
                                                callback=callback,
                                                img_callback=img_callback,
                                                quantize_denoised=quantize_x0,
                                                mask=mask, x0=x0,
                                                ddim_use_original_steps=False,
                                                noise_dropout=noise_dropout,
                                                temperature=temperature,
                                                score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                x_T=x_T,
                                                log_every_t=log_every_t,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning,
                                                dynamic_threshold=dynamic_threshold,
                                                ucg_schedule=ucg_schedule
                                                )
    return samples, intermediates

@torch.no_grad()
def ddim_sampling(cond, shape,
                    x_T=None, ddim_use_original_steps=False,
                    callback=None, timesteps=None, quantize_denoised=False,
                    mask=None, x0=None, img_callback=None, log_every_t=100,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                    ucg_schedule=None):
    device = model.betas.device
    b = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=device)
    else:
        img = x_T


    # global ddim_timesteps
    # global ddpm_num_timesteps

    if timesteps is None:
        timesteps = ddpm_num_timesteps if ddim_use_original_steps else ddim_timesteps
    elif timesteps is not None and not ddim_use_original_steps:
        subset_end = int(min(timesteps / ddim_timesteps.shape[0], 1) * ddim_timesteps.shape[0]) - 1
        timesteps = ddim_timesteps[:subset_end]

    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
    total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    print(f"Running DDIM Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)

        if mask is not None:
            assert x0 is not None
            img_orig = model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            img = img_orig * mask + (1. - mask) * img

        if ucg_schedule is not None:
            assert len(ucg_schedule) == len(time_range)
            unconditional_guidance_scale = ucg_schedule[i]

        outs = p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    dynamic_threshold=dynamic_threshold)
        img, pred_x0 = outs
        if callback: callback(i)
        if img_callback: img_callback(pred_x0, i)

        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

    return img, intermediates

@torch.no_grad()
def p_sample_ddim(x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = model.apply_model(x, t, c)
    else:
        model_t = model.apply_model(x, t, c)
        model_uncond = model.apply_model(x, t, unconditional_conditioning)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    
    attn_map_cache.extend(get_local.cache['CrossAttention.forward'][0:46])
    get_local.clear()  

    if model.parameterization == "v":
        e_t = model.predict_eps_from_z_and_v(x, t, model_output)
    else:
        e_t = model_output

    if score_corrector is not None:
        assert model.parameterization == "eps", 'not implemented'
        e_t = score_corrector.modify_score(model, e_t, x, t, c, **corrector_kwargs)

    alphas = model.alphas_cumprod if use_original_steps else ddim_alphas
    alphas_prev = model.alphas_cumprod_prev if use_original_steps else ddim_alphas_prev
    sqrt_one_minus_alphas = model.sqrt_one_minus_alphas_cumprod if use_original_steps else ddim_sqrt_one_minus_alphas
    sigmas = model.ddim_sigmas_for_original_num_steps if use_original_steps else ddim_sigmas 
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    if model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    else:
        pred_x0 = model.predict_start_from_z_and_v(x, t, model_output)

    if quantize_denoised:
        pred_x0, _, *_ = model.first_stage_model.quantize(pred_x0)

    if dynamic_threshold is not None:
        raise NotImplementedError()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    if noise_dropout > 0.:
        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev, pred_x0


get_local.activate() 

debugpy.listen(("0.0.0.0", 7979))
print("Waiting for client to attach...")
debugpy.wait_for_client()


parser = argparse.ArgumentParser(description='help')
parser.add_argument('--dataset_path', type=str, default='/workspace/dataset/dataset/deepfashion')
parser.add_argument('--DataConfigPath', type=str, default='./dataConfig/data.yaml')
parser.add_argument('--batch_size', type=int, default=25)

args = parser.parse_args()

DataConf = DataConfig(args.DataConfigPath)
DataConf.data.path = args.dataset_path

# DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> s
batch_size = args.batch_size
DataConf.data.val.batch_size = batch_size

val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)

ckpt_list = ["new_exp_sd21_epoch=200_step=744000.ckpt"]
dir = 'checkpoint_for_idea4_all_attnFliter_only_Attn/'
path = "/workspace/ControlNet_idea1_2/" + dir

# dir_list = os.listdir(path)
# print("Files and directories in '", path, "' :")
# # prints all files
# print(dir_list)
gpu = 1
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
torch.cuda.set_device(gpu)



for ckpt in ckpt_list:
    eIdx = ckpt.find("epoch")
    epoch = ckpt[eIdx:eIdx+9]
    epoch = epoch.replace("=", "_")

    model = create_model('./models/idea4.yaml').cpu()
    model.load_state_dict(load_state_dict('./'+ dir + ckpt, location='cpu'))
    model = model.cuda(gpu) #
    # model.eval()

    init(model)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)
    index = 0

    ddim_steps = 50
    strength = 1.0
    guess_mode = False
    count = 0
    batch_size = args.batch_size
    modelId = 0

    for x in val_dataset:
        count += batch_size
        if count > len(val_dataset.dataset):
            batch_size = batch_size - (count - len(val_dataset.dataset))

        get_local.clear()
        attn_map_cache = []
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
            samples, intermediates = sample(ddim_steps, batch_size,
                                                            shape, cond, verbose=False, eta=0.0,
                                                            unconditional_guidance_scale=9.0,
                                                            unconditional_conditioning=uc_full)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(batch_size)]
            path = './inferenceValDataset_idea4_all_attnFliter_only_Attn_' + epoch 
            if not os.path.exists(path):
                os.makedirs(path)
            index = 0
            for result in results:
                path = './inferenceValDataset_idea4_all_attnFliter_only_Attn_' + epoch 
                path = path + '/' + x["path"][index]
                Image.fromarray(result).save(path)
                index += 1

        cache = get_local.cache
        print(list(cache.keys()))

        source, target = x['path'][0].split('_2_')
        source += '.png'
        target = target.split('_vis')[0] +'.png'
        datasetDir = '/workspace/dataset/dataset/deepfashion/real_testDataset/test_256x256/'
        filePath = './AttnMapImage/grid/model_' + str(modelId) + '/'

        if not os.path.exists(filePath):
            os.makedirs(filePath)

        source = Image.open(datasetDir + source)
        target = Image.open(datasetDir + target)

        
        
        attn_map = attn_map_cache
        attn_map = [np.expand_dims(item, axis=0) for item in attn_map]
        # attn_map = attn_map.unsqueeze(0)
        grid = 340
        iterate = 50
        attn_layer = 46 * iterate - 1
        index = 0
        for i in range(8):
            visualize_grid_to_grid(attn_map[attn_layer][0,i,:,:], grid, target, source, filePath, index)
            index += 1

        modelId += 1