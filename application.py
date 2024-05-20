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
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import math

from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector

MISSING_VALUE = -1

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

colorsMap = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def trans_keypoins(keypoints, param, img_size):
    missing_keypoint_index = keypoints == -1
    
    # crop the white line in the original dataset
    keypoints[:,0] = (keypoints[:,0]-40)

    # resize the dataset
    img_h, img_w = img_size
    scale_w = 1.0/176.0 * img_w
    scale_h = 1.0/256.0 * img_h

    if 'scale_size' in param and param['scale_size'] is not None:
        print("helllo")
        new_h, new_w = param['scale_size']
        scale_w = scale_w / img_w * new_w
        scale_h = scale_h / img_h * new_h
    # if param  is not None:
    #     new_h, new_w = param['scale_size']
    #     scale_w = new_w / 176
    #     scale_h = new_h / 256
    # else:
    #     scale_w = 1
    #     scale_h = 1

    # if 'crop_param' in param and param['crop_param'] is not None:
    #     w, h, _, _ = param['crop_param']
    # else:
    w, h = 0, 0
    keypoints[:,0] = keypoints[:,0]*scale_w - w
    keypoints[:,1] = keypoints[:,1]*scale_h - h
    keypoints[missing_keypoint_index] = -1
    return keypoints

def get_label_tensor(path, img, param=None):
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
        print(canvas.shape)
        keypoint = np.loadtxt(path)
        keypoint = trans_keypoins(keypoint, param, img.shape[1:])
        stickwidth = 4
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colorsMap[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(limbSeq[i])-1, 0]
            X = keypoint[np.array(limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colorsMap[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

        from torchvision.utils import save_image
        save_image(pose, "./styleChangeImage/testPose.png")
        # print(type(pose))

        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = F.to_tensor(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        label_tensor = torch.cat((pose, tensors_dist), dim=0)
        if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
            y0, x0 = keypoint[14, 0:2]
            y1, x1 = keypoint[15, 0:2]
            face_center = torch.tensor([y0, x0, y1, x1]).float()
        else:
            face_center = torch.tensor([-1, -1, -1, -1]).float()               
        return label_tensor, face_center

def get_joints_tensor(joints, pose):
    tensors_dist = 0
    e = 1
    for i in range(len(joints)):
        im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
        tensor_dist = F.to_tensor(Image.fromarray(im_dist))
        tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
        e += 1

    label_tensor = torch.cat((pose, tensors_dist), dim=0)             
    return label_tensor

def get_random_params(size, scale_param):
    w, h = size
    scale = random.random() * scale_param

    new_w = int( w * (1.0+scale) )
    new_h = int( h * (1.0+scale) )
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))
    return {'crop_param': (x, y, w, h), 'scale_size':(new_h, new_w)} 

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

# val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = False)

ckpt_list = ["temp_new_exp_sd21_epoch=200_step=744000.ckpt"]
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

transformsT = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])

for ckpt in ckpt_list:
    eIdx = ckpt.find("epoch")
    epoch = ckpt[eIdx:eIdx+9]
    epoch = epoch.replace("=", "_")

    model = create_model('./models/idea4.yaml').cpu()
    model.load_state_dict(load_state_dict('./'+ dir + ckpt, location='cpu'))
    model = model.cuda(gpu) #
    # model.eval()
    ddim_sampler = DDIMSampler(model)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)
    index = 0

    ddim_steps = 50
    strength = 1.0
    guess_mode = False
    count = 0
    batch_size = args.batch_size
    apply_openpose = OpenposeDetector()

    with torch.no_grad():
        
        model_id = 8
        for i in range(16,17):
            #load style image
            image = './styleChangeImage/style/style'+ str(i) +'.png'
            src = Image.open(image)
            src = transformsT(src).unsqueeze(0).cuda()
            
            # load reference image
            ref_img = './styleChangeImage/reference/reference' + str(model_id) +'.png'
            ref = Image.open(ref_img)

            # openpose detect
            input_image = HWC3(np.asarray(ref))
            detected_map, _, joints = apply_openpose(resize_image(input_image, 256))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, 256)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
            path = './styleChangeImage/' + '/reference' + str(model_id) + '_output/'
            cv2.imwrite(path + "pose.png", detected_map)
            tensor = F.to_tensor(detected_map)
            tensor = get_joints_tensor(joints, tensor)


            ref = transformsT(ref).unsqueeze(0).cuda()
            encoder_posterior = model.encode_first_stage(ref)
            #z = self.get_first_stage_encoding(encoder_posterior).detach()# latent code
            z = model.get_first_stage_encoding(encoder_posterior)
            
            # pose load by npy
            # ref_pose = './styleChangeImage/reference_pose_' + str(model_id) + '.npy'

            # pose load by cord txt
            # img = torch.rand(3,256,256)
            # param = {'scale_size':(256,256)}
            # param = get_random_params((280,256), 0.05)
            # tensor, _ = get_label_tensor('./styleChangeImage/poseTxt/reference' + str(model_id) +'_pose.txt', img, param)
            # resize = torchvision.transforms.Resize((256, 256))
            # tensor = resize(tensor)

            ref_mask = './styleChangeImage/reference/reference_mask_' + str(model_id) + '.png'
            
            mask = transforms.ToTensor()(Image.open(ref_mask)).cuda()
            resize = torchvision.transforms.Resize((32, 32))
            latent_mask = resize(mask.clone()).unsqueeze(0)
            new_channel = latent_mask[:, 0:1, :, :]
            latent_mask = torch.cat((latent_mask, new_channel), dim=1)

            # pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()
            pose = tensor.unsqueeze(0).cuda()

            
            control =  pose
            c_style = src
            cond = {"c_concat": [control], "c_style" : [c_style]}
            uc_full = {"c_concat": [control],  "c_style" : [c_style]}
            shape = (4, DataConf.data.resolution // 8, DataConf.data.resolution // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                            shape, cond, verbose=False, eta=0.0,
                                                            unconditional_guidance_scale=9.0,
                                                            unconditional_conditioning=uc_full,
                                                            ref_mask = latent_mask,
                                                            ref = z)
            

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)

            # mask = mask.resize((256, 256))
            x_samples = x_samples*mask + ref*(1-mask)

            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(batch_size)]
            path = './styleChangeImage/' + '/reference' + str(model_id) + '_output/'
            if not os.path.exists(path):
                os.makedirs(path)
            index = 0
            for result in results:
                path = './styleChangeImage/' 
                path = path + '/reference' + str(model_id) + '_output/' + "pplstyle" + str(i) + ".png"
                Image.fromarray(result).save(path)
                index += 1
            
# import numpy as np
# print(np.load('./styleChangeImage/reference_pose_0.npy').shape)