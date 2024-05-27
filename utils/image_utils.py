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

import torch
import numpy as np
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_from_saved_images(gen_path, gt_path, bg_color):
    gen_img = np.array(Image.open(gen_path).convert("RGB")).astype(np.float32) / 255.0
    h, w = gen_img.shape[:2]
    gt_img = np.array(Image.open(gt_path).convert("RGBA").resize((h, w), Image.LANCZOS)).astype(np.float32) / 255.0
    gt_img = gt_img[:, :, :3] * gt_img[:, :, 3:4] + (1.0 - gt_img[:, :, 3:4]) * bg_color.detach().cpu().numpy().astype(np.float32)
    psnr = 20 * np.log10(1.0 / np.sqrt(np.mean((gen_img - gt_img) ** 2)))
    return psnr
