from typing import Dict, Sequence
import math
import random
import time
import glob
import os
import cv2

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.image import augment, random_crop_arr, center_crop_arr
from utils.file import load_file_list


class PROJECTDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(self, 
                file_path: list,
                out_size: int,
                scale: int,
                hr_pattern: str,
                blur_sigma: float,
                blur_kernel: int,
                crop_type: str,
                use_hflip: bool,
                use_rot: bool,
                ):
        super().__init__()
        
        self.patch_size = out_size // 4
        self.scale = scale
        
        lq_files = []
        hr_files = []
        for path in file_path:
            lq_files += glob.glob(os.path.join(path, "*phone.*"))
            hr_files += glob.glob(os.path.join(path, "*screen.*"))

        self.lq_files = lq_files
        self.hr_files = hr_files
        self.lq_files.sort()
        self.hr_files.sort()

        self.hr_pattern = hr_pattern
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        self.noise_scale1 = 0.002
        self.noise_scale2 = 0.003
        self.blur_kernal_list = [ i for i in range(1, self.blur_kernel * 2, 2)]
        self.blur_sigma_list = [ i / 10 for i in range(1, int(2 * self.blur_sigma * 10), 1)]

        self.edge_pixel = 0

    @torch.no_grad()
    def __len__(self):
        return len(self.lq_files)

    def __getitem__(self, index):
        lowRes_file = self.lq_files[index]
        highRes_file = self.hr_files[index]

        hightRes_img = Image.open(highRes_file).convert("RGB")
        hightRes_img = np.array(hightRes_img)
        hightRes_img = (hightRes_img[..., ::-1] / 255.0).astype(np.float32)
        lowRes_img = Image.open(lowRes_file).convert("RGB")
        lowRes_img = np.array(lowRes_img)
        lowRes_img = (lowRes_img[..., ::-1] / 255.0).astype(np.float32)
        
        low_h = lowRes_img.shape[0]
        high_h = hightRes_img.shape[0]
        if low_h == high_h:
            lowRes_img = cv2.resize(lowRes_img, None, fx=1 / self.scale, fy=1 / self.scale)

        # scale为4数据集用作2x训练
        if high_h // low_h == 4 and self.scale == 2:
            hightRes_img = cv2.resize(hightRes_img, None, fx=1 / 2, fy=1 / 2)
        
        # scale为2数据集用作4x训练
        if "20231021_plant_lightroom_Crop320" in lowRes_file and self.scale == 4:
            lowRes_img = cv2.resize(lowRes_img, None, fx=1 / 2, fy=1 / 2)


        img_size_ori_h, img_size_ori_w = lowRes_img.shape[:2]
        i = random.randint(self.edge_pixel, img_size_ori_h - self.patch_size - self.edge_pixel)
        j = random.randint(self.edge_pixel, img_size_ori_w - self.patch_size - self.edge_pixel)
        hightRes_img = hightRes_img[i * self.scale : i * self.scale + self.patch_size * self.scale, j * self.scale : j * self.scale + self.patch_size * self.scale]
        lowRes_img = lowRes_img[i : i + self.patch_size, j : j + self.patch_size]

        if random.uniform(0, 1) < 0.5:
            hightRes_img = hightRes_img[::-1]
            lowRes_img = lowRes_img[::-1]
        if random.uniform(0, 1) < 0.5:
            hightRes_img = hightRes_img[:,::-1]
            lowRes_img = lowRes_img[:,::-1]
        if 1:
            sigma_x_offset = random.choice(self.blur_sigma_list)
            sigma_y_offset = random.choice(self.blur_sigma_list)
            kernal_x_offset = random.choice(self.blur_kernal_list)
            kernal_y_offset = random.choice(self.blur_kernal_list)
            lowRes_img = cv2.GaussianBlur(lowRes_img, (kernal_x_offset, kernal_y_offset), sigmaX=sigma_x_offset, sigmaY=sigma_y_offset)
            # lowRes_img = lowRes_img * np.random.normal(loc=1, scale=self.noise_scale1, size=lowRes_img.shape)  +  np.random.normal(loc=0, scale=self.noise_scale2, size=lowRes_img.shape)

        hightRes_img = np.ascontiguousarray(hightRes_img)
        lowRes_img = np.ascontiguousarray(lowRes_img)        

        # lowRes_img = lowRes_img[:,:,None]
        # hightRes_img = hightRes_img[:,:,None]
        lowRes_img = lowRes_img.clip(0, 1)
        hightRes_img = hightRes_img.clip(0, 1)

        lowRes_img = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(lowRes_img, axis=0), (2, 0, 1)))).float()
        hightRes_img = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(hightRes_img, axis=0), (2, 0, 1)))).float()

        return {'lq': lowRes_img, 'hq': hightRes_img, 'txt': '', 'file_name': lowRes_file}