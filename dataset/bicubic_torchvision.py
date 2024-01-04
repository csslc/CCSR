from typing import Dict, Sequence
import math
import random
import time

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.image import augment, random_crop_arr, center_crop_arr
from utils.file import load_file_list


class BicubicDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool
    ) -> "BicubicDataset":
        super(BicubicDataset, self).__init__()
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        # self.blur_kernel_size = blur_kernel_size
        # self.kernel_list = kernel_list
        # # a list for each kernel probability
        # self.kernel_prob = kernel_prob
        # self.blur_sigma = blur_sigma
        # # betag used in generalized Gaussian blur kernels
        # self.betag_range = betag_range
        # # betap used in plateau blur kernels
        # self.betap_range = betap_range
        # # the probability for sinc filters
        # self.sinc_prob = sinc_prob

        # self.blur_kernel_size2 = blur_kernel_size2
        # self.kernel_list2 = kernel_list2
        # self.kernel_prob2 = kernel_prob2
        # self.blur_sigma2 = blur_sigma2
        # self.betag_range2 = betag_range2
        # self.betap_range2 = betap_range2
        # self.sinc_prob2 = sinc_prob2
        
        # # a final sinc filter
        # self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 7 to 21
        # self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # # TODO: kernel range is now hard-coded, should be in the configure file
        # # convolving with pulse tensor brings no blurry effect
        # self.pulse_tensor = torch.zeros(21, 21).float()
        # self.pulse_tensor[10, 10] = 1

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #
        hq_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(hq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"
        
        if self.crop_type == "random":
            pil_img = random_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "center":
            pil_img = center_crop_arr(pil_img, self.out_size)
        # self.crop_type is "none"
        else:
            pil_img = np.array(pil_img)
            assert pil_img.shape[:2] == (self.out_size, self.out_size)
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (pil_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        
        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()

        return {
            "hq": img_hq,
            'txt': ""
        }

    def __len__(self) -> int:
        return len(self.paths)
