# evaluate the L-STD of restored N images


import cv2
import argparse, os, sys, glob
import logging
from datetime import datetime

from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """from BasicSR
    Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k in opt:
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-imgs",
		nargs="+",
		help="path to the input image",
		default=['****/sample0',
		'****/sample1',
		'****/sample2',
		'****/sample3',
		'****/sample4',
		'****/sample5',
		'****/sample6',
		'****/sample7',
		'****/sample8',
		'****/sample9',
		]
	)

	

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	name_log = 'ours'
	path_log = '****/L-STD'
	os.makedirs(path_log, exist_ok=True)

	# init logger
	setup_logger('base', path_log, 'test_' + name_log, level=logging.INFO, screen=True, tofile=True)
	logger = logging.getLogger('base')
	logger.info(name_log)

	num_samples = len(opt.init_imgs)

	# combine all the sampled data
	
	img_sr_dirs = []
	for dir_idx in range(num_samples):
		img_sr_dir = opt.init_imgs[dir_idx]
		img_sr_dirs.append(img_sr_dir)

	img_sr_list = sorted(glob.glob(os.path.join(img_sr_dir, '*.png')))
	num_imgs = len(img_sr_list)
	input_sr_img = cv2.imread(img_sr_list[0], cv2.IMREAD_COLOR)
	sr = img2tensor(input_sr_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

	sr = sr[..., 4:-4, 4:-4]/255. #[0,1]
	B, C, H, W = sr.shape
	sr = sr[0]
	
	std_alls = []
	
	for img_idx in range(len(img_sr_list)):
		d = torch.zeros([num_samples,C,H,W]) 
		for dir_idx in range(num_samples): 
			img_sr_dir = opt.init_imgs[dir_idx]
			img_sr_list = sorted(glob.glob(os.path.join(img_sr_dir, '*.png')))
			
			input_sr_img = cv2.imread(img_sr_list[img_idx], cv2.IMREAD_COLOR)
			img_sr_name = os.path.basename(img_sr_list[img_idx])

			sr = img2tensor(input_sr_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

			sr = sr[..., 4:-4, 4:-4]/255. #[0,1]
			sr = sr[0]
			d[dir_idx,...] = sr

		d = np.array(d)
		stds = np.std(d, axis=0)
		stds = np.mean(stds)
		logger.info('_{} - L-STD: {:.6f}'.format(img_sr_name, stds))
		std_alls.append(stds)
	std_all = np.mean(std_alls)
	print(std_all)
	logger.info('_all - L-STD: {:.6f}; '.format(std_all))

main()