# evaluate the restored images with IQA
# PSNR, SSIM, LPIPS are given as example, you can add more IQA in this file

import cv2
import argparse, os, sys, glob
import logging
from datetime import datetime

import pyiqa

from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch

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
		],
	)

	parser.add_argument(
		"--init-imgs-names",
		nargs="+",
		help="name of the input image",
		default=['****-0', '****-1', '****-2', '****-3', '****-4', '****-5', '****-6', '****-7', '****-8', '****-9'
		],
		)

	parser.add_argument(
		"--gt-imgs",
		nargs="+",
		help="path to the gt image, you need to add the paths of gt folders corresponding to init-imgs",
		default=['****', '****', '****', '****', '****','****',
		 '****', '****', '****', '****', '****','****',
		],
		
	)
	
	parser.add_argument(
		"--log",
		type=str,
		nargs="?",
		help="path to the log",
		default='/home/notebook/data/group/SunLingchen/code/CCSR/CCSR-main/experiments')

	parser.add_argument(
		"--log-name",
		type=str,
		nargs="?",
		help="name of your log",
		default='test',
	)

	parser.add_argument(
		"--num_img",
		type=int,
		nargs="?",
		help="the number of images evaluated in the folder; 0: all the images are evaludated.",
		default=0,
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	os.makedirs(opt.log, exist_ok=True)
	# init logger
	setup_logger('base', opt.log, 'test_' + opt.log_name, level=logging.INFO,
                  screen=True, tofile=True)
	logger = logging.getLogger('base')
	logger.info(opt)

	# init metrics: you can add more metrics here

	iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
	iqa_lpips = pyiqa.create_metric('lpips', device=device)
	


	for dir_idx in range(len(opt.init_imgs)):

		gt_dir = opt.gt_imgs[dir_idx]
		img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

		img_sr_dir = opt.init_imgs[dir_idx]
		img_sr_list = sorted(glob.glob(os.path.join(img_sr_dir, '*.png')))
		print(opt.init_imgs_names[dir_idx])
		print(f'GT IMAGES LEN IS {len(img_gt_list)}, SR IMAGES LEN IS {len(img_sr_list)}')

		assert len(img_gt_list) == len(img_sr_list)


	for dir_idx in range(len(opt.init_imgs)):

		# record metrics
		metrics = {}
		metrics['psnr'], metrics['ssim'], metrics['lpips'] = \
			 [], [], []

		gt_dir = opt.gt_imgs[dir_idx]
		img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

		img_sr_dir = opt.init_imgs[dir_idx]
		img_sr_list = sorted(glob.glob(os.path.join(img_sr_dir, '*.png')))

		if opt.num_img != 0:
			img_gt_list = img_gt_list[0:opt.num_img]
			img_sr_list = img_sr_list[0:opt.num_img]
			
		PSNR_all, SSIM_all, lpips_all = 0.0, 0.0, 0.0
		logger.info('\nTesting [{:s}]...'.format(opt.init_imgs_names[dir_idx]))

		for img_idx in range(len(img_sr_list)):
			img_sr_name = os.path.basename(img_sr_list[img_idx])
			print(f'Processing {img_sr_name} ...')
			print(img_sr_list[img_idx])
			
			input_sr_img = cv2.imread(img_sr_list[img_idx], cv2.IMREAD_COLOR)
			sr = img2tensor(input_sr_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()
				
			input_gt_img = cv2.imread(img_gt_list[img_idx], cv2.IMREAD_COLOR)
			hr = img2tensor(input_gt_img, bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

			if sr.shape != hr.shape:
				continue

			# PSNR: convert the ycbcr to calculate
			hr = hr[..., 4:-4, 4:-4]/255.
			sr = sr[..., 4:-4, 4:-4]/255.

			PSNR_now = iqa_psnr(sr, hr).item()
			PSNR_all += PSNR_now
			metrics['psnr'].append(PSNR_now)

			# SSIM
			ssim_now = iqa_ssim(sr, hr).item()
			SSIM_all += ssim_now
			metrics['ssim'].append(ssim_now)

			# lpips
			lpips_now = iqa_lpips(sr, hr).item()
			lpips_all += lpips_now
			metrics['lpips'].append(lpips_now)

			logger.info('{:20s}_{} - PSNR: {:.6f} dB; SSIM: {:.6f};  LPIPS: {:.6f}'.format(opt.init_imgs_names[dir_idx], img_sr_name, PSNR_now, ssim_now, lpips_now))

		PSNR_all = round(PSNR_all/len(img_sr_list) , 4)
		SSIM_all = round(SSIM_all/len(img_sr_list) , 4)
		lpips_all = round(lpips_all/len(img_sr_list) , 4)

		logger.info('{:20s}_all - PSNR: {:.6f} dB; SSIM: {:.6f};  LPIPS: {:.6f}'.format(opt.init_imgs_names[dir_idx], PSNR_all, SSIM_all, lpips_all))

		# save metrics
		npy_path = os.path.join(opt.log, 'test_' + opt.log_name + '_npy')
		os.makedirs(npy_path, exist_ok=True)
		np.save(npy_path + '/' + opt.init_imgs_names[dir_idx]+'.npy', metrics)

main()