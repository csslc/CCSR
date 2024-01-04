# evaluate the G-STD of restored N images


import os
import numpy as np
import glob
from datetime import datetime
import logging

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

# log_name
name_log = 'DIV2K-valid'
path_log = '****/G-STD'
os.makedirs(path_log, exist_ok=True)

# load metrics
npy_file_path = '****'
npy_file_lists = sorted(glob.glob(os.path.join(npy_file_path, '*npy')))
l_sample = len(npy_file_lists)
a = np.load(npy_file_lists[0], allow_pickle=True).item()
l_file = len(a['psnr'])

# init logger
setup_logger('base', path_log, 'test_' + name_log, level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(name_log)

# init the metrics: you can add other metrics here
metric_psnr = np.zeros([l_file, l_sample])
metric_ssim = np.zeros([l_file, l_sample])
metric_lpips = np.zeros([l_file, l_sample])

i = 0
for npy_file in npy_file_lists:
    # npy_file_list = sorted(glob.glob(os.path.join(npy_files, '*')))
    a = np.load(npy_file, allow_pickle=True).item()
    metric_psnr[:, i] = np.array(a['psnr'])
    metric_ssim[:, i] = np.array(a['ssim'])
    metric_lpips[:, i] = np.array(a['lpips'])
    i = i + 1
        
# calculate the mean of the metrics
mean_psnr = np.mean(metric_psnr, axis=1)
mean_ssim = np.mean(metric_ssim, axis=1)
mean_lpips = np.mean(metric_lpips, axis=1)

mean_mean_psnr = np.mean(mean_psnr)
mean_mean_ssim = np.mean(mean_ssim)
mean_mean_lpips = np.mean(mean_lpips)


# calculate the std of the metrics
std_psnr = np.std(metric_psnr, axis=1)
std_ssim = np.std(metric_ssim, axis=1)
std_lpips = np.std(metric_lpips, axis=1)

mean_std_psnr = np.mean(std_psnr)
mean_std_ssim = np.mean(std_ssim)
mean_std_lpips = np.mean(std_lpips)


logger.info('mean_mean - PSNR: {:.6f} dB; SSIM: {:.6f};  LPIPS: {:.6f}'.format(mean_mean_psnr, mean_mean_ssim, mean_mean_lpips))

logger.info('G_STD - PSNR: {:.6f} dB; SSIM: {:.6f};  LPIPS: {:.6f}'.format(mean_std_psnr, mean_std_ssim, mean_std_lpips))