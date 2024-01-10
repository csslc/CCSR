from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
from torch.nn import functional as F
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.q_sampler import SpacedSampler
from model.ccsr_stage1 import ControlLDM
from model.cond_fn import MSEGuidance
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    t_max: float,
    t_min: float,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply CCSR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        t_max (float):
        t_min (float):
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13
    
    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)
    
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    init_latent = model.encode_first_stage(control)
    init_latent = model.get_first_stage_encoding(init_latent)
    if not tiled:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0, cond_fn=cond_fn,
        #     color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff_ccsr(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    
    return preds, stage1_preds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt", type=str, help="full checkpoint path", default='/home/notebook/data/group/SunLingchen/code/CCSR/CCSR_weights/step=59.ckpt')
    parser.add_argument("--config", type=str, help="model config path", default='configs/model/ccsr_stage2.yaml')
    
    parser.add_argument("--input", type=str, default='inputs/real47')
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--sr_scale", type=float, default=4)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    
    parser.add_argument("--color_fix_type", type=str, default="adain", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str,default="experiments/output")
    parser.add_argument("--t_max", type=float, default=0.6667)
    parser.add_argument("--t_min", type=float, default=0.3333)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device

def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    args.device = check_device(args.device)
    
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    # reload preprocess model if specified
    args.reload_swinir = False
    args.disable_preprocess_model = True
    if args.reload_swinir:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {args.swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(args.swinir_ckpt, map_location="cpu"), strict=True)
    model.freeze()
    model.to(args.device)
    
    assert os.path.isdir(args.input)
    args.input_list = [args.input]
    for file_path in list_image_files(args.input_list, follow_links=True):
        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        if not args.tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
        x = pad(np.array(lq_resized), scale=64)
        
        for i in range(args.repeat_times):
            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path_now = os.path.join(parent_path, 'sample'+str(i))
            
            save_path = os.path.join(save_path_now, f"{stem}.png")
            if os.path.exists(save_path):
                if args.skip_if_exist:
                    print(f"skip {save_path}")
                    continue
                else:
                    raise RuntimeError(f"{save_path} already exist")
            # os.makedirs(parent_path, exist_ok=True)
            os.makedirs(save_path_now, exist_ok=True)
            
            # initialize latent image guidance
            cond_fn = None
            
            preds, stage1_preds = process(
                model, [x], steps=args.steps,
                t_max=args.t_max,  t_min=args.t_min,
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                cond_fn=cond_fn,
                tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride
            )
            pred, stage1_pred = preds[0], stage1_preds[0]
            
            # remove padding
            pred = pred[:lq_resized.height, :lq_resized.width, :]
            
            if args.show_lq:
                pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
                stage1_pred = np.array(Image.fromarray(stage1_pred).resize(lq.size, Image.LANCZOS))
                lq = np.array(lq)
                images = [lq, pred] if args.disable_preprocess_model else [lq, stage1_pred, pred]
                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
                # pred.save(save_path)
            print(f"save to {save_path}")

if __name__ == "__main__":
    main()
