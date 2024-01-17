# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import math
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path
import sys

# run git clone https://github.com/CompVis/taming-transformers.git
sys.path.insert(0, "taming-transformers")

from model.q_sampler import SpacedSampler
from model.ccsr_stage1 import ControlLDM
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        config_file = "configs/model/ccsr_stage2.yaml"
        self.model = instantiate_from_config(OmegaConf.load(config_file))
        load_state_dict(
            self.model,
            torch.load("weights/real-world_ccsr.ckpt", map_location="cpu"),
            strict=True,
        )
        self.model.freeze()
        self.model.to("cuda")

    def predict(
        self,
        image: Path = Input(description="Low quality input image."),
        sr_scale: float = Input(description="Super-resolution scale.", default=4),
        steps: int = Input(
            description="Number of sampling steps", ge=1, le=500, default=45
        ),
        tile_diffusion: bool = Input(
            description="If specified, a patch-based sampling strategy for diffusion peocess will be used for sampling.",
            default=False,
        ),
        tile_diffusion_size: int = Input(
            description="Size of patch for diffusion process.",
            default=512,
        ),
        tile_diffusion_stride: int = Input(
            description="Stride of sliding patch for diffusion process.", default=256
        ),
        tile_vae: bool = Input(
            description="If specified, a patch-based sampling strategy for the encoder and decoder in VAE will be used.",
            default=False,
        ),
        vae_decoder_tile_size: int = Input(
            description="Size of patch for VAE decoder, latent size.",
            default=224,
        ),
        vae_encoder_tile_size: int = Input(
            description="Size of patch for VAE encoder, image size.", default=1024
        ),
        color_fix_type: str = Input(
            description="Size of patch.",
            choices=["wavelet", "adain", "none"],
            default="adain",
        ),
        t_max: float = Input(
            description="The starting point of uniform sampling strategy.",
            default=0.6667,
        ),
        t_min: float = Input(
            description="The ending point of uniform sampling strategy.", default=0.3333
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        pl.seed_everything(seed)

        lq = Image.open(str(image)).convert("RGB")
        if not sr_scale == 1:
            lq = lq.resize(
                tuple(math.ceil(x * sr_scale) for x in lq.size), Image.BICUBIC
            )
        lq_resized = (
            auto_resize(lq, tile_diffusion_size)
            if tile_diffusion
            else auto_resize(lq, 512)
        )
        x = lq_resized.resize(
            tuple(s // 64 * 64 for s in lq_resized.size), Image.LANCZOS
        )
        x = np.array(x)

        pred = process(
            self.model,
            [x],
            steps=steps,
            t_max=t_max,
            t_min=t_min,
            strength=1,
            color_fix_type=color_fix_type,
            tile_diffusion=tile_diffusion,
            tile_diffusion_size=tile_diffusion_size,
            tile_diffusion_stride=tile_diffusion_stride,
            tile_vae=tile_vae,
            vae_decoder_tile_size=vae_decoder_tile_size,
            vae_encoder_tile_size=vae_encoder_tile_size,
        )[0]

        out_path = "/tmp/out.png"
        Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(out_path)
        return Path(out_path)


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: list[np.ndarray],
    steps: int,
    t_max: float,
    t_min: float,
    strength: float,
    color_fix_type: str,
    tile_diffusion: bool,
    tile_diffusion_size: int,
    tile_diffusion_stride: int,
    tile_vae: bool,
    vae_decoder_tile_size: int,
    vae_encoder_tile_size: int,
) -> tuple[list[np.ndarray]]:
    """
    Apply CCSR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        t_max (float): The starting point of uniform sampling strategy.
        t_min (float): The ending point of uniform sampling strategy.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        tile_diffusion (bool): If specified, a patch-based sampling strategy for diffusion peocess will be used for sampling.
        tile_diffusion_size (int): Size of patch for diffusion peocess.
        tile_diffusion_stride (int): Stride of sliding patch for diffusion peocess.
        tile_vae (bool): If specified, a patch-based sampling strategy for the encoder and decoder in VAE will be used.
        vae_decoder_tile_size (int): Size of patch for VAE decoder.
        vae_encoder_tile_size (int): Size of patch for VAE encoder.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
    """

    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(
        np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device
    ).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    model.control_scales = [strength] * 13

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if not tile_diffusion and not tile_vae:
        samples = sampler.sample_ccsr(
            steps=steps,
            t_max=t_max,
            t_min=t_min,
            shape=shape,
            cond_img=control,
            positive_prompt="",
            negative_prompt="",
            x_T=x_T,
            cfg_scale=1.0,
            color_fix_type=color_fix_type,
        )
    else:
        if tile_vae:
            model._init_tiled_vae(
                encoder_tile_size=vae_encoder_tile_size,
                decoder_tile_size=vae_decoder_tile_size,
            )
        if tile_diffusion:
            samples = sampler.sample_with_tile_ccsr(
                tile_size=tile_diffusion_size,
                tile_stride=tile_diffusion_stride,
                steps=steps,
                t_max=t_max,
                t_min=t_min,
                shape=shape,
                cond_img=control,
                positive_prompt="",
                negative_prompt="",
                x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type,
            )
        else:
            samples = sampler.sample_ccsr(
                steps=steps,
                t_max=t_max,
                t_min=t_min,
                shape=shape,
                cond_img=control,
                positive_prompt="",
                negative_prompt="",
                x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type,
            )

    x_samples = samples.clamp(0, 1)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    preds = [x_samples[i] for i in range(n_samples)]

    return preds
