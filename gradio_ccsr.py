from typing import List
import math
from argparse import ArgumentParser

import numpy as np
import torch
import einops
import pytorch_lightning as pl
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.xformers_state import disable_xformers
from model.q_sampler import SpacedSampler
from model.ccsr_stage1 import ControlLDM
from utils.image import auto_resize
from utils.common import instantiate_from_config, load_state_dict

parser = ArgumentParser()
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()

# load model
if args.device == "cpu":
    disable_xformers()
model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
model.freeze()
model.to(args.device)
# load sampler
sampler = SpacedSampler(model, var_type="fixed_small")


@torch.no_grad()
def process(
        control_img: Image.Image,
        num_samples: int,
        sr_scale: int,
        strength: float,
        positive_prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        steps: int,
        use_color_fix: bool,
        seed: int,
        tile_diffusion: bool,
        tile_diffusion_size: int,
        tile_diffusion_stride: int,
        tile_vae: bool,
        vae_encoder_tile_size: int,
        vae_decoder_tile_size: int
) -> List[np.ndarray]:
    print(
        f"control image shape={control_img.size}\n"
        f"num_samples={num_samples}, sr_scale={sr_scale}, strength={strength}\n"
        f"positive_prompt='{positive_prompt}', negative_prompt='{negative_prompt}'\n"
        f"cdf scale={cfg_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
        f"seed={seed}\n"
        f"tile_diffusion={tile_diffusion}, tile_diffusion_size={tile_diffusion_size}, tile_diffusion_stride={tile_diffusion_stride}"
        f"tile_vae={tile_vae}, vae_encoder_tile_size={vae_encoder_tile_size}, vae_decoder_tile_size={vae_decoder_tile_size}"
    )
    pl.seed_everything(seed)

    # resize lr
    if sr_scale != 1:
        control_img = control_img.resize(
            tuple(math.ceil(x * sr_scale) for x in control_img.size),
            Image.BICUBIC
        )

    input_size = control_img.size

    # resize the lr image to 512
    if not tile_diffusion:
        control_img = auto_resize(control_img, 512)
    else:
        control_img = auto_resize(control_img, tile_diffusion_size)

    # resize image to be multiples of 64
    control_img = control_img.resize(
        tuple((s // 64 + 1) * 64 for s in control_img.size), Image.LANCZOS
    )
    control_img = np.array(control_img)

    # convert to tensor (NCHW, [0,1])
    control = torch.tensor(control_img[None] / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    height, width = control.size(-2), control.size(-1)
    model.control_scales = [strength] * 13

    preds = []
    for _ in tqdm(range(num_samples)):
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
        if not tile_diffusion and not tile_vae:
            samples = sampler.sample_ccsr(
                steps=steps, t_max=0.6667, t_min=0.3333, shape=shape, cond_img=control,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale,
                color_fix_type="adain" if use_color_fix else "none"
            )
        else:
            if tile_vae:
                model._init_tiled_vae(encoder_tile_size=vae_encoder_tile_size, decoder_tile_size=vae_decoder_tile_size)
            if tile_diffusion:
                samples = sampler.sample_with_tile_ccsr(
                    tile_size=tile_diffusion_size, tile_stride=tile_diffusion_stride,
                    steps=steps, t_max=0.6667, t_min=0.3333, shape=shape, cond_img=control,
                    positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                    cfg_scale=cfg_scale,
                    color_fix_type="adain" if use_color_fix else "none"
                )
            else:
                samples = sampler.sample_ccsr(
                    steps=steps, t_max=0.6667, t_min=0.3333, shape=shape, cond_img=control,
                    positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                    cfg_scale=cfg_scale,
                    color_fix_type="adain" if use_color_fix else "none"
                )

        x_samples = samples.clamp(0, 1)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(
            np.uint8)

        # resize to input size
        img = Image.fromarray(x_samples[0, ...]).resize(input_size, Image.LANCZOS)

        preds.append(np.array(img))

    return preds


MARKDOWN = \
    """
    ## Improving the Stability of Diffusion Models for Content Consistent Super-Resolution

    [GitHub](https://github.com/csslc/CCSR) | [Paper](https://arxiv.org/pdf/2401.00877.pdf) | [Project Page](https://csslc.github.io/project-CCSR/)

    If CCSR is helpful for you, please help star the GitHub Repo. Thanks!
    """

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                tile_diffusion = gr.Checkbox(label="Tile diffusion", value=False)
                tile_diffusion_size = gr.Slider(label="Tile diffusion size", minimum=512, maximum=1024, value=512,
                                                step=256)
                tile_diffusion_stride = gr.Slider(label="Tile diffusion stride", minimum=256, maximum=512, value=256,
                                                  step=128)
                tile_vae = gr.Checkbox(label="Tile VAE", value=True)
                vae_encoder_tile_size = gr.Slider(label="Encoder tile size", minimum=512, maximum=5000, value=1024,
                                                  step=256)
                vae_decoder_tile_size = gr.Slider(label="Decoder tile size", minimum=64, maximum=512, value=224,
                                                  step=128)
                num_samples = gr.Slider(label="Number Of Samples", minimum=1, maximum=12, value=1, step=1)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=45, step=1)
                sr_scale = gr.Number(label="SR Scale", value=4)
                positive_prompt = gr.Textbox(label="Positive Prompt", value="")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)",
                                      minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)

                use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=2,
                                                                                                   height="auto")

    inputs = [
        input_image,
        num_samples,
        sr_scale,
        strength,
        positive_prompt,
        negative_prompt,
        cfg_scale,
        steps,
        use_color_fix,
        seed,
        tile_diffusion,
        tile_diffusion_size,
        tile_diffusion_stride,
        tile_vae,
        vae_encoder_tile_size,
        vae_decoder_tile_size,
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])
block.launch()