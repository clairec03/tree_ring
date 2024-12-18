#!/usr/bin/env python3
import torch
import torchvision

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, DDIMScheduler
from methods.inverse_stable_diffusion import InversableStableDiffusionPipeline

from methods._get_noise import get_noise
from methods._detect import detect

from PIL import Image
import requests
from io import BytesIO

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
checkpoint = "sdxl_lightning_4step_unet.safetensors" 

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# IMPORTANT: We need to make sure to be able to use a normal diffusion pipeline so that people see 
# the tree-ring-watermark method as general enough
scheduler = DPMSolverMultistepScheduler.from_pretrained(base, subfolder='scheduler')
# or
# scheduler = DDIMScheduler.from_pretrained(base, subfolder='scheduler')

pipe = InversableStableDiffusionPipeline.from_pretrained(
            base,
            torch_dtype=torch.float32,
            variant="fp16"
        ).to(device)
pipe.unet.load_state_dict(
    load_file(
        hf_hub_download(repo, checkpoint),
        device=device
    )
)
shape = (1, 4, 128, 128)

latents, w_key, w_mask = get_noise(shape, pipe)

watermarked_image = pipe(prompt="an astronaut", latents=latents).images[0]

is_watermarked = detect(watermarked_image, pipe, w_key, w_mask)
print(f'is_watermarked: {is_watermarked}')
