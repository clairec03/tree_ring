# Adapted from: https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/src/tree_ring_watermark/_detect.py
import numpy as np
import torch
from torchvision import transforms
import PIL
from typing import Union
from diffusers import DDIMInverseScheduler
from ._get_noise import _circle_mask
import os
from torch.fft import fft2, fftshift 


def get_w_key(key, shape): 
    # Create a tensor from the integer key 
    key_tensor = torch.ones(shape, dtype=torch.float32) * key 
    # Compute the Fourier transform 
    w_key = fftshift(fft2(key_tensor)) 
    return w_key

def get_w_key_2(key, shape): 
    # Create a tensor with random values based on the key for more meaningful Fourier noise 
    rng = torch.Generator().manual_seed(key)
     # Seed with the key 
    key_tensor = torch.randn(shape, generator=rng, dtype=torch.float32) 
    # Compute the Fourier transform 
    w_key = fftshift(fft2(key_tensor)) 
    return w_key


def _transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def detect_key(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe):
    detection_time_num_inference = 4
    # detection_time_num_inference = 50
    threshold = 77 # TODO: to be adjusted

    
    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    print(f"Image latents before inversion: {image_latents}") 
    if torch.isnan(image_latents).any() or torch.isinf(image_latents).any(): 
        print("NaNs or Infs found in image_latents before inversion")
              
    inverted_latents = pipe(
        prompt='',
        latents=image_latents,
        guidance_scale=1,
        num_inference_steps=detection_time_num_inference,
        output_type='latent',
    )
    inverted_latents = inverted_latents.images.float().cpu()

    # check if one key matches
    shape = image_latents.shape

    keys = list(range(8))

    min_dist, best_key = 1e20, None
    
    for key in keys:    
        w_key = get_w_key(key, shape)
        w_radius = 10

        np_mask = _circle_mask(shape[-1], r=int(w_radius))
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, :] = torch_mask

        # calculate the distance
        inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))

        # Check and handle NaN or infinite values 
        if torch.isnan(inverted_latents_fft).any() or torch.isinf(inverted_latents_fft).any(): 
            print(f"NaNs or Infs found in inverted_latents_fft for key {key}") 
            
        if torch.isnan(w_key).any() or torch.isinf(w_key).any(): 
            print(f"NaNs or Infs found in w_key for key {key}") 
            
        # Replace NaNs or Infs with zeros 
        inverted_latents_fft = torch.nan_to_num(inverted_latents_fft) 
        w_key = torch.nan_to_num(w_key)

        # Compute dist metric
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        print("key", key, "dist", dist)

        if dist <= min_dist:
            best_key = w_key
            min_dist = dist
        
    pipe.scheduler = curr_scheduler
    return best_key