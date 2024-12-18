# Adapted from: https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/src/tree_ring_watermark/_detect.py
from huggingface_hub import snapshot_download
import numpy as np
import torch
from torchvision import transforms
import PIL
from typing import Union
from diffusers import DDIMInverseScheduler
from ._get_noise import _circle_mask
import os
from torch.fft import fft2, fftshift 
from methods.optim_utils import inject_watermark, get_watermarking_pattern, get_watermarking_mask, eval_watermark

# def load_keys(cache_dir):
#     # Initialize an empty dictionary to store the numpy arrays
#     arrays = {}

#     # List all files in the directory
#     for file_name in os.listdir(cache_dir):
#         # Check if the file is a .npy file
#         if file_name.endswith('.npy'):
#             # Define the file path
#             file_path = os.path.join(cache_dir, file_name)

#             # Load the numpy array and store it in the dictionary
#             arrays[file_name] = np.load(file_path)

#     # Return the 'arrays' dictionary
#     return arrays

def get_w_key(key, shape): 
    # Create a tensor from the integer key 
    key_pattern = (key - 3.5) * 90
    key_tensor = torch.ones(shape, dtype=torch.float16) * key_pattern
    # Compute the Fourier transform 
    # w_key = fftshift(fft2(key_tensor))
    return key_tensor

# def get_w_key_2(key, shape): 
#     # Create a tensor with random values based on the key for more meaningful Fourier noise 
#     rng = torch.Generator().manual_seed(key)
#      # Seed with the key 
#     key_tensor = torch.randn(shape, generator=rng, dtype=torch.float16) 
#     # Compute the Fourier transform 
#     w_key = fftshift(fft2(key_tensor)) 
#     return w_key


def _transform_img(image, target_size=1024):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return image #2.0 * image - 1.0


# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def detect_key(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe):
    org = get_org()
    repo_id = os.path.join(org, model_hash)
    cache_dir = snapshot_download(repo_id, repo_type="dataset")
    keys = load_keys(cache_dir)
    detection_time_num_inference = 100 # TODO: to be adjusted
    print("keys")
    print(keys.items())
    # detection_time_num_inference = 50
    # threshold = 77 # TODO: to be adjusted

    
    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    # upcast vae to fp32 to prevent overflow
    pipe.vae.config.force_upcast = True
    pipe.upcast_vae()
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    # print("pipe.config", pipe.config)
    # print("img dtype before", img.dtype)
    # img = img.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    # img = img.to(torch.float32)
    img = img.to(torch.float32)
    # print("img dtype after", img.dtype)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    # print("image_latents dtype", image_latents.dtype)
    # image_latents = pipe.get_image_latents(img).type(torch.float16)
    if torch.isnan(image_latents).any() or torch.isinf(image_latents).any(): 
        print("NaNs or Infs found in image_latents before inversion")
    
    print("detecting key") 
    pipe = pipe.to(torch.float32)
    inverted_latents = pipe(
        prompt='',
        latents=image_latents,
        guidance_scale=1,
        num_inference_steps=detection_time_num_inference,
        output_type='latent',
    )
    # inverted_latents = inverted_latents.images.float().cpu()
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents.images), dim=(-1, -2))
    print("==========START OF detect_key===========")
    # print("detect_key")
    print(inverted_latents_fft.std())
    # check if one key matches
    shape = image_latents.shape

    keys = list(range(8))

    min_dist, best_key = 1e8, None
    
    shape = (1, 4, 128, 128)

    latents, w_key, w_mask = get_noise(shape, pipe)
    
    for key in keys:    
        # w_key = get_w_key(key, shape)
        # w_radius = 10

        # np_mask = _circle_mask(shape[-1], r=int(w_radius))
        # torch_mask = torch.tensor(np_mask)
        # w_mask = torch.zeros(shape, dtype=torch.bool)
        # w_mask[:, :] = torch_mask
        w_key = get_watermarking_pattern(pipe, pipe.device, key, inverted_latents_fft.shape, inverted_latents_fft.dtype)
        w_mask = get_watermarking_mask(inverted_latents_fft, pipe.device)

        # calculate the distance

        # # Check and handle NaN or infinite values 
        # if torch.isnan(inverted_latents_fft).any() or torch.isinf(inverted_latents_fft).any(): 
        #     print(f"NaNs or Infs found in inverted_latents_fft for key {key}") 
            
        # if torch.isnan(w_key).any() or torch.isinf(w_key).any(): 
        #     print(f"NaNs or Infs found in w_key for key {key}") 
            
        # # Replace NaNs or Infs with zeros 
        # inverted_latents_fft = torch.nan_to_num(inverted_latents_fft) 
        # w_key = torch.nan_to_num(w_key)

        # # Compute dist metric
        # import pdb; pdb.set_trace()
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        print("key", key, "dist", dist)
        
        if dist <= min_dist:
            best_key = key
            min_dist = dist
        
    # print("=========END OF detect_key============")
    pipe.scheduler = curr_scheduler
    return best_key


def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe, model_hash):
    detection_time_num_inference = 50
    threshold = 77

    org = get_org()
    repo_id = os.path.join(org, model_hash)

    cache_dir = snapshot_download(repo_id, repo_type="dataset")
    keys = load_keys(cache_dir)

    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
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
    for filename, w_key in keys.items():
        w_channel, w_radius = filename.split(".npy")[0].split("_")[1:3]

        np_mask = _circle_mask(shape[-1], r=int(w_radius))
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, int(w_channel)] = torch_mask

        # calculate the distance
        inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        if dist <= threshold:
            pipe.scheduler = curr_scheduler
            return True

    return False
