# Source: https://github.com/varshakishore/FNNS/blob/main/demo.ipynb
# import relevant packages

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np

import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch
import numpy as np 
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from steganogan import SteganoGAN

from torch.optim import LBFGS
import torch.nn.functional as F
import os

# set seed
seed = 11111
np.random.seed(seed)

# set paramaters
# The mode can be random, pretrained-de or pretrained-d. Refer to the paper for details
mode = "pretrained-d"
steps = 2000
max_iter = 10
alpha = 0.1
eps = 0.3
num_bits = 3

# Resolving 

# some pre-trained steganoGAN models can be found here: 
# https://drive.google.com/drive/folders/1-U2NDKUfqqI-Xd5IqT1nkymRQszAlubu?usp=sharing
model_path = os.getcwd() + "/pretrained_stegogan_models/mscoco_3bits.steg"

steganogan = SteganoGAN.load(path=model_path, cuda=True, verbose=True)
print("steganogan model loaded!")

torch.serialization.add_safe_globals([SteganoGAN])
torch.nn.Module.dump_patches = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model loaded successfully on device:", device)

input_im = "generated_images/generated_image_9.png"
output_im = "test_stego.png"

inp_image = imread(input_im, pilmode='RGB')

# you can add a custom target message here
target = torch.bernoulli(torch.empty(1, num_bits, inp_image.shape[1], inp_image.shape[0]).uniform_(0, 1)).to('cuda')

target = "secret msg"
print(target)

# Not sure if this works
# # Define a custom 3-bit target message
# custom_message = "101"  # Example 3-bit message
# target = torch.tensor([int(bit) for bit in custom_message], dtype=torch.float32).view(1, num_bits, 1, 1).expand(1, num_bits, inp_image.shape[1], inp_image.shape[0]).to('cuda')


steganogan.encode(input_im, output_im, target)
output = steganogan.decode(output_im)

if mode == "pretrained-de":
    image = output_im
else:
    image = input_im

image = imread(image, pilmode='RGB') / 255.0
image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
image = image.to('cuda')


im1 = np.array(imread(input_im, pilmode='RGB')).astype(float)
im2 = np.array(imread(output_im, pilmode='RGB')).astype(float)
print("PSNR:", peak_signal_noise_ratio(im1, im2, data_range=255))
print("SSIM:",structural_similarity(im1, im2, data_range=255, multichannel=True))
print("target:", target)
print("decoded output:", output)
err = ((target !=output.float()).sum().item()+0.0)/target.numel()
print("Iniitial error:", err)

# FNNS Optimization
model = steganogan.decoder
print("Model:", model)
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')


out = model(image)
target = target.to(out.device)

count = 0

adv_image = image.clone().detach()

for i in range(steps // max_iter):
    adv_image.requires_grad = True
    optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)

    def closure():
        outputs = model(adv_image)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)
    delta = torch.clamp(adv_image - image, min=-eps, max=eps)
    adv_image = torch.clamp(image + delta, min=0, max=1).detach()

    err = len(torch.nonzero((model(adv_image)>0).float().view(-1) != target.view(-1))) / target.numel()
    print("Error:", err)
    if err < 0.00001: eps = 0.7
    if err==0: count+=1; eps = 0.3
    if count==10: break


# Final stats
print("PSNR:", peak_signal_noise_ratio(np.array(imread(input_im, pilmode='RGB')).astype(float), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), data_range=255))
print("SSIM:", structural_similarity(np.array(imread(input_im, pilmode='RGB')).astype(float), (adv_image.squeeze().permute(2,1,0)*255).detach().cpu().numpy(), data_range=255, multichannel=True))
print("Error:", err)
lbfgsimg = (adv_image.cpu().squeeze().permute(2,1,0).numpy()*255).astype(np.uint8)

Image.fromarray(lbfgsimg).save(output_im)
image_read = imread(output_im, pilmode='RGB') / 255.0
image_read = torch.FloatTensor(image_read).permute(2, 1, 0).unsqueeze(0).to('cuda')

print("\nAfter writing to file and reading from file")
im1 = np.array(imread(input_im, pilmode='RGB')).astype(float)
im2 = np.array(imread(output_im, pilmode='RGB')).astype(float)
print("PSNR:", peak_signal_noise_ratio(im1, im2, data_range=255))
print("SSIM:", structural_similarity(im1, im2, data_range=255, multichannel=True))
print("Error:", len(torch.nonzero((model(image_read)>0).float().view(-1) != target)))


class BaseWatermarkedDiffusionPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_pipeline()

    def load_pipeline(self):
        # Constants for SDXL-Lightning
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        checkpoint = "sdxl_lightning_4step_unet.safetensors"
        
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(base, subfolder='scheduler')

        # Initialize pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        
        # # Set up scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            prediction_type="epsilon"
        )
        
        # Load the model weights
        pipe.unet.load_state_dict(
            load_file(
                hf_hub_download(repo, checkpoint),
                device=self.device
            )
        )
        return pipe

    def generate(self, prompt: str, key: int = None, **generate_kwargs) -> Image.Image:
        """
        Generates an image from the prompt, embedding a watermark if a key is provided.

        Args:
            prompt (str): The text prompt for image generation.
            key (int, optional): An integer key used to embed the watermark.

        Returns:
            Image: Generated (and potentially watermarked) image.
        """
        # Generate image using the customized model
        image = self.model(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
            **generate_kwargs
        ).images[0]

        return image


    def detect(self, image: Image.Image) -> int:
        # Simple example of watermark detection (students should improve this)
        """
        Detects a watermark in an image.
        return a random integer from 1 to 100 for the baseline
        """
        return np.random.randint(0, 8)


class FNNSWatermarkPipeline(BaseWatermarkedDiffusionPipeline):
    def __init__(self, device: str = "cuda"):
        self.load_pipeline()

    def load_pipeline(self):
        # some pre-trained steganoGAN models can be found here: 
        # https://drive.google.com/drive/folders/1-U2NDKUfqqI-Xd5IqT1nkymRQszAlubu?usp=sharing
        model_path = os.getcwd() + "/pretrained_stegogan_models/mscoco_3bits.steg"

        steganogan = SteganoGAN.load(path=model_path, cuda=True, verbose=True)
        print("steganogan model loaded!")

        torch.serialization.add_safe_globals([SteganoGAN])
        torch.nn.Module.dump_patches = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Model loaded successfully on device:", device)
        self.model = steganogan


    def generate(self, prompt: str, key: int = None, **generate_kwargs) -> Image.Image:
        raise NotImplementedError

    def detect(self, image: Image.Image) -> int:
        raise NotImplementedError
