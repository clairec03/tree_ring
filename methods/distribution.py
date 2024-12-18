# methods/watermarked_diffusion_pipeline.py

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

from methods.inverse_stable_diffusion import InversableStableDiffusionPipeline
from methods.modified_stable_diffusion import retrieve_timesteps
from diffusers import DPMSolverMultistepScheduler
from methods.optim_utils import inject_watermark, get_watermarking_pattern, get_watermarking_mask, eval_watermark
from methods.io_utils import *
from methods._detect import detect_key
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler 
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

import time

class DistributionPipeline:
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

    def adjust_odds(self, array, k):
        # Copy the array to avoid modifying the original
        adjusted_array = array.copy()
        # Iterate through each element in the array
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if adjusted_array[i, j] % 2 == k:  # Check if the number is odd
                    if np.random.rand() < 0.9:  # Apply 90% probability
                        # Adjust to the nearest even number
                        adjusted_array[i, j] = adjusted_array[i, j] - 1 if adjusted_array[i, j] > 0 else adjusted_array[i, j] + 1
        return adjusted_array

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

        image_np = np.array(image)

        base2 = ['000', '001', '010', '011', '100', '101', '110', '111']
        key_t = base2[key]

        # import pdb; pdb.set_trace()

        for d in range(3):
            image_np[:,:,d] = self.adjust_odds(image_np[:,:,d], int(key_t[d]))
        
        image_dist = Image.fromarray(image_np)

        # Save the image
        image_dist.save(f"generated_image_{time.time()}.png")

        return image_dist

    def compare(self, count):
        print(count)
        if count >= 1024 * 1024 * 0.6:
            return 1
        
        if count <= 1024 * 1024 * 0.4:
            return 0
        return None

    def detect(self, image: Image.Image) -> int:
        # Simple example of watermark detection (students should improve this)
        """
        Detects a watermark in an image.
        return a random integer from 1 to 100 for the baseline
        """
        image_np = np.array(image)
        key = [self.compare(np.sum(image_np[:,:,d] % 2 == 0)) for d in range(3)]
        return key[0] * 4 + key[1] * 2 + key[2]