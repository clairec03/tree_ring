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
import open_clip
from methods.optim_utils import inject_watermark, get_watermarking_pattern, get_watermarking_mask, eval_watermark
from methods.io_utils import *
from methods._detect import detect_key
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler 
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

class BaseWatermarkedDiffusionPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_pipeline()

    def load_pipeline(self):
        # Constants for SDXL-Lightning
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        checkpoint = "sdxl_lightning_4step_unet.safetensors"
        
        scheduler = DPMSolverMultistepScheduler.from_pretrained(base, subfolder='scheduler')

        # Initialize pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        
        # # Set up scheduler
        # pipe.scheduler = EulerDiscreteScheduler.from_config(
        #     pipe.scheduler.config,
        #     timestep_spacing="trailing",
        #     prediction_type="epsilon"
        # )
        
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

class TreeRingWatermark:
    # Source: https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/run_tree_ring_watermark.py
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_pipeline()

    def load_pipeline(self):
        # Constants for SDXL-Lightning
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        checkpoint = "sdxl_lightning_4step_unet.safetensors"
        
        # # Initialize pipeline
        # vae = AutoencoderKL.from_pretrained(base) 
        # # Load the text encoder and tokenizer 
        # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14") 
        # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") 
        # # Load the UNet model
        # unet = UNet2DConditionModel.from_pretrained(base)
        # scheduler = DDIMScheduler.from_pretrained(base)

        # pipe = InversableStableDiffusionPipeline(
        #     vae,
        #     text_encoder,
        #     tokenizer,
        #     unet,
        #     scheduler,
        # )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        
        # Set up scheduler
        # pipe.scheduler = EulerDiscreteScheduler.from_config(
        #     pipe.scheduler.config,
        #     timestep_spacing="trailing",
        #     prediction_type="epsilon"
        # )
        
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
        self.model = self.load_pipeline()

        self.text_embeddings = self.model.get_text_embedding(prompt)

        # num_inference_steps = 4
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.model.scheduler, num_inference_steps, self.device
        # )

        # init_latents_w = self.model.get_random_latents()
        # # self.init_latents_w = init_latents_w

        # # get watermarking mask
        # watermarking_mask = get_watermarking_mask(init_latents_w, self.device)
        # # self.watermarking_mask = watermarking_mask

        # gt_patch = get_watermarking_pattern(self.model, self.device, key, init_latents_w.shape, init_latents_w.dtype)
        # # self.gt_patch = gt_patch

        # # inject watermark
        # # init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch)

        outputs_w = self.model(
            prompt=prompt,
            key = key,
            num_images_per_prompt=1,
            guidance_scale=0.5, # args.guidance_scale,
            num_inference_steps=4, # args.num_inference_steps,
            # height=300, # args.image_length,
            # width=300, # args.image_length,
            # latents=init_latents_w,
            **generate_kwargs,
        )
        
        orig_image_w = outputs_w.images[0]

        return ToPILImage()(orig_image_w) # TODO: check if need to convert to an image object 


    def detect(self, image: Image.Image) -> int:
        # return np.random.randint(0, 8)
        # Simple example of watermark detection (students should improve this)
        """
        Detects a watermark in an image.
        return a random integer from 1 to 100 for the baseline
        """
        # reverse img with watermarking
        pipe = self.model
        transform = transforms.Compose([ 
            transforms.ToTensor() # convert to torch tensor 
        ]) 
        # Apply the transformation 
        image_tensor = transform(image)
        image_latents_w = pipe.get_image_latents(image_tensor, sample=False)

        # reversed_latents_w = pipe.forward_diffusion(
        #     latents=image_latents_w,
        #     text_embeddings=self.text_embeddings,
        #     guidance_scale=1,
        #     num_inference_steps=4,
        # )

        key = detect_key(image, pipe=self.model)

        if key is None:
            return np.random.randint(0, 8)
        
        return key
