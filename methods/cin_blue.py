# methods/watermarked_diffusion_pipeline.py

import sys
sys.path.append('methods/CIN/codes')
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

from methods.inverse_stable_diffusion import InversableStableDiffusionPipeline
from methods.modified_stable_diffusion import retrieve_timesteps
from diffusers import DPMSolverMultistepScheduler
from methods.optim_utils import inject_watermark, get_watermarking_pattern, get_watermarking_mask, eval_watermark
from methods.io_utils import *
from methods._detect import detect_key
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler 
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms


import torch
from utils.yml import parse_yml, dict_to_nonedict, set_random_seed, dict2str
import random
import os
import time
import torch
import utils.utils as utils
import logging
from models.Network import Network
class CINBlue:
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

        image_np = (np.array(image) / 255.0).astype(np.float32)
        message = torch.from_numpy(np.ones((1,30)).astype(np.float32)).to("cuda")

        image_wm = self.post_wm(image_np, message)

        # Save the image
        image_wm.save(f"generated_image_{time.time()}.png")

        return image_wm

    def post_wm(self, image, message):

        name = str("CIN")

        # Read config
        yml_path = '/home/ec2-user/10799_hw2/methods/CIN/codes/options/opt.yml'
        option_yml = parse_yml(yml_path)

        # convert to NoneDict, which returns None for missing keys
        opt = dict_to_nonedict(option_yml)
        
        # cudnn
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # parallel
        os.environ["CUDA_VISIBLE_DEVICES"] = opt["train"]["os_environ"]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # path (log, experiment results and loss)
        time_now_NewExperiment = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
        if opt['subfolder'] != None:
            subfolder_name = opt['subfolder'] + '/-'
        else:
            subfolder_name = ''
        #
        folder_str = opt['path']['logs_folder'] + name + '/' + subfolder_name + str(time_now_NewExperiment) + '-' + opt['train/test']
        log_folder = folder_str + '/logs'
        img_w_folder_tra = folder_str  + '/img/train'
        img_w_folder_val = folder_str  + '/img/val'
        img_w_folder_test = folder_str + '/img/test'
        loss_w_folder = folder_str  + '/loss'
        path_checkpoint = folder_str  + '/path_checkpoint'
        opt_folder = folder_str  + '/opt'
        opt['path']['folder_temp'] = folder_str  + '/temp'
        #
        path_in = {'log_folder':log_folder, 'img_w_folder_tra':img_w_folder_tra, \
                        'img_w_folder_val':img_w_folder_val,'img_w_folder_test':img_w_folder_test,\
                            'loss_w_folder':loss_w_folder, 'path_checkpoint':path_checkpoint, \
                                'opt_folder':opt_folder, 'time_now_NewExperiment':time_now_NewExperiment}

        # create logger
        utils.mkdir(log_folder)
        # logging.basicConfig(level=logging.INFO,
        #                 format='%(message)s',
        #                 handlers=[
        #                     logging.FileHandler(os.path.join(log_folder, f'{name}-{time_now_NewExperiment}.log')),
        #                     logging.StreamHandler(sys.stdout)
        #                 ])
                        
        # log option_yml
        utils.mkdir(opt_folder)
        utils.setup_logger('base', opt_folder, 'train_' + name, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        # logger.info(dict2str(opt))
        
        # seed
        # logging.info('\nSeed = {}'.format(seed))
        
        # load data
        # train_data, val_data = utils.train_val_loaders(opt)      # from one folders (train only)

        # log datesets number
        # file_count_tr = len(train_data.dataset)
        # file_count_val = len(val_data.dataset)
        # logging.info('\nTrain_Img_num {} \nval_Img_num {}'.format(file_count_tr, file_count_val))
        
        # step config
        # total_epochs = opt['train']['epoch']
        # start_epoch = opt['train']['set_start_epoch']
        # start_step = opt['train']['start_step']

        # log BPP
        # Bpp = opt['network']['message_length'] / (opt['network']['H'] * opt['network']['W'] * opt['network']['input']['in_img_nc'])
        # logging.info('BPP = {:.4f}\n'.format(Bpp))

        # log
        # logging.info('\nStarting epoch {}\nstart_step {}'.format(start_epoch, start_step))
        # logging.info('Batch size = {}\n'.format(opt['train']['batch_size']))
        
        #
        network = Network(opt, device, path_in)
        return network.post_wm(image, message)

    def detect(self, img):

        name = str("CIN")

        # Read config
        yml_path = '/home/ec2-user/10799_hw2/methods/CIN/codes/options/opt.yml'
        option_yml = parse_yml(yml_path)

        # convert to NoneDict, which returns None for missing keys
        opt = dict_to_nonedict(option_yml)
        
        # cudnn
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # parallel
        os.environ["CUDA_VISIBLE_DEVICES"] = opt["train"]["os_environ"]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # path (log, experiment results and loss)
        time_now_NewExperiment = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
        if opt['subfolder'] != None:
            subfolder_name = opt['subfolder'] + '/-'
        else:
            subfolder_name = ''
        #
        folder_str = opt['path']['logs_folder'] + name + '/' + subfolder_name + str(time_now_NewExperiment) + '-' + opt['train/test']
        log_folder = folder_str + '/logs'
        img_w_folder_tra = folder_str  + '/img/train'
        img_w_folder_val = folder_str  + '/img/val'
        img_w_folder_test = folder_str + '/img/test'
        loss_w_folder = folder_str  + '/loss'
        path_checkpoint = folder_str  + '/path_checkpoint'
        opt_folder = folder_str  + '/opt'
        opt['path']['folder_temp'] = folder_str  + '/temp'
        #
        path_in = {'log_folder':log_folder, 'img_w_folder_tra':img_w_folder_tra, \
                        'img_w_folder_val':img_w_folder_val,'img_w_folder_test':img_w_folder_test,\
                            'loss_w_folder':loss_w_folder, 'path_checkpoint':path_checkpoint, \
                                'opt_folder':opt_folder, 'time_now_NewExperiment':time_now_NewExperiment}

        # create logger
        utils.mkdir(log_folder)
        # logging.basicConfig(level=logging.INFO,
        #                 format='%(message)s',
        #                 handlers=[
        #                     logging.FileHandler(os.path.join(log_folder, f'{name}-{time_now_NewExperiment}.log')),
        #                     logging.StreamHandler(sys.stdout)
        #                 ])
                        
        # log option_yml
        utils.mkdir(opt_folder)
        utils.setup_logger('base', opt_folder, 'train_' + name, level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        # logger.info(dict2str(opt))
        
        # seed
        # logging.info('\nSeed = {}'.format(seed))
        
        # load data
        # train_data, val_data = utils.train_val_loaders(opt)      # from one folders (train only)

        # log datesets number
        # file_count_tr = len(train_data.dataset)
        # file_count_val = len(val_data.dataset)
        # logging.info('\nTrain_Img_num {} \nval_Img_num {}'.format(file_count_tr, file_count_val))
        
        # step config
        # total_epochs = opt['train']['epoch']
        # start_epoch = opt['train']['set_start_epoch']
        # start_step = opt['train']['start_step']

        # log BPP
        # Bpp = opt['network']['message_length'] / (opt['network']['H'] * opt['network']['W'] * opt['network']['input']['in_img_nc'])
        # logging.info('BPP = {:.4f}\n'.format(Bpp))

        # log
        # logging.info('\nStarting epoch {}\nstart_step {}'.format(start_epoch, start_step))
        # logging.info('Batch size = {}\n'.format(opt['train']['batch_size']))
        
        #
        network = Network(opt, device, path_in)
        msg_detected = network.msg_detect(img)
        return msg_detected