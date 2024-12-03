# Source: https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/src/tree_ring_watermark/_get_noise.py

import torch
from typing import Union, List, Tuple
import numpy as np
import hashlib
import os
import tempfile
from huggingface_hub import hf_api
# from .utils import get_org

api = hf_api.HfApi()

def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2