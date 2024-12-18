# methods/output_pixel_watermarking.py

from methods.watermarked_diffusion_pipeline import BaseWatermarkedDiffusionPipeline
import numpy as np
from PIL import Image

class SmolOutputPixelWatermarking(BaseWatermarkedDiffusionPipeline):
    def generate(self, prompt: str, key: int = None, **generate_kwargs):
        # Generate image using the base pipeline
        image = super().generate(prompt, key=None, **generate_kwargs)
        if key is not None:
            # Embed the key into a specific pixel
            image = self.embed_watermark(image, key)
        return image
    
    def embed_key(self, r, g, b, key):
        # Embed the key into the last 3 bits of each color channel
        r = (r & 0xF8) | ((key & 0x04) >> 2)
        g = (g & 0xF8) | ((key & 0x02) >> 1)
        b = (b & 0xF8) | (key & 0x01)
        return r, g, b

    def embed_watermark(self, image: Image.Image, key: int) -> Image.Image:
        # Ensure the key is within 3-bit range
        if key < 0 or key > 7:
            raise ValueError("Key must be within the range 0 to 7 (3 bits).")

        # Embed the key into every pixel
        pixels = image.load()

        for x in range(image.width):
            for y in range(image.height):
                r, g, b = pixels[x, y]
                r, g, b = self.embed_key(r, g, b, key)
                pixels[x, y] = (r, g, b)
        
        return image

    def detect(self, image: Image.Image) -> int:
        # Extract the key from the specific pixel
        pixels = image.load()
        width, height = image.size
        keys_map = dict()
        # Randomly sample 50 pixels to extract the key
        for _ in range(50):
            # Randomly sample pixel coordinates to extract potential
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            r, g, b = pixels[x, y]
            
            # Extract the last 3 bits and combine them into a 3-bit key
            key = ((r & 0x07) << 2) | ((g & 0x03) << 1) | (b & 0x01)
            if key in keys_map:
                keys_map[key] += 1
            else:
                keys_map[key] = 1

        # Then take majority vote to determine the key
        most_freq_key = max(keys_map, key=keys_map.get)
        return most_freq_key