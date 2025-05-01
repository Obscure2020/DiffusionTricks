from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import numpy as np

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False)
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.enable_model_cpu_offload()

img = pipeline("Albert Einstein, impressionist painting, 8k", num_images_per_prompt=1).images[0]
img.save("test_img_5.png", compress_level=1)