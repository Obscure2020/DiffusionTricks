from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import numpy as np
import oxipng
import io

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False)
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.enable_model_cpu_offload()

img = pipeline("Albert Einstein, impressionist painting, 8k", num_images_per_prompt=1).images[0]
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG", compress_level=9)
img_bytes = img_bytes.getvalue()
img_bytes_optim = oxipng.optimize_from_memory(img_bytes, level=6, force=True, filter=[oxipng.RowFilter.NoOp, oxipng.RowFilter.Sub, oxipng.RowFilter.Up, oxipng.RowFilter.Average, oxipng.RowFilter.Paeth, oxipng.RowFilter.Bigrams, oxipng.RowFilter.BigEnt, oxipng.RowFilter.Brute], interlace=oxipng.Interlacing.Off, optimize_alpha=True, strip=oxipng.StripChunks.safe(), deflate=oxipng.Deflaters.libdeflater(12))
with open("test_img_10.png", "wb") as f:
    f.write(img_bytes_optim)