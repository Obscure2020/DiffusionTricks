from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import oxipng
import io

prompt = "Albert Einstein, impressionist painting, 8k"

pipeline = []
if torch.cuda.is_available():
    print("Initializing CUDA pipeline...")
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False)
    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.enable_model_cpu_offload() # This last line contains a hidden, implicit default that specifies that CUDA should be used.
else:
    print("Initializing CPU pipeline...")
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, add_watermarker=False).to("cpu")
    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

img = pipeline(prompt, num_images_per_prompt=1, num_inference_steps=10, guidance_scale=0.0).images[0]
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG", compress_level=9)
img_bytes = img_bytes.getvalue()
img_bytes_optim = oxipng.optimize_from_memory(img_bytes, level=6, force=True, filter=[oxipng.RowFilter.NoOp, oxipng.RowFilter.Sub, oxipng.RowFilter.Up, oxipng.RowFilter.Average, oxipng.RowFilter.Paeth, oxipng.RowFilter.Bigrams, oxipng.RowFilter.BigEnt, oxipng.RowFilter.Brute], interlace=oxipng.Interlacing.Off, optimize_alpha=True, strip=oxipng.StripChunks.safe(), deflate=oxipng.Deflaters.libdeflater(12))
with open("test_img_13_CUDA.png", "wb") as f:
    f.write(img_bytes_optim)