from diffusers import AutoPipelineForImage2Image
import torch
from PIL import Image
import numpy as np
import oxipng
import io
import gc

def img_check(arr_obj):
    assert type(arr_obj) is np.ndarray, "arr_obj is not an ndarray."
    assert len(arr_obj.shape)==3, "arr_obj is not a 3D structure."
    assert arr_obj.shape[2]==3, "arr_obj does not appear to contain 3 channels."

def write_PNG_from_ndarray(filename, arr_obj):
    img_check(arr_obj)
    assert arr_obj.dtype.kind=='f', "arr_obj does not contain floating point data."
    minCheck = arr_obj.min() >= 0.0
    maxCheck = arr_obj.max() <= 1.0
    assert minCheck and maxCheck, "arr_obj is not normalized within the 0.0 to 1.0 range."
    print(f"Exporting \"{filename}\"")
    img = Image.fromarray((arr_obj * 255).astype(np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG", compress_level=9)
    img_bytes = img_bytes.getvalue()
    img_bytes_optim = oxipng.optimize_from_memory(img_bytes, level=6, force=True, filter=[oxipng.RowFilter.NoOp, oxipng.RowFilter.Sub, oxipng.RowFilter.Up, oxipng.RowFilter.Average, oxipng.RowFilter.Paeth, oxipng.RowFilter.Bigrams, oxipng.RowFilter.BigEnt, oxipng.RowFilter.Brute], interlace=oxipng.Interlacing.Off, optimize_alpha=True, strip=oxipng.StripChunks.safe(), deflate=oxipng.Deflaters.zopfli(15))
    with open(filename, "wb") as f:
        f.write(img_bytes_optim)

def transform_img_180(arr_obj):
    img_check(arr_obj)
    return np.rot90(arr_obj, k=2, axes=(0, 1))

transform_img_180_undo = transform_img_180

def transform_img_halve(arr_obj):
    img_check(arr_obj)
    axis = 1
    roll_amount = int(arr_obj.shape[axis] / 2)
    return np.roll(arr_obj, roll_amount, axis)

def transform_img_halve_undo(arr_obj):
    img_check(arr_obj)
    axis = 1
    roll_amount = -1 * (int(arr_obj.shape[axis] / 2))
    return np.roll(arr_obj, roll_amount, axis)

prompt_a = "Albert Einstein, impressionist painting, 8k"
prompt_b = "Marilyn Monroe, impressionist painting, 8k"
strength_schedule = [0.98, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
steps_schedule = [4, 4, 4, 5, 5, 6, 7, 8, 9, 10]

pipeline = []
if torch.cuda.is_available():
    print("Initializing CUDA pipeline...")
    pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to("cuda")
    pipeline.unet.to(memory_format=torch.channels_last)
    #pipeline.enable_model_cpu_offload() # This line contains a hidden, implicit default that specifies that CUDA should be used.
else:
    print("Initializing CPU pipeline...")
    pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, add_watermarker=False).to("cpu")
    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# The main event: two-prompt generation!
img_a = np.random.random_sample((512,512,3)).astype(np.float32)
img_b = np.random.random_sample((512,512,3)).astype(np.float32)
for i in range(len(steps_schedule)):
    img_a = pipeline(prompt_a, image=img_a, num_images_per_prompt=1, num_inference_steps=steps_schedule[i], strength=strength_schedule[i], guidance_scale=0.0, output_type="np").images[0]
    img_b = pipeline(prompt_b, image=img_b, num_images_per_prompt=1, num_inference_steps=steps_schedule[i], strength=strength_schedule[i], guidance_scale=0.0, output_type="np").images[0]

# Clean up pipeline memory after we're done using it
del pipeline
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

write_PNG_from_ndarray("test_img_a.png", img_a)
write_PNG_from_ndarray("test_img_b.png", img_b)

print("Work complete. Cleaning up and exiting...")