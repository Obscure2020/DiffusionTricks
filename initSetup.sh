#!/usr/bin/env bash

set -e
set -o pipefail

echo "Initializing project environment..."
echo ""

rm -rf venv
python3 -m venv "$(pwd)/venv"
source venv/bin/activate
pip3 install --require-virtualenv diffusers transformers accelerate pyoxipng
echo ""

echo 'from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline' > initSetupWorker.py
echo 'import torch' >> initSetupWorker.py
echo 'pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False)' >> initSetupWorker.py
echo 'del pipeline' >> initSetupWorker.py
echo 'for i in range(6):' >> initSetupWorker.py
echo '    print()' >> initSetupWorker.py
echo 'refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", add_watermarker=False)' >> initSetupWorker.py
echo 'del refiner' >> initSetupWorker.py
echo 'for i in range(6):' >> initSetupWorker.py
echo '    print()' >> initSetupWorker.py
python3 initSetupWorker.py
rm initSetupWorker.py

deactivate
echo "Project environment setup complete."