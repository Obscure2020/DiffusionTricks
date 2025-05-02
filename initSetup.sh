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

echo 'from diffusers import AutoPipelineForText2Image' > initSetupWorker.py
echo 'import torch' >> initSetupWorker.py
echo 'pipeline = []' >> initSetupWorker.py
echo 'if torch.cuda.is_available():' >> initSetupWorker.py
echo '    print("Pre-loading CUDA weights...")' >> initSetupWorker.py
echo '    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False)' >> initSetupWorker.py
echo 'else:' >> initSetupWorker.py
echo '    print("Pre-loading CPU weights...")' >> initSetupWorker.py
echo '    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, add_watermarker=False)' >> initSetupWorker.py
echo 'del pipeline' >> initSetupWorker.py
echo 'for i in range(6):' >> initSetupWorker.py
echo '    print()' >> initSetupWorker.py
python3 initSetupWorker.py
rm initSetupWorker.py

deactivate
echo "Project environment setup complete."