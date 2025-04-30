#!/usr/bin/env bash

set -e
set -o pipefail

echo "Initializing project environment..."
echo ""

rm -rf venv
python3 -m venv "$(pwd)/venv"
source venv/bin/activate
pip3 install --require-virtualenv diffusers transformers accelerate
deactivate

echo ""
echo "Project environment setup complete."