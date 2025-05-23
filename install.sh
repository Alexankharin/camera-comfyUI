#!/usr/bin/env bash
set -euo pipefail

# 1. Install PyTorch with CUDA 12.8 wheels
echo "Installing PyTorch, TorchVision, TorchAudio..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Update apt repositories and install system dependencies
echo "Updating apt and installing build-essential, ffmpeg, libsm6, libxext6..."
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg libsm6 libxext6

# 3. Clone ComfyUI and install its Python requirements
echo "Cloning ComfyUI..."
git clone https://github.com/comfyanonymous/ComfyUI.git
echo "Installing ComfyUI requirements..."
pip3 install -r ComfyUI/requirements.txt

# 4. Enter the custom_nodes folder
cd ComfyUI/custom_nodes

# 5. camera-comfyUI
echo "Cloning camera-comfyUI..."
git clone https://github.com/Alexankharin/camera-comfyUI.git
echo "Installing camera-comfyUI requirements..."
pip3 install camera-comfyUI/requirements.txt

# 6. ComfyUI-Flux-Inpainting
echo "Cloning ComfyUI-Flux-Inpainting..."
git clone https://github.com/rubi-du/ComfyUI-Flux-Inpainting.git
echo "Installing ComfyUI-Flux-Inpainting requirements..."
pip3 install ComfyUI-Flux-Inpainting/requirements.txt

# 7. ComfyUI-Image-Filters
echo "Cloning ComfyUI-Image-Filters..."
git clone https://github.com/spacepxl/ComfyUI-Image-Filters.git
echo "Installing ComfyUI-Image-Filters requirements..."
pip3 install ComfyUI-Image-Filters/requirements.txt

# 8. Tidy up Flux Inpainting folder name
echo "Renaming Flux Inpainting folder..."
cd ..
mv custom_nodes/ComfyUI-Flux-Inpainting-main custom_nodes/inpainting_flux

# 9. Install Hugging Face Hub Python package
echo "Installing huggingface_hub..."
pip3 install huggingface_hub

echo "All done! 🎉"
