#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Functions
# ----------------------------------------
install_pytorch() {
  echo "Installing PyTorch, TorchVision, TorchAudio..."
  pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

install_system_deps() {
  echo "Updating apt and installing system dependencies..."
  sudo apt-get update
  sudo apt-get install -y build-essential ffmpeg libsm6 libxext6
}

clone_and_install_comfyui() {
  echo "Cloning ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI.git
  echo "Installing ComfyUI requirements..."
  pip3 install -r ComfyUI/requirements.txt
}

install_camera_node() {
  echo "Cloning camera‑ComfyUI..."
  git clone https://github.com/Alexankharin/camera-comfyUI.git \
    ComfyUI/custom_nodes/camera-comfyUI
  echo "Installing camera‑ComfyUI requirements..."
  pip3 install -r ComfyUI/custom_nodes/camera-comfyUI/requirements.txt
}

install_image_filters() {
  echo "Cloning Image‑Filters node..."
  git clone https://github.com/spacepxl/ComfyUI-Image-Filters.git \
    ComfyUI/custom_nodes/ComfyUI-Image-Filters
  echo "Installing Image‑Filters requirements..."
  pip3 install -r ComfyUI/custom_nodes/ComfyUI-Image-Filters/requirements.txt
}

clone_flux_inpainting() {
  echo "Cloning ComfyUI‑Flux‑Inpainting..."
  git clone https://github.com/rubi-du/ComfyUI-Flux-Inpainting.git \
    ComfyUI/custom_nodes/Flux-Inpainting
  echo "Renaming Flux‑Inpainting folder..."
  mv ComfyUI/custom_nodes/Flux-Inpainting \
     ComfyUI/custom_nodes/inpainting_flux
}

install_hf_hub() {
  echo "Installing huggingface_hub..."
  pip3 install huggingface_hub
}

download_vae_models() {
  echo "Downloading WAN‑VACE models via wget..."
<<<<<<< HEAD
  wget -O models/vae/wan_2.1_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true"

  wget -O models/text_encoders/umt5_xxl_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"

  wget -O models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors?download=true"
=======
  mkdir -p models/diffusion_models models/text_encoders models/vae

  wget -O models/vae/wan_2.1_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true"

  wget -O models/text_encoders/umt5_xxl_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"

  wget -O models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors?download=true"
>>>>>>> d604ec8646f778a5ef340034855b34e24ae91fd7
}

login_hf_hub() {
  echo "Logging in to Hugging Face Hub..."
  huggingface-cli login
}

# ----------------------------------------
# Main CLI
# ----------------------------------------
# default to "install" if no arg given
MODE="${1:-install}"

case "$MODE" in
  modules)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    install_image_filters
    install_hf_hub
    ;;

  flux)
    clone_flux_inpainting
    ;;

  vae)
    download_vae_models
    ;;

  install)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    clone_flux_inpainting
    install_image_filters
    install_hf_hub
    ;;

  all)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    clone_flux_inpainting
    install_image_filters
    install_hf_hub
    download_vae_models
    login_hf_hub
    echo "All done! 🎉"
    ;;

  *)
    echo "Usage: $0 {install|modules|flux|vae|all}"
    exit 1
    ;;
esac
