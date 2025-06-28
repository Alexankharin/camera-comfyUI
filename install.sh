#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Functions
# ----------------------------------------
install_pytorch() {
  echo "==> Installing PyTorch, TorchVision, TorchAudio, bitsandbytes, accelerate…"
  pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip3 install -U bitsandbytes accelerate
}

install_system_deps() {
  echo "==> Updating apt and installing system packages…"
  sudo apt-get update
  sudo apt-get install -y build-essential ffmpeg libsm6 libxext6 python3.10-dev
}

clone_and_install_comfyui() {
  echo "==> Cloning ComfyUI…"
  git clone https://github.com/comfyanonymous/ComfyUI.git
  echo "==> Installing ComfyUI Python requirements…"
  pip3 install -r ComfyUI/requirements.txt
}

install_camera_node() {
  echo "==> Installing camera‑ComfyUI node…"
  mkdir -p ComfyUI/custom_nodes
  git clone https://github.com/Alexankharin/camera-comfyUI.git \
    ComfyUI/custom_nodes/camera-comfyUI
  pip3 install -r ComfyUI/custom_nodes/camera-comfyUI/requirements.txt
}

install_image_filters() {
  echo "==> Installing Image‑Filters node…"
  mkdir -p ComfyUI/custom_nodes
  git clone https://github.com/spacepxl/ComfyUI-Image-Filters.git \
    ComfyUI/custom_nodes/ComfyUI-Image-Filters
  pip3 install -r ComfyUI/custom_nodes/ComfyUI-Image-Filters/requirements.txt
}

clone_flux_inpainting() {
  echo "==> Installing Flux‑Inpainting node…"
  mkdir -p ComfyUI/custom_nodes
  git clone https://github.com/rubi-du/ComfyUI-Flux-Inpainting.git \
    ComfyUI/custom_nodes/inpainting_flux
}

install_metric_video_depth_anything() {
  echo "==> Installing Metric Video Depth Anything…"
  # Clone into ComfyUI root, not custom_nodes
  git clone https://github.com/DepthAnything/Video-Depth-Anything.git \
    ComfyUI/Video-Depth-Anything

  echo "    • Installing easydict…"
  pip3 install -U easydict

  echo "    • Copying util.py…"
  mkdir -p ComfyUI/utils
  cp ComfyUI/Video-Depth-Anything/metric_depth/utils/util.py \
     ComfyUI/utils/util.py

  echo "    • Downloading Metric Video Depth checkpoint…"
  mkdir -p ComfyUI/models/checkpoints
  wget -q -O ComfyUI/models/checkpoints/metric_video_depth_anything_vitl.pth \
    "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth"
}


install_comfyui_manager() {
  echo "==> Installing ComfyUI-Manager extension…"
  mkdir -p ComfyUI/custom_nodes
  git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
    ComfyUI/custom_nodes/ComfyUI-Manager
  pip3 install -r ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt
}

install_hf_hub() {
  echo "==> Installing huggingface_hub…"
  pip3 install -U huggingface_hub
}

download_vae_models() {
  echo "==> Downloading WAN‑VACE models…"
  mkdir -p ComfyUI/models/vae
  wget -q -O ComfyUI/models/vae/wan_2.1_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true"

  mkdir -p ComfyUI/models/text_encoders
  wget -q -O ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"

  mkdir -p ComfyUI/models/diffusion_models
  wget -q -O ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"
}

login_hf_hub() {
  echo "==> Hugging Face login…"
  huggingface-cli login
}

# ----------------------------------------
# Main
# ----------------------------------------
MODE="${1:-install}"

case "$MODE" in
  modules)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    install_image_filters
    install_comfyui_manager
    install_hf_hub
    ;;

  flux)
    clone_flux_inpainting
    ;;

  vae)
    download_vae_models
    ;;

  depth)
    install_metric_video_depth_anything
    ;;

  install)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    clone_flux_inpainting
    install_image_filters
    install_comfyui_manager
    install_hf_hub
    install_metric_video_depth_anything
    ;;

  all)
    install_pytorch
    install_system_deps
    clone_and_install_comfyui
    install_camera_node
    clone_flux_inpainting
    install_image_filters
    install_comfyui_manager
    install_hf_hub
    download_vae_models
    install_metric_video_depth_anything
    login_hf_hub
    echo "✅ All done!"
    ;;

  *)
    echo "Usage: $0 {install|modules|flux|vae|depth|all}"
    exit 1
    ;;
esac
