import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from typing import Dict, Any, Tuple
import math
import torchvision

# list of available HF depth-anything models
possible_models = [
    "Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Depth-Anything-V2-Metric-Indoor-Large-hf",
    "Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth-Anything-V2-Metric-Outdoor-Large-hf",
]

# cache one pipeline per model
_PIPELINES: Dict[str, Any] = {}

class DepthEstimatorNode:
    """
    A ComfyUI node that runs depth estimation via HuggingFace's depth-anything models,
    returning only the raw metric-depth tensor.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"choices": possible_models, "default": possible_models[0]}),
                "depth_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Scale factor for depth values"
                }),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("depth tensor",)
    FUNCTION = "estimate_depth"
    CATEGORY = "Camera/depth"

    def estimate_depth(
        self,
        image: torch.Tensor,
        model_name: str,
        depth_scale: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        # lazy-load HuggingFace pipeline
        if model_name not in _PIPELINES:
            _PIPELINES[model_name] = pipeline(
                task="depth-estimation",
                model=f"depth-anything/{model_name}"
            )
        pipe = _PIPELINES[model_name]

        # BHWC float [1,H,W,3] -> uint8 HxW C
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # inference
        out = pipe(pil_img)
        pred_depth: torch.Tensor = out["predicted_depth"] * depth_scale  # [H,W]
        # ensure [1,H,W,1]
        pred_depth = pred_depth.unsqueeze(0).unsqueeze(-1)

        return (pred_depth,)


class DepthToImageNode:
    """
    A ComfyUI node that converts a single-channel depth tensor into a
    grayscale IMAGE for visualization, normalizing by min-max if vmin/vmax
    are not provided.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth": ("TENSOR",),
                "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Invert the depth map values"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth image",)
    FUNCTION = "depth_to_image"
    CATEGORY = "Camera/depth"

    def depth_to_image(
        self,
        depth: torch.Tensor,
        invert_depth: bool = False,

    ) -> Tuple[torch.Tensor]:
        # depth: [1, H, W, 1]
        d = depth.clone().detach().squeeze(0).squeeze(-1)  # [H, W]
        # find non-zero min/max
        d_min = d[d > 0].min()
        d_max = d[d > 0].max()
        # clamp to min/max
        d = torch.clamp(d, min=d_min, max=d_max)
        if invert_depth:
            d = 1.0/d # avoid div by zero
        # if either clamp bound is None, compute from tensor
        # normalize to 0–1 using vmin/vmax
        d_norm = d - d.min()
        d_norm = d_norm / (d_norm.max() - d_norm.min() + 1e-6)
        # replicate to RGB and batch dims: [1, H, W, 3]
        img = d_norm.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
        img = img.unsqueeze(0)
        return (img,)

class ZDepthToRayDepthNode:
    """
    A ComfyUI node that converts a single-channel depth tensor into
    a ray-depth tensor, taking into account the camera intrinsics.
    Supports a pinhole camera model; additional input is horizontal FOV.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth": ("TENSOR",),
                "fov": ("FLOAT", {
                    "default": 60.0, "min": 1.0, "max": 179.0, "step": 0.1,
                    "tooltip": "Horizontal field of view in degrees"
                }),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("ray depth",)
    FUNCTION = "depth_to_ray_depth"
    CATEGORY = "Camera/depth"

    def depth_to_ray_depth(
        self,
        depth: torch.Tensor,
        fov: float,
    ) -> Tuple[torch.Tensor]:
        # depth: [1, H, W, 1]  → squeeze out batch & channel dims → [H, W]
        d = depth.clone().detach().squeeze(0).squeeze(-1)
        H, W = d.shape
        device = d.device

        # Convert horizontal FOV to focal length (px)
        fov_rad = fov * math.pi / 180.0
        fx = W / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # assume square pixels

        # Build pixel coordinate grids
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

        # Principal point at image center
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # Normalized ray directions components
        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy

        # Per-pixel ray-length factor = ||[x, y, 1]||
        factor = torch.sqrt(1 + x**2 + y**2)

        # Ray-depth = metric depth (z) × ray-length factor
        ray_depth = d * factor  # → [H, W]

        # Restore batch & channel dims → [1, H, W, 1]
        ray_depth = ray_depth.unsqueeze(0).unsqueeze(-1)

        return (ray_depth,)



class CombineMode:
    MODES = ["SRC", "DST", "AVERAGE", "SOFTMERGE"]

class CombineDepthsNode:
    """
    A ComfyUI node that combines two single‐channel depth tensors according to two masks,
    either by averaging, hard overlay (SRC or DST priority), or soft‐merge (smooth blending),
    and returns both the blended depth and mask. All non‐spatial dims are flattened so that
    combine logic works purely on H×W maps.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depthSRC": ("TENSOR",),
                "maskSRC":  ("MASK",),
                "depthDST": ("TENSOR",),
                "maskDST":  ("MASK",),
                "mode":     (CombineMode.MODES, {
                    "default": "AVERAGE",
                    "tooltip": "AVERAGE = mean; SRC/DST = hard overlay; SOFTMERGE = smooth blend"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip masks (1→0, 0→1) before combining"
                }),
                "softmerge_radius": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Gaussian blur radius for SOFTMERGE mode"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "MASK")
    RETURN_NAMES = ("combined depth", "combined mask")
    FUNCTION = "combine_depths"
    CATEGORY = "Camera/depth"

    def combine_depths(
        self,
        depthSRC: torch.Tensor,
        maskSRC: torch.Tensor,
        depthDST: torch.Tensor,
        maskDST: torch.Tensor,
        mode: str,
        invert_mask: bool,
        softmerge_radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Validate input shapes
        #if depthSRC.shape != depthDST.shape or maskSRC.shape != maskDST.shape or maskSRC.shape != depthSRC.shape:
        #    raise ValueError("All inputs (depthSRC, depthDST, maskSRC, maskDST) must have the same shape")
        orig_shape = depthSRC.shape
        if len(orig_shape) < 2:
            raise ValueError("Input tensors must have at least 2 dimensions (H, W)")

        # Spatial dims
        H, W = orig_shape[-3], orig_shape[-2]
        # Flatten all leading dims (batch, channels, etc.) into one axis
        batch = orig_shape[0]
        # batch = int(np.prod(lead)) if lead else 1

        # Reshape to (batch, H, W)
        d0 = depthSRC.reshape(batch, H, W)
        d1 = depthDST.reshape(batch, H, W)
        m0 = maskSRC.reshape(batch, H, W)
        m1 = maskDST.reshape(batch, H, W)
        print("BHW are", batch, H, W)
        # Optionally invert masks
        if invert_mask:
            m0 = 1.0 - m0
            m1 = 1.0 - m1

        eps = 1e-6
        m0b = (m0 > 0.5).float()
        m1b = (m1 > 0.5).float()
        combined_m = ((m0b + m1b)>0.5).float()
        if mode == "AVERAGE":
            combined_d = d0*m0 + d1*m1
            combined_d = combined_d / (m0 + m1 + eps)

        elif mode in ("SRC", "DST"):  
            avg = 0.5 * (d0 + d1)
            combined_d = (
                m0b * d0 +
                m1b * d1 +
                (1 - m0b - m1b).clamp(min=0) * avg
            )
            overlap = m0b * m1b
            if mode == "SRC":
                combined_d = overlap * d0 + (1 - overlap) * combined_d
            else:
                combined_d = overlap * d1 + (1 - overlap) * combined_d

        else:  # SOFTMERGE
            # prepare for gaussian blur: reshape to (batch,1,H,W)
            m0v = m0.unsqueeze(1)
            m1v = m1.unsqueeze(1)
            k = 2 * softmerge_radius + 1
            sigma = float(softmerge_radius)
            # apply Gaussian blur
            m0_blur = torchvision.transforms.functional.gaussian_blur(m0v, (k, k), sigma=(sigma, sigma)).squeeze(1)
            m1_blur = torchvision.transforms.functional.gaussian_blur(m1v, (k, k), sigma=(sigma, sigma)).squeeze(1)
            # blur only overlapping regions
            mask_blur = ((m0_blur>0) * (m1_blur>0)).float()
            wsum = m0_blur*mask_blur+ m1_blur*mask_blur + (1-mask_blur)
            combined_d = (d0 * m0_blur + d1 * m1_blur)/ wsum
            average_d=(d0*m0 + d1*m1)/(m0 + m1 + (1-combined_m))*combined_m
            combined_d = combined_d*mask_blur + average_d*(1-mask_blur)
            # soft mask = normalized sum, clamped to [0,1]
        
        combined_d=combined_d*combined_m
        # Restore original leading dims
        combined_d = combined_d.reshape(batch, H, W)
        combined_m = combined_m.reshape(batch, H, W)
        # Return depth tensor (float32) and mask tensor (float)
        return combined_d.unsqueeze(-1), combined_m.unsqueeze(1)

NODE_CLASS_MAPPINGS = {
    "DepthEstimatorNode": DepthEstimatorNode,
    "DepthToImageNode": DepthToImageNode,
    "ZDepthToRayDepthNode": ZDepthToRayDepthNode,
    "CombineDepthsNode": CombineDepthsNode,
}