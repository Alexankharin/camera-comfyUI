import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from typing import Dict, Any, Tuple
import math
import torchvision
import scipy.ndimage as ndi

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
    Combines two depth maps + binary masks using:
      • AVERAGE: mean where either mask is true
      • SRC/DST: hard overlay
      • SOFTMERGE: smooth Gaussian-blended transition
    Outputs a float32 depth tensor [B,H,W,1] and a binary mask [B,H,W].
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depthSRC":         ("TENSOR",),
                "maskSRC":          ("MASK",),
                "depthDST":         ("TENSOR",),
                "maskDST":          ("MASK",),
                "mode":             (["SRC","DST","AVERAGE","SOFTMERGE"], {"default":"AVERAGE"}),
                "invert_mask":      ("BOOLEAN", {"default":False}),
                "softmerge_radius": ("INT", {"default":5, "min":1, "max":50}),
            }
        }

    RETURN_TYPES = ("TENSOR","MASK")
    RETURN_NAMES = ("combined_depth","combined_mask")
    FUNCTION = "combine_depths"
    CATEGORY = "Camera/depth"

    def combine_depths(
        self,
        depthSRC: torch.Tensor,
        maskSRC:  torch.Tensor,
        depthDST: torch.Tensor,
        maskDST:  torch.Tensor,
        mode: str,
        invert_mask: bool,
        softmerge_radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        B, H, W, _ = depthSRC.shape
        device = depthSRC.device
        eps = 1e-6

        # --- Flatten to [B,H,W] & binarize masks ---
        d0 = depthSRC.view(B, H, W)
        d1 = depthDST.view(B, H, W)
        m0 = maskSRC.view(B, H, W)
        m1 = maskDST.view(B, H, W)
        if invert_mask:
            m0 = 1.0 - m0
            m1 = 1.0 - m1

        m0b = (m0 > 0.5).float()
        m1b = (m1 > 0.5).float()
        combined_m = ((m0b + m1b) > 0).float()  # final binary mask

        # --- AVERAGE ---
        if mode == "AVERAGE":
            out = (d0*m0b + d1*m1b) / (m0b + m1b + eps)

        # --- SRC / DST (hard overlay) ---
        elif mode in ("SRC","DST"):
            avg = 0.5 * (d0 + d1)
            base = m0b*d0 + m1b*d1 + (1 - m0b - m1b).clamp(min=0)*avg
            overlap = m0b * m1b
            if mode == "SRC":
                out = overlap*d0 + (1 - overlap)*base
            else:
                out = overlap*d1 + (1 - overlap)*base

        # --- SOFTMERGE via separable Gaussian blur of masks ---
        else:
            # build 1D Gaussian kernel
            k = 2*softmerge_radius + 1
            sigma = float(softmerge_radius)
            coords = torch.arange(k, device=device, dtype=torch.float32) - softmerge_radius
            g1 = torch.exp(-0.5*(coords/sigma)**2)
            g1 /= g1.sum()
            # reshape for conv2d
            kh = g1.view(1,1,1,k)
            kv = g1.view(1,1,k,1)

            # prepare masks as [B,1,H,W]
            M0 = m0b.unsqueeze(1)
            M1 = m1b.unsqueeze(1)
            # horizontal blur
            B0 = F.conv2d(M0, kh, padding=(0, softmerge_radius))
            B1 = F.conv2d(M1, kh, padding=(0, softmerge_radius))
            # vertical blur
            B0 = F.conv2d(B0, kv, padding=(softmerge_radius, 0)).squeeze(1)
            B1 = F.conv2d(B1, kv, padding=(softmerge_radius, 0)).squeeze(1)

            # normalized weights & blend
            wsum = B0 + B1 + eps
            w0 = B0 / wsum
            w1 = B1 / wsum
            out = d0*w0 + d1*w1

        # --- Pack outputs ---
        combined_depth = out.unsqueeze(-1)     # [B,H,W,1]
        combined_mask  = combined_m            # [B,H,W], binary 0/1

        return combined_depth, combined_mask

class DepthRenormalizer:
    """
    Renormalize `depth` to match `guidance_depth` within the intersection
    of depth_mask and guidance_mask (optionally dilated & blurred), using
    a single global linear scale & offset.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth":           ("TENSOR",),
                "guidance_depth":  ("TENSOR",),
                "depth_mask":      ("MASK",),
                "guidance_mask":   ("MASK",),
                "use_inverse":     ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("depth tensor",)
    FUNCTION = "renormalize_depth"
    CATEGORY = "Camera/depth"

    def renormalize_depth(
        self,
        depth: torch.Tensor,
        guidance_depth: torch.Tensor,
        depth_mask: torch.Tensor,
        guidance_mask: torch.Tensor,
        use_inverse: bool = False
    ) -> Tuple[torch.Tensor]:
        # collapse [1,H,W,1] → [H,W]
        # mask is 11HW
        dm= depth_mask.squeeze(0).squeeze(0)
        gm= guidance_mask.squeeze(0).squeeze(0)
        def hw(t):
            t = t.squeeze(0).squeeze(-1)
            if t.dim() == 3:
                t = t.mean(dim=2)
            return t

        d  = hw(depth)
        gd = hw(guidance_depth)

        # 1) Intersection mask of valid depth & guidance
        mask = (dm > 0.5) & (gm > 0.5)

        # 2) (optional) dilate & blur that mask to soften edges
        kernel = torch.ones((1,1,3,3), device=d.device)
        m_dil = torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0)*1.0, kernel, padding=1) > 0
        m_smooth = torchvision.transforms.functional.gaussian_blur(
            m_dil.float(), (11,11), sigma=(5,5)
        ) > 0.5
        mask = m_smooth.squeeze(0).squeeze(0)

        eps = 1e-6
        # 3) choose working space
        if use_inverse:
            d_work  = 1.0 / d.clamp(min=eps)
            gd_work = 1.0 / gd.clamp(min=eps)
        else:
            d_work, gd_work = d, gd

        # 4) compute a single linear scale & offset over the masked region
        vals_d  = d_work[mask]
        vals_gd = gd_work[mask]
        if vals_d.numel() == 0:
            # nothing to renormalize
            out = d
        else:
            scale  = (vals_gd.std(unbiased=False) + eps) / (vals_d.std(unbiased=False) + eps)
            offset = vals_gd.mean() - vals_d.mean() * scale
            out = d * scale + offset

            if use_inverse:
                out = 1.0 / out.clamp(min=eps)

        return (out.unsqueeze(0).unsqueeze(-1),)

NODE_CLASS_MAPPINGS = {
    "DepthEstimatorNode": DepthEstimatorNode,
    "DepthToImageNode": DepthToImageNode,
    "ZDepthToRayDepthNode": ZDepthToRayDepthNode,
    "CombineDepthsNode": CombineDepthsNode,
    "DepthRenormalizer": DepthRenormalizer,
}