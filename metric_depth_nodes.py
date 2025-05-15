import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from typing import Dict, Any, Tuple
import math
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


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
    Runs depth estimation via HuggingFace depth-anything models,
    returning a metric-depth tensor with optional median blur.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"choices": possible_models, "default": possible_models[0]}),
                "depth_scale": ("FLOAT", {"default":1.0, "min":0.0, "max":100.0, "step":0.01}),
                "median_blur_kernel": ("INT", {"default":1, "min":1, "max":99, "step":2, "tooltip":"Odd kernel size for depth median blur"}),
            }
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
        median_blur_kernel: int = 1,
    ) -> Tuple[torch.Tensor]:
        # Lazy-load HF pipeline
        if model_name not in _PIPELINES:
            _PIPELINES[model_name] = pipeline(
                task="depth-estimation",
                model=f"depth-anything/{model_name}"
            )
        pipe = _PIPELINES[model_name]

        # Convert BHWC [1,H,W,3] float to HxW uint8
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Inference
        out = pipe(pil_img)
        pred = out["predicted_depth"] * depth_scale  # numpy array or torch?
        if isinstance(pred, np.ndarray):
            depth_map = torch.from_numpy(pred)
        else:
            depth_map = pred
        depth_map = depth_map.to(dtype=torch.float32, device=image.device)

        # Depth_map is [H,W]; add batch & channel dims: [1,1,H,W]
        depth = depth_map.unsqueeze(0).unsqueeze(0)

        # Median blur if kernel > 1
        k = median_blur_kernel
        if k > 1:
            pad = k // 2
            # pad and unfold for median
            padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')
            # shape [1,1,H+k-1, W+k-1]
            patches = padded.unfold(2, k, 1).unfold(3, k, 1)
            # [1,1,H,W,k,k]
            patches = patches.contiguous().view(1,1,depth.shape[2], depth.shape[3], k*k)
            depth, _ = patches.median(dim=-1)

        # Final shape [1,H,W,1]
        depth = depth.permute(0,2,3,1)
        return (depth,)


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
        d = depth.squeeze(0).squeeze(-1)  # [H, W]
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
      • SOFTMERGE: Gaussian‐blurred transition
      • DISTANCE_AWARE: disparity‐space, distance‐transform blending
    Outputs a float32 depth tensor [B,H,W,1] and a binary mask [B,H,W].
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str,Any]:
        return {
            "required": {
                "depthSRC":         ("TENSOR",),
                "maskSRC":          ("MASK",),
                "depthDST":         ("TENSOR",),
                "maskDST":          ("MASK",),
                "mode":             (["SRC","DST","AVERAGE","SOFTMERGE","DISTANCE_AWARE"], {"default":"AVERAGE"}),
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
        B, H, W, _ = depthSRC.shape
        device = depthSRC.device
        eps = 1e-6

        # flatten depth & masks to [B,H,W]
        d0 = depthSRC.view(B,H,W)
        d1 = depthDST.view(B,H,W)
        m0 = maskSRC.view(B,H,W)
        m1 = maskDST.view(B,H,W)
        if invert_mask:
            m0 = 1.0 - m0
            m1 = 1.0 - m1
        m0b = (m0 > 0.5).float()
        m1b = (m1 > 0.5).float()
        combined_m = ((m0b + m1b) > 0).float()

        # AVERAGE
        if mode == "AVERAGE":
            out = (d0*m0b + d1*m1b) / (m0b + m1b + eps)

        # SRC/DST hard overlay
        elif mode in ("SRC","DST"):
            avg = 0.5*(d0 + d1)
            base = m0b*d0 + m1b*d1 + (1 - m0b - m1b).clamp(min=0)*avg
            overlap = m0b * m1b
            if mode == "SRC":
                out = overlap*d0 + (1 - overlap)*base
            else:
                out = overlap*d1 + (1 - overlap)*base

        # SOFTMERGE (Gaussian mask blend)
        elif mode == "SOFTMERGE":
            M0 = m0b.unsqueeze(1)
            M1 = m1b.unsqueeze(1)
            k = 2*softmerge_radius + 1
            sigma = float(softmerge_radius)
            coords = torch.arange(k, device=device, dtype=torch.float32) - softmerge_radius
            g1 = torch.exp(-0.5*(coords/sigma)**2)
            g1 /= g1.sum()
            g_h = g1.view(1,1,1,k)
            g_v = g1.view(1,1,k,1)
            # separable blur
            B0 = F.conv2d(F.conv2d(M0, g_h, padding=(0,softmerge_radius)), g_v, padding=(softmerge_radius,0))
            B1 = F.conv2d(F.conv2d(M1, g_h, padding=(0,softmerge_radius)), g_v, padding=(softmerge_radius,0))
            denom = B0 + B1 + eps
            w0 = (B0/denom) * M0
            w1 = (B1/denom) * M1
            out = (d0.unsqueeze(1)*w0 + d1.unsqueeze(1)*w1).squeeze(1)

        # DISTANCE_AWARE (disparity + distance-transform)
        else:  # mode == "DISTANCE_AWARE"
            # compute binary masks
            m0_bin = m0b.cpu().numpy().astype(np.uint8)
            m1_bin = m1b.cpu().numpy().astype(np.uint8)

            # disparity (inverse depth)
            #disp0 = 1.0/(d0 + eps)
            #disp1 = 1.0/(d1 + eps)

            # distance transform per batch
            D0 = torch.zeros((B,H,W), device=device)
            D1 = torch.zeros((B,H,W), device=device)
            for b in range(B):
                D0_b = distance_transform_edt(m0_bin[b])
                D1_b = distance_transform_edt(m1_bin[b])
                D0[b] = torch.from_numpy(D0_b).to(device)
                D1[b] = torch.from_numpy(D1_b).to(device)

            # build weights
            overlap = (m0b * m1b) > 0
            w0 = torch.zeros_like(d0)
            w1 = torch.zeros_like(d0)

            # in overlap region: distance-based ratio
            denomD = D0 + D1 + eps
            w0_o = D0 / denomD
            w1_o = D1 / denomD
            w0[overlap] = w0_o[overlap]
            w1[overlap] = w1_o[overlap]

            # in src-only region: full src weight
            mask0_only = (m0b > 0) & (m1b == 0)
            w0[mask0_only] = 1.0

            # in dst-only region: full dst weight
            mask1_only = (m1b > 0) & (m0b == 0)
            w1[mask1_only] = 1.0

            # blend in disparity space, then invert back
            #disp_blend = w0 * disp0 + w1 * disp1
            #out = 1.0/(disp_blend + eps)
            d_blend= w0 * d0 + w1 * d1
            out = d_blend
        # pack outputs
        combined_depth = out.unsqueeze(-1)      # [B,H,W,1]
        return combined_depth, combined_m

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