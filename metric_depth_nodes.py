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
    CATEGORY = "Camera/Depth"

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
                model=f"depth-anything/{model_name}",
                device=0 if torch.cuda.is_available() else -1,
            )
        pipe = _PIPELINES[model_name]

        # Convert BHWC [B,H,W,3] float batch to PIL images
        imgs_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(a) for a in imgs_np]

        # Inference — the pipeline accepts a list; run the whole batch
        outs = pipe(pil_imgs, batch_size=8)
        if isinstance(outs, dict):  # single-image call returns a dict
            outs = [outs]
        preds = []
        for out in outs:
            pred = out["predicted_depth"] * depth_scale  # numpy array or torch?
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            preds.append(pred)
        depth_map = torch.stack(preds).to(dtype=torch.float32, device=image.device)

        # Depth_map is [B,H,W]; add channel dim: [B,1,H,W]
        depth = depth_map.unsqueeze(1)

        # Median blur if kernel > 1
        k = median_blur_kernel
        if k > 1:
            pad = k // 2
            # pad and unfold for median
            padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')
            # shape [1,1,H+k-1, W+k-1]
            patches = padded.unfold(2, k, 1).unfold(3, k, 1)
            # [B,1,H,W,k,k]
            patches = patches.contiguous().view(depth.shape[0], 1, depth.shape[2], depth.shape[3], k*k)
            depth, _ = patches.median(dim=-1)

        # Final shape [B,H,W,1]
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
    CATEGORY = "Camera/Depth"

    def depth_to_image(
        self,
        depth: torch.Tensor,
        invert_depth: bool = False,

    ) -> Tuple[torch.Tensor]:
        # depth: [B, H, W, 1] / [B, H, W] / [H, W] → [B, H, W]
        d = depth
        if d.dim() == 4:
            d = d.squeeze(-1)
        elif d.dim() == 2:
            d = d.unsqueeze(0)
        # find non-zero min/max over the whole batch so a sequence normalizes
        # consistently (no per-frame flicker)
        d_min = d[d > 0].min()
        d_max = d[d > 0].max()
        # clamp to min/max
        d = torch.clamp(d, min=d_min, max=d_max)
        if invert_depth:
            d = 1.0/d # avoid div by zero
        # normalize to 0–1
        d_norm = d - d.min()
        d_norm = d_norm / (d_norm.max() - d_norm.min() + 1e-6)
        # replicate to RGB: [B, H, W, 3]
        img = d_norm.unsqueeze(-1).repeat(1, 1, 1, 3)
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
    CATEGORY = "Camera/Depth"

    def depth_to_ray_depth(
        self,
        depth: torch.Tensor,
        fov: float,
    ) -> Tuple[torch.Tensor]:
        # depth: [B, H, W, 1] (or [H, W]) → drop channel dim → [B, H, W]
        d = depth.clone().detach()
        if d.dim() == 4:
            d = d.squeeze(-1)
        elif d.dim() == 2:
            d = d.unsqueeze(0)
        B, H, W = d.shape
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

        # Ray-depth = metric depth (z) × ray-length factor (broadcast over batch)
        ray_depth = d * factor  # → [B, H, W]

        # Restore channel dim → [B, H, W, 1]
        ray_depth = ray_depth.unsqueeze(-1)

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
    CATEGORY = "Camera/Depth"

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
    CATEGORY = "Camera/Depth"

    def renormalize_depth(
        self,
        depth: torch.Tensor,
        guidance_depth: torch.Tensor,
        depth_mask: torch.Tensor,
        guidance_mask: torch.Tensor,
        use_inverse: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Fit a centrally‐symmetric affine transform
            out = A(x,y) * d + B(x,y),
        where
            A(x,y) = a0 + a1*(x-0.5)**2 + a2*(y-0.5)**2
            B(x,y) = b0 + b1*(x-0.5)**2 + b2*(y-0.5)**2
        so that it best matches guidance_depth over valid pixels,
        then apply it everywhere (including holes).

        Batch-aware: inputs may be [B,H,W,1] / [B,H,W] / [H,W]; each frame is
        fitted independently and the result is returned as [B,H,W,1].
        """
        # normalize any of [B,H,W,C] / [B,H,W] / [H,W] → [B,H,W]
        def bhw(t):
            if t.dim() == 4:
                t = t.squeeze(-1) if t.shape[-1] == 1 else t.mean(dim=-1)
            elif t.dim() == 2:
                t = t.unsqueeze(0)
            return t

        d_all  = bhw(depth)           # [B,H,W]
        gd_all = bhw(guidance_depth)  # [B,H,W]
        dm_all = bhw(depth_mask)  > 0.5
        gm_all = bhw(guidance_mask) > 0.5
        B = d_all.shape[0]
        # broadcast singleton batches (e.g. one guidance frame for a video)
        if gd_all.shape[0] == 1 and B > 1:
            gd_all = gd_all.expand(B, -1, -1)
        if dm_all.shape[0] == 1 and B > 1:
            dm_all = dm_all.expand(B, -1, -1)
        if gm_all.shape[0] == 1 and B > 1:
            gm_all = gm_all.expand(B, -1, -1)

        outs = []
        for b in range(B):
            outs.append(self._renormalize_single(
                d_all[b], gd_all[b], dm_all[b], gm_all[b], use_inverse))
        out = torch.stack(outs)  # [B,H,W]
        # restore [B,H,W,1]
        return (out.unsqueeze(-1),)

    def _renormalize_single(
        self,
        d: torch.Tensor,       # [H,W]
        gd: torch.Tensor,      # [H,W]
        dm: torch.Tensor,      # [H,W] bool
        gm: torch.Tensor,      # [H,W] bool
        use_inverse: bool,
    ) -> torch.Tensor:

        # choose linear or inverse-depth space
        eps = 1e-6
        if use_inverse:
            d_work  = 1.0 / d.clamp(min=eps)
            gd_work = 1.0 / gd.clamp(min=eps)
        else:
            d_work, gd_work = d, gd

        # valid pixels for fitting
        valid = dm & (d_work > 0) & (gd_work > 0)
        # remove nan and inf from valid mask
        valid = valid & (~torch.isnan(d_work)) & (~torch.isinf(d_work))
        # erode valid mask by 2 pixels
        valid = F.max_pool2d(-valid.float().unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0) < -0.5
        if valid.sum() < 6:
            # fallback to global linear if too few
            vals_d  = d_work[valid]
            vals_gd = gd_work[valid]
            scale   = (vals_gd.std(unbiased=False)+eps)/(vals_d.std(unbiased=False)+eps)
            offset  = vals_gd.mean() - scale*vals_d.mean()
            out_work = d_work*scale + offset

        else:
            H, W = d_work.shape
            ys = torch.arange(H, device=d_work.device, dtype=d_work.dtype)
            xs = torch.arange(W, device=d_work.device, dtype=d_work.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')

            # normalized in [0,1], then center at 0
            xi = (xx / (W-1)) - 0.5
            yi = (yy / (H-1)) - 0.5
            xi2 = xi**2
            yi2 = yi**2

            # pick out valid pixels
            dv  = d_work[valid]       # [N]
            gv  = gd_work[valid]      # [N]
            xiv = xi[valid]           # [N]
            yiv = yi[valid]           # [N]
            xiv2= xi2[valid]          # [N]
            yiv2= yi2[valid]          # [N]

            # design matrix: columns for [dv, dv*x, dv*y, dv*x2, dv*y2, 1, x, y, x2, y2]
            X = torch.stack([
                dv,
                dv * xiv,
                dv * yiv,
                dv * xiv2,
                dv * yiv2,
                torch.ones_like(dv),
                xiv,
                yiv,
                xiv2,
                yiv2
            ], dim=1)  # -> [N,10]
            y = gv.unsqueeze(1)        # [N,1]

            # solve least-squares for 10 coefficients
            sol = torch.linalg.lstsq(X, y).solution.squeeze(1)
            a0, a1, a2, a3, a4, b0, b1, b2, b3, b4 = sol.unbind(0)

            # build full A_map, B_map
            A_map = a0 + a1*xi + a2*yi + a3*xi2 + a4*yi2
            B_map = b0 + b1*xi + b2*yi + b3*xi2 + b4*yi2

            # apply everywhere
            out_work = A_map * d_work + B_map

        # convert back if inverse-depth
        return (1.0 / out_work.clamp(min=eps)) if use_inverse else out_work
    

NODE_CLASS_MAPPINGS = {
    "DepthEstimatorNode": DepthEstimatorNode,
    "DepthToImageNode": DepthToImageNode,
    "ZDepthToRayDepthNode": ZDepthToRayDepthNode,
    "CombineDepthsNode": CombineDepthsNode,
    "DepthRenormalizer": DepthRenormalizer,
}