import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
import math

ZBUFFER_RESOLUTION = 1024

class Projection:
    """
    A class to define supported projection types.
    """
    PROJECTIONS = ["PINHOLE", "FISHEYE", "EQUIRECTANGULAR"]

# ==== Depth to pointcloud conversion functions ==== #
def pinhole_depth_to_XYZ(depth: torch.Tensor, fov: float):
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2)
    H, W = depth.shape
    u = torch.linspace(-1.0, 1.0, W, device=depth.device).unsqueeze(0).expand(H, W)
    v = torch.linspace(-1.0, 1.0, H, device=depth.device).unsqueeze(1).expand(H, W)
    Ruv = torch.sqrt(u**2 + v**2)
    theta = torch.atan(Ruv / f)
    phi   = torch.atan2(v, u)
    X = depth * torch.sin(theta) * torch.cos(phi)
    Y = depth * torch.sin(theta) * torch.sin(phi)
    Z = depth * torch.cos(theta)
    return X, Y, Z

def fisheye_depth_to_XYZ(depth: torch.Tensor, fov: float):
    # equidistant fisheye: θ = Ruv * (fov/2), Ruv∈[-1,1]
    fov_rad = math.radians(fov)
    H, W = depth.shape
    u = torch.linspace(-1.0, 1.0, W, device=depth.device).unsqueeze(0).expand(H, W)
    v = torch.linspace(-1.0, 1.0, H, device=depth.device).unsqueeze(1).expand(H, W)
    Ruv = torch.sqrt(u**2 + v**2).clamp(max=1.0)
    theta = Ruv * (fov_rad / 2)
    phi   = torch.atan2(v, u)
    X = depth * torch.sin(theta) * torch.cos(phi)
    Y = depth * torch.sin(theta) * torch.sin(phi)
    Z = depth * torch.cos(theta)
    return X, Y, Z

def equirect_depth_to_XYZ(depth: torch.Tensor, *_):
    # full 360°×180° equirectangular
    H, W = depth.shape
    lon = torch.linspace(-math.pi, math.pi, W, device=depth.device).unsqueeze(0).expand(H, W)
    lat = torch.linspace( math.pi/2, -math.pi/2, H, device=depth.device).unsqueeze(1).expand(H, W)
    X = depth * torch.cos(lat) * torch.cos(lon)
    Y = depth * torch.sin(lat)
    Z = depth * torch.cos(lat) * torch.sin(lon)
    return X, Y, Z


# ==== XYZ→Normalized UV + depth ====

def XYZ_to_pinhole(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float):
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2)
    depth = torch.sqrt(X**2 + Y**2 + Z**2)
    phi   = torch.atan2(Y, X)
    theta = torch.acos(Z / depth)
    r     = f * torch.tan(theta)
    u     = r * torch.cos(phi)
    v     = r * torch.sin(phi)
    return u, v, depth

def XYZ_to_fisheye(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float):
    # equidistant fisheye: u = (θ/(fov/2))·cosφ, etc.
    fov_rad = math.radians(fov)
    depth   = torch.sqrt(X**2 + Y**2 + Z**2)
    theta   = torch.acos(Z / depth)
    phi     = torch.atan2(Y, X)
    r       = theta / (fov_rad / 2)
    u       = r * torch.cos(phi)
    v       = r * torch.sin(phi)
    return u, v, depth

def XYZ_to_equirect(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float):
    # full 360°×180°  
    fov_rad = math.radians(fov)
    depth = torch.sqrt(X**2 + Y**2 + Z**2)
    lon   = torch.atan2(X, Z)            # –π → +π
    lat   = torch.asin(Y / depth)        # –π/2 → +π/2
    u     = lon / fov_rad                # –1 → +1 across width
    v     = lat / (math.pi / 2)          # –1 → +1 down height
    return u, v, depth

def project_first_hit(volume_sparse: torch.Tensor) -> torch.Tensor:
    volume = volume_sparse.to_dense().float()      # (H, W, D, 4)
    
    hit       = volume[..., 3] > 0                # per‑slice hit
    cumsum    = hit.cumsum(dim=2)                 # cumulative hit count
    first_hit = hit & (cumsum == 1)               # only first
    
    # extract exactly one RGBA per pixel
    rgba = (volume * first_hit.unsqueeze(-1)).sum(dim=2)  # (H, W, 4)
    
    return rgba.permute(2, 0, 1), first_hit.any(dim=2)

# ==== Node Definitions ==== #
class DepthToPointCloud:
    """
    Convert an (optional) depth map and RGB(A) image into a single pointcloud tensor of shape (N,7) [X,Y,Z,R,G,B,A].
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "input_projection": (Projection.PROJECTIONS, {"tooltip": "projection type of depth map"}),
                "input_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            },
            "optional": {
                "depthmap": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "depth_to_pointcloud"
    CATEGORY = "pointcloud"

    def depth_to_pointcloud(
        self,
        image: torch.Tensor,
        input_projection: str,
        input_horizontal_fov: float,
        depthmap: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        # ----- handle image tensor -----
        img = image
        # if batched
        if img.dim() == 4:
            img = img.squeeze(0)
        # convert NHWC to NCHW or HWC to CHW
        if img.dim() == 3 and img.shape[2] in (3,4):  # H,W,C
            img = img.permute(2,0,1)
        # now img is (C,H,W)
        C, H, W = img.shape

        # ----- handle depth tensor -----
        if depthmap is None:
            depth = torch.ones((H, W), device=img.device)
        else:
            d = depthmap
            if d.dim() == 4:
                d = d.squeeze(0)
            # collapse channel dim
            if d.dim() == 3:
                # if single-channel, squeeze; else average channels
                if d.shape[0] == 1:
                    d = d.squeeze(0)
                else:
                    d = d.mean(dim=0)
            # now d is (H_d, W_d)
            H_d, W_d = d.shape
            if (H_d, W_d) != (H, W):
                d = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
                d = d.squeeze(0).squeeze(0)
            depth = d

        # ----- convert to XYZ -----
        if input_projection == "PINHOLE":
            X, Y, Z = pinhole_depth_to_XYZ(depth, input_horizontal_fov)
        elif input_projection == "FISHEYE":
            X, Y, Z = fisheye_depth_to_XYZ(depth, input_horizontal_fov)
        else:
            X, Y, Z = equirect_depth_to_XYZ(depth, input_horizontal_fov)

        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

        # ----- extract colors -----
        rgba = img.permute(1,2,0).float()  # H,W,C
        # add alpha if missing
        if C == 3:
            alpha = torch.ones((H, W, 1), device=rgba.device)*255
            rgba = torch.cat([rgba, alpha], dim=2)
        colors = rgba.reshape(-1, 4)

        # ----- concat into pointcloud -----
        pointcloud = torch.cat([coords, colors], dim=1)
        return (pointcloud,)

class TransformPointCloud:
    """
    Apply a 4×4 transform to a point cloud tensor (N,7) -> (N,7).
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud": ("TENSOR",),
                "transform_matrix": ("MAT_4X4",),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "transform_pointcloud"
    CATEGORY = "pointcloud"
    
    def transform_pointcloud(
        self,
        pointcloud: torch.Tensor,
        transform_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        coords = pointcloud[:, :3]
        attrs = pointcloud[:, 3:]
        N = coords.shape[0]
        transform_matrix=torch.tensor(transform_matrix, device=coords.device).reshape(4, 4).float()
        # convert to homogeneous coordinates
        homo = torch.cat([coords, torch.ones(N, 1, device=coords.device)], dim=1)
        # apply transform and drop homogeneous
        transformed = (transform_matrix.to(coords.device) @ homo.T).T[:, :3]
        # concatenate attributes back
        return (torch.cat([transformed, attrs], dim=1),)

class ProjectPointCloud:
    """
    Project a point cloud tensor (N,7) back into image & mask using z-buffering.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud": ("TENSOR",),
                "output_projection": (Projection.PROJECTIONS, {}),
                "output_horizontal_fov": ("FLOAT", {"default": 90.0}),
                "output_width": ("INT", {"default": 512, "min": 1}),
                "output_height": ("INT", {"default": 512, "min": 1}),
                "zbuffer_resolution": ("INT", {"default": ZBUFFER_RESOLUTION}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "project_pointcloud"
    CATEGORY = "pointcloud"

    def project_pointcloud(
        self,
        pointcloud: torch.Tensor,
        output_projection: str,
        output_horizontal_fov: float,
        output_width: int,
        output_height: int,
        zbuffer_resolution: int  # we no longer use this
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = pointcloud.device
        coords = pointcloud[:, :3]
        colors = pointcloud[:, 3:].float()      # assume in [0–255]

        # 1) project into camera space
        X, Y, Z = coords[:,0], coords[:,1], coords[:,2]
        if output_projection == "PINHOLE":
            u, v, depth = XYZ_to_pinhole(X, Y, Z, output_horizontal_fov)
        elif output_projection == "FISHEYE":
            u, v, depth = XYZ_to_fisheye(X, Y, Z, output_horizontal_fov)
        else:
            u, v, depth = XYZ_to_equirect(X, Y, Z, output_horizontal_fov)

        # 2) rasterize to integer pixel coords
        W, H = output_width, output_height
        px = (u * (W - 1) / 2) + (W - 1) / 2
        py = (v * (H - 1) / 2) + (H - 1) / 2
        ix = px.round().clamp(0, W-1).long()
        iy = py.round().clamp(0, H-1).long()

        # flatten 2D → 1D pixel index
        pix = iy * W + ix           # shape (N,)
        M   = W * H

        # 3) for each pixel, find its nearest depth
        min_depth = torch.full((M,), float('inf'), device=device)
        min_depth.scatter_reduce_(0, pix, depth, reduce='amin', include_self=True)

        # mask which points actually sit at that nearest depth
        is_min = depth == min_depth[pix]    # (N,)

        # 4) break ties by picking the first point seen
        order      = torch.arange(depth.shape[0], device=device)
        order_mask = torch.where(is_min, order, torch.full_like(order, depth.shape[0]))
        min_order  = torch.full((M,), depth.shape[0], device=device)
        min_order.scatter_reduce_(0, pix, order_mask, reduce='amin', include_self=True)
        winner     = order == min_order[pix]  # final 1‑point mask

        # 5) scatter that point’s RGBA into a flat H×W image
        flat_rgba = torch.zeros((M,4), device=device)
        flat_rgba[pix[winner]] = colors[winner]

        # 6) reshape and split out
        img_hw4 = flat_rgba.view(H, W, 4)      # (H,W,4)
        rgb     = img_hw4[..., :3].clamp(0.0,255.0)
        alpha   = (img_hw4[..., 3] > 0).float()  # (H,W)

        # apply alpha → premultiplied RGB
        rgb = rgb * alpha.unsqueeze(-1)         # (H,W,3)

        # 7) build outputs in the shape ComfyUI expects:
        #    image: (1, H, W, 3), mask: (H, W)
        img  = rgb.unsqueeze(0)      # (1, H, W, 3), floats in [0,1]
        mask = alpha                            # (H, W), floats 0 or 1

        return img, mask

NODE_CLASS_MAPPINGS = {
    "DepthToPointCloud": DepthToPointCloud,
    "TransformPointCloud": TransformPointCloud,
    "ProjectPointCloud": ProjectPointCloud,
}