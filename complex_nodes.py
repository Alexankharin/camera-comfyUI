import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
from PIL import Image

# reuse existing nodes
from .reprojection_nodes import ReprojectImage, ReprojectDepth
from .metric_depth_nodes import (
    DepthEstimatorNode,
    ZDepthToRayDepthNode,
    CombineDepthsNode,
    DepthRenormalizer
)

# HuggingFace models
possible_models = [
    "Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Depth-Anything-V2-Metric-Indoor-Large-hf",
    "Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth-Anything-V2-Metric-Outdoor-Large-hf",
]

class FisheyeDepthEstimator:
    """
    Estimates a full 180° fisheye depthmap by:
      1. Estimating depth on the full fisheye image
      2. Extracting 90° pinhole views (front, left, right, up, down)
      3. Running depth estimation + ray-depth conversion per view
      4. Reprojecting back to fisheye
      5. Renormalizing overlaps
      6. Soft-merging all six maps
    Outputs:
      - depthmap: [B,H,W,1]
      - mask: [B,H,W], circular mask of fisheye region
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image":                ("IMAGE",),
                "model_name":           ("STRING", {"choices": possible_models, "default": possible_models[0]}),
                "depth_scale":          ("FLOAT",  {"default":1.0,  "min":0.0, "max":1000.0, "step":0.01}),
                "pinhole_fov":          ("FLOAT",  {"default":90.0, "min":1.0, "max":179.0}),
                "pinhole_resolution":   ("INT",    {"default":1024, "min":64}),
                "fisheye_resolution":   ("INT",    {"default":4096, "min":64}),
                "softmerge_radius":     ("INT",    {"default":25,   "min":1,   "tooltip":"Gaussian radius for merging"}),
            }
        }
    RETURN_TYPES = ("TENSOR","MASK")
    RETURN_NAMES = ("depthmap","mask")
    FUNCTION = "estimate_fisheye_depth"
    CATEGORY = "Camera/depth"

    def estimate_fisheye_depth(
        self,
        image: torch.Tensor,
        model_name: str,
        depth_scale: float,
        pinhole_fov: float,
        pinhole_resolution: int,
        fisheye_resolution: int,
        softmerge_radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # instantiate helper nodes
        de_node   = DepthEstimatorNode()
        z2r_node  = ZDepthToRayDepthNode()
        ri_node   = ReprojectImage()
        rd_node   = ReprojectDepth()
        ren_node  = DepthRenormalizer()
        comb_node = CombineDepthsNode()

        # constants
        fisheye_fov = 180.0
        pin_w = pin_h = pinhole_resolution
        fish_w = fish_h = fisheye_resolution

        # 1) Full fisheye depth + ray-depth + mask
        depth_full, = de_node.estimate_depth(image, model_name, depth_scale)
        ray_full,   = z2r_node.depth_to_ray_depth(depth_full, fisheye_fov)
        mask_full   = (depth_full.squeeze(-1) > 0).float()

        # 2) Pinhole orientations
        rotations = [
            (0,   0,  0),    # front
            (0,  45,  0),    # right
            (0, -45,  0),    # left
            (45,  0,  0),    # up
            (-45, 0,  0),    # down
        ]

        fisheye_depths = [ray_full]
        fisheye_masks  = [mask_full]

        # helper: euler → 4×4 matrix
        def euler_to_matrix(pitch, yaw, roll):
            p, y, r = map(math.radians, (pitch, yaw, roll))
            Rx = torch.tensor([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]], dtype=torch.float32)
            Ry = torch.tensor([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]], dtype=torch.float32)
            Rz = torch.tensor([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]], dtype=torch.float32)
            R = Rz @ Ry @ Rx
            M = torch.eye(4, dtype=torch.float32)
            M[:3,:3] = R
            return M

        # 3) Process rotated views
        for pitch, yaw, roll in rotations:
            M    = euler_to_matrix(pitch, yaw, roll)
            M_inv= torch.inverse(M).numpy()
            M_np = M.numpy()

            # fisheye → pinhole
            img_pin, mask_pin = ri_node.reproject_image(
                image,
                input_horiszontal_fov = fisheye_fov,
                output_horiszontal_fov= pinhole_fov,
                input_projection     = "FISHEYE",
                output_projection    = "PINHOLE",
                output_width         = pin_w,
                output_height        = pin_h,
                transform_matrix     = M_np,
                feathering           = 0,
            )

            # depth estimation + ray-depth
            depth_pin, = de_node.estimate_depth(img_pin, model_name, depth_scale)
            ray_pin,   = z2r_node.depth_to_ray_depth(depth_pin, pinhole_fov)

            # pinhole → fisheye
            fish_depth, fish_mask = rd_node.reproject_depth(
                ray_pin,
                input_horizontal_fov = pinhole_fov,
                output_horizontal_fov= fisheye_fov,
                input_projection     = "PINHOLE",
                output_projection    = "FISHEYE",
                output_width         = fish_w,
                output_height        = fish_h,
                transform_matrix     = M_inv,
            )
            fisheye_depths.append(fish_depth)
            fisheye_masks.append(fish_mask)

        # 4) Merge all maps
        d_acc = fisheye_depths[0]
        m_acc = fisheye_masks[0]
        for d_new, m_new in zip(fisheye_depths[1:], fisheye_masks[1:]):
            # renormalize new to accumulated
            d_norm, = ren_node.renormalize_depth(
                d_new,
                d_acc,
                m_new,
                m_acc,
                use_inverse=False
            )

            # soft-merge
            d_acc, m_acc = comb_node.combine_depths(
                d_acc,
                m_acc,
                d_norm,
                m_new,
                mode               = "SOFTMERGE",
                invert_mask        = False,
                softmerge_radius   = softmerge_radius
            )

        # 5) Generate circular mask
        # grid of coords
        ys = torch.arange(fish_h, device=d_acc.device).view(1,fish_h,1)
        xs = torch.arange(fish_w, device=d_acc.device).view(1,1,fish_w)
        cy = (fish_h - 1) / 2.0
        cx = (fish_w - 1) / 2.0
        dist2 = (ys - cy)**2 + (xs - cx)**2
        radius2 = (min(fish_w, fish_h) / 2.0)**2
        circ_mask = (dist2 <= radius2).float()  # [1,H,W]

        return (d_acc, circ_mask)

NODE_CLASS_MAPPINGS = {"FisheyeDepthEstimator": FisheyeDepthEstimator}
