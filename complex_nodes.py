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
from .pointcloud_nodes import (
    ProjectPointCloud,
    DepthToPointCloud,
    TransformPointCloud)
from .flux_fisheye_filling_nodes import OutpaintAnyProjection

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
        ray_full,   = z2r_node.depth_to_ray_depth(depth_full, 90)
        mask_full   = (depth_full.squeeze(-1) > 0).float()

        # 2) Pinhole orientations
        rotations = [
            (0,   0,  0),    # front
            (0,  45,  0),    # right
            (0, -45,  0),    # left
            (45,  0,  0),    # up
            (-45, 0,  0),    # down
        ]

        fisheye_depths = []
        fisheye_masks  = []

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
            print("depth_pin", depth_pin.shape, depth_pin.min(), depth_pin.max(), depth_pin.mean(), depth_pin.std())
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
            print("fish_depth", fish_depth.shape, fish_depth.min(), fish_depth.max(), fish_depth.mean(), fish_depth.std())
            fisheye_depths.append(fish_depth)
            fisheye_masks.append(fish_mask)
        # add full fisheye depth
        fisheye_depths.append(depth_full)
        fisheye_masks.append(mask_full)
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
        print("dacc", d_acc.shape, d_acc.min(), d_acc.max(), d_acc.mean(), d_acc.std())
        return (d_acc, circ_mask)

class PointcloudTrajectoryEnricher:
    """
    Enriches a pointcloud along a camera trajectory by outpainting missing regions per view.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "pointcloud":     ("TENSOR",),
                "trajectory":     ("TENSOR",),  # (K,4,4)
                "camera_type":    (("PINHOLE","FISHEYE","EQUIRECTANGULAR"),),
                "horizontal_fov": ("FLOAT", {"default":90.0}),
                "width":          ("INT",   {"default":512}),
                "height":         ("INT",   {"default":512}),
                # outpainting params
                "patch_projection":    (("PINHOLE","FISHEYE","EQUIRECTANGULAR"),),
                "patch_horiz_fov":     ("FLOAT", {"default":90.0}),
                "patch_res":           ("INT",   {"default":512}),
                "patch_phi":           ("FLOAT", {"default":0.0}),
                "patch_theta":         ("FLOAT", {"default":0.0}),
                "prompt":              ("STRING",{"default":""}),
                "num_inference_steps": ("INT",   {"default":50}),
                "guidance_scale":      ("FLOAT", {"default":7.5}),
                "mask_blur":           ("INT",   {"default":5}),
            }
        }
    RETURN_TYPES = ("TENSOR","IMAGE")
    RETURN_NAMES = ("enriched_pointcloud","debug_image")
    FUNCTION = "enrich_trajectory"
    CATEGORY = "Camera/pointcloud"

    def enrich_trajectory(
        self,
        pointcloud: torch.Tensor,
        trajectory: torch.Tensor,
        camera_type: str,
        horizontal_fov: float,
        width: int,
        height: int,
        patch_projection: str,
        patch_horiz_fov: float,
        patch_res: int,
        patch_phi: float,
        patch_theta: float,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        mask_blur: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = pointcloud.device

        proj_node      = ProjectPointCloud()
        outpaint_node  = OutpaintAnyProjection()
        depth_node     = DepthEstimatorNode()
        renorm_node    = DepthRenormalizer()
        depth2pc_node  = DepthToPointCloud()
        transform_node = TransformPointCloud()

        enriched_pc = pointcloud.clone()
        debug_img = None

        for M in trajectory:
            # 1) transform into camera space and filter points with Z<=0
            pc_cam, = transform_node.transform_pointcloud(enriched_pc, M.numpy())
            pc_cam = pc_cam[pc_cam[:,2] > 0]

            # 2) project to image, mask, and depth
            img, mask, depth_map = proj_node.project_pointcloud(
                pc_cam,
                camera_type,
                horizontal_fov,
                width,
                height,
                point_size=1,
                return_inverse_depth=False,
            )

            # 3) outpaint: fill holes where mask==0
            hole_mask = (mask < 0.5).float()
            out_img, out_mask = outpaint_node.outpaint_any(
                img,
                input_projection    = camera_type,
                input_horiz_fov     = horizontal_fov,
                output_projection   = camera_type,
                output_horiz_fov    = horizontal_fov,
                output_width        = width,
                output_height       = height,
                patch_projection    = patch_projection,
                patch_horiz_fov     = patch_horiz_fov,
                patch_res           = patch_res,
                patch_phi           = patch_phi,
                patch_theta         = patch_theta,
                prompt              = prompt,
                num_inference_steps = num_inference_steps,
                cached              = False,
                guidance_scale      = guidance_scale,
                mask_blur           = mask_blur,
                mask                = hole_mask,
                debug               = False,
            )
            debug_img = out_img

            # 4) estimate new depth + renormalize using original mask as guidance
            new_depth, = depth_node.estimate_depth(out_img, "Depth-Anything-V2-Metric-Indoor-Base-hf", 1.0)
            norm_depth, = renorm_node.renormalize_depth(
                new_depth,
                depth_map,
                depth_mask=mask>-1,
                guidance_mask=mask,
                use_inverse=False,
            )

            # 5) convert inpainted depth->pointcloud for newly filled pixels only
            new_region = ((hole_mask > 0.5)).float()
            print("new region area ", new_region.sum().item(), "out of", mask.numel())
            pc_new, = depth2pc_node.depth_to_pointcloud(
                out_img,
                camera_type,
                horizontal_fov,
                depth_scale=1.0,
                invert_depth=False,
                depthmap=norm_depth,
                mask=new_region,
            )

            # 6) transform new points to world and append
            M_inv = torch.inverse(M).numpy()
            pc_world, = transform_node.transform_pointcloud(pc_new, M_inv)
            enriched_pc = torch.cat([enriched_pc, pc_world], dim=0)

        return (enriched_pc, debug_img)
    

NODE_CLASS_MAPPINGS = {"FisheyeDepthEstimator": FisheyeDepthEstimator,
                       "PointcloudTrajectoryEnricher": PointcloudTrajectoryEnricher}
