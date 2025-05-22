import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
from PIL import Image
from tqdm import tqdm

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
    TransformPointCloud,
    PointCloudCleaner,
    ProjectAndClean
)
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
      3. Running depth estimation per view
      4. Reprojecting back to fisheye
      5. Renormalizing overlaps
      6. Merging all six maps with selectable mode
    Outputs:
      - depthmap: [B,H,W,1]
      - mask: [B,H,W], circular mask of fisheye region
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image":              ("IMAGE",),
                "model_name":         (possible_models, {"default": possible_models[0]}),
                "depth_scale":        ("FLOAT",  {"default":1.0, "min":0.0, "max":1000.0, "step":0.01}),
                "pinhole_fov":        ("FLOAT",  {"default":90.0, "min":1.0, "max":179.0}),
                "pinhole_resolution": ("INT",    {"default":1024, "min":64}),
                "fisheye_resolution": ("INT",    {"default":4096, "min":64, "max":8192}),
                "mode":               (["SRC","DST","AVERAGE","SOFTMERGE","DISTANCE_AWARE"], {"default":"AVERAGE"}),
                "softmerge_radius":   ("INT",    {"default":25,    "min":1, "tooltip":"Gaussian radius for merging"}),
                "median_blur_kernel": ("INT",    {"default":1,     "min":1, "max":31, "tooltip":"Kernel size for median blur of depthmap"}),
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
        mode: str,
        softmerge_radius: int,
        median_blur_kernel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # helper nodes
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

        # 1) Full fisheye depth + mask
        depth_full, = de_node.estimate_depth(image, model_name, depth_scale)
        mask_full   = (depth_full > 0).float()

        # 2) Generate Pinhole Views
        fisheye_depths, fisheye_masks = self._generate_pinhole_views(
            image,
            de_node, z2r_node, ri_node, rd_node,
            fisheye_fov, pinhole_fov,
            pin_w, pin_h, fish_w, fish_h,
            model_name, depth_scale, median_blur_kernel
        )

        fisheye_depths.append(depth_full) # [B,H,W]
        fisheye_masks.append(mask_full.squeeze(-1)) # [B,H,W 1]
        # merged mask
        merged_mask = torch.sum(torch.stack(fisheye_masks), dim=0) > 0.5
        # print(fisheye_depths[0].shape, fisheye_depths[-1].shape, merged_mask.shape)
        # 4) Merge in sequence
        d_acc, m_acc = self._merge_depths(
            fisheye_depths, fisheye_masks,
            ren_node, comb_node,
            mode, softmerge_radius
        )

        # 5) Circular mask
        ys = torch.arange(fish_h, device=d_acc.device).view(1, fish_h, 1)
        xs = torch.arange(fish_w, device=d_acc.device).view(1, 1, fish_w)
        cy = (fish_h - 1) / 2.0
        cx = (fish_w - 1) / 2.0
        dist2   = (ys - cy)**2 + (xs - cx)**2
        radius2 = (min(fish_w, fish_h) / 2.0)**2
        circ_mask = (dist2 <= radius2).float()
        return d_acc, circ_mask

    def _merge_depths(
        self,
        fisheye_depths: list,
        fisheye_masks: list,
        ren_node: DepthRenormalizer,
        comb_node: CombineDepthsNode,
        mode: str,
        softmerge_radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_acc = fisheye_depths[0]
        m_acc = fisheye_masks[0]
        for d_new, m_new in zip(fisheye_depths[1:-1], fisheye_masks[1:-1]):
            d_norm, = ren_node.renormalize_depth(d_new, d_acc, m_new, m_acc, use_inverse=False)
            d_acc, m_acc = comb_node.combine_depths(
                d_acc,
                m_acc,
                d_norm,
                m_new,
                mode             = mode,
                invert_mask      = False,
                softmerge_radius = softmerge_radius
            )
        # last depth is full fisheye -mode is SRC to prevent resolution drawbacks
        m_new_last = fisheye_masks[-1]
        d_new_last = fisheye_depths[-1]
        d_norm, = ren_node.renormalize_depth(d_new_last, d_acc, m_new_last, m_acc, use_inverse=False)
        d_acc,m_acc = comb_node.combine_depths(
            d_acc,
            m_acc,
            d_norm,
            m_new_last,
            mode             = "SRC", # Use SRC for the full fisheye to preserve its details
            invert_mask      = False,
            softmerge_radius = softmerge_radius
        )
        return d_acc, m_acc

    def _generate_pinhole_views(
        self,
        image: torch.Tensor,
        de_node: DepthEstimatorNode,
        z2r_node: ZDepthToRayDepthNode,
        ri_node: ReprojectImage,
        rd_node: ReprojectDepth,
        fisheye_fov: float,
        pinhole_fov: float,
        pin_w: int,
        pin_h: int,
        fish_w: int,
        fish_h: int,
        model_name: str,
        depth_scale: float,
        median_blur_kernel: int,
    ) -> Tuple[list, list]:
        rotations = [
            (0,   0,  0),    # front
            (0,  45,  0),    # right
            (0, -45,  0),    # left
            (45,  0,  0),    # up
            (-45, 0,  0),    # down
        ]

        fisheye_depths = []
        fisheye_masks  = []

        for pitch, yaw, roll in rotations:
            M     = self._euler_to_matrix(pitch, yaw, roll)
            M_np  = M.numpy()
            M_inv = torch.inverse(M).numpy()

            fish_depth, fish_mask = self._process_view(
                image, M_np, M_inv,
                de_node, z2r_node, ri_node, rd_node,
                fisheye_fov, pinhole_fov,
                pin_w, pin_h, fish_w, fish_h,
                model_name, depth_scale, median_blur_kernel
            )

            fisheye_depths.append(fish_depth)
            fisheye_masks.append(fish_mask)
        
        return fisheye_depths, fisheye_masks

    def _process_view(
        self,
        image: torch.Tensor,
        M_np: np.ndarray,
        M_inv: np.ndarray,
        de_node: DepthEstimatorNode,
        z2r_node: ZDepthToRayDepthNode,
        ri_node: ReprojectImage,
        rd_node: ReprojectDepth,
        fisheye_fov: float,
        pinhole_fov: float,
        pin_w: int,
        pin_h: int,
        fish_w: int,
        fish_h: int,
        model_name: str,
        depth_scale: float,
        median_blur_kernel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # estimate pinhole depth
        depth_pin, = de_node.estimate_depth(img_pin, model_name, depth_scale, median_blur_kernel=median_blur_kernel)
        depth_pin, = z2r_node.depth_to_ray_depth(
            depth_pin,
            pinhole_fov,
        )
        # pinhole → fisheye
        fish_depth, fish_mask = rd_node.reproject_depth(
            depth_pin,
            input_horizontal_fov = pinhole_fov,
            output_horizontal_fov= fisheye_fov,
            input_projection     = "PINHOLE",
            output_projection    = "FISHEYE",
            output_width         = fish_w,
            output_height        = fish_h,
            transform_matrix     = M_inv,
        )
        # squeeze mask to [B,H,W]
        fish_mask = fish_mask.squeeze(1)
        return fish_depth, fish_mask

    def _euler_to_matrix(self, pitch, yaw, roll):
        p, y, r = map(math.radians, (pitch, yaw, roll))
        Rx = torch.tensor([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]], dtype=torch.float32)
        Ry = torch.tensor([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]], dtype=torch.float32)
        Rz = torch.tensor([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]], dtype=torch.float32)
        R = Rz @ Ry @ Rx
        M = torch.eye(4, dtype=torch.float32)
        M[:3, :3] = R
        return M

class PointcloudTrajectoryEnricher:
    """
    Enriches a pointcloud along a camera trajectory by outpainting missing regions per view.

    Returns:
      - enriched_pointcloud: [N,4+] tensor
      - debug_image:         [1,H,W,3] tensor
      - debug_depth:         [1,H,W,1] tensor
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "pointcloud":            ("TENSOR",),
                "trajectory":            ("TENSOR",),  # (K,4,4)
                "camera_type":           (("PINHOLE","FISHEYE","EQUIRECTANGULAR"),),
                "horizontal_fov":        ("FLOAT", {"default":90.0}),
                "width":                 ("INT",   {"default":512}),
                "height":                ("INT",   {"default":512}),
                # outpainting params
                "patch_projection":      (("PINHOLE","FISHEYE","EQUIRECTANGULAR"),),
                "patch_horiz_fov":       ("FLOAT", {"default":90.0}),
                "patch_res":             ("INT",   {"default":512}),
                "patch_phi":             ("FLOAT", {"default":0.0}),
                "patch_theta":           ("FLOAT", {"default":0.0}),
                "prompt":                ("STRING",{"default":""}),
                "num_inference_steps":   ("INT",   {"default":50}),
                "guidance_scale":        ("FLOAT", {"default":7.5}),
                "mask_blur":             ("INT",   {"default":5}),
                # cleaning params
                "voxel_size":            ("FLOAT", {"default":0.07}),
                "min_points_per_voxel":  ("INT",   {"default":3}),
                # depth estimation model
                "model_name":            ("STRING", {"choices": possible_models, "default": possible_models[0]}),
            }
        }
    RETURN_TYPES = ("TENSOR","IMAGE","TENSOR")
    RETURN_NAMES = ("enriched_pointcloud","debug_image","debug_depth")
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
        voxel_size: float,
        min_points_per_voxel: int,
        model_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = pointcloud.device

        # reuse stateless nodes
        proj_node      = ProjectPointCloud()
        outpaint_node  = OutpaintAnyProjection()
        depth_node     = DepthEstimatorNode()
        renorm_node    = DepthRenormalizer()
        depth2pc_node  = DepthToPointCloud()
        transform_node = TransformPointCloud()
        clean_node     = PointCloudCleaner()
        zdepth_node   = ZDepthToRayDepthNode()

        # initial clean on GPU
        with torch.no_grad():
            pointcloud, = clean_node.clean_pointcloud(
                pointcloud,
                voxel_size=voxel_size,
                min_points_per_voxel=min_points_per_voxel,
                width=4096,
                height=4096,
            )
        debug_img = torch.zeros((1, height, width, 3), device=device)
        debug_depth = torch.zeros((1, height, width, 1), device=device)
        enriched_pc = pointcloud
        # Initialize debug_img and debug_depth which will be updated in the loop
        # and will hold the values from the last processed view.
        debug_img = torch.zeros((1, height, width, 3), device=device)
        debug_depth = torch.zeros((1, height, width, 1), device=device)

        # loop over trajectory (limit or full)
        for M in tqdm(trajectory[:15], desc="Enriching trajectory"):
            enriched_pc, view_debug_img, view_debug_depth = self._process_single_view(
                M, enriched_pc, device,
                proj_node, outpaint_node, depth_node, renorm_node,
                depth2pc_node, transform_node, clean_node, zdepth_node,
                camera_type, horizontal_fov, width, height,
                patch_projection, patch_horiz_fov, patch_res,
                patch_phi, patch_theta, prompt,
                num_inference_steps, guidance_scale, mask_blur,
                voxel_size, min_points_per_voxel, model_name
            )
            debug_img = view_debug_img
            debug_depth = view_debug_depth
        return enriched_pc, debug_img, debug_depth

    def _process_single_view(
        self,
        M_matrix: torch.Tensor,
        current_enriched_pc: torch.Tensor,
        device: torch.device,
        proj_node: ProjectPointCloud,
        outpaint_node: OutpaintAnyProjection,
        depth_node: DepthEstimatorNode,
        renorm_node: DepthRenormalizer,
        depth2pc_node: DepthToPointCloud,
        transform_node: TransformPointCloud,
        clean_node: PointCloudCleaner,
        zdepth_node: ZDepthToRayDepthNode,
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
        voxel_size: float,
        min_points_per_voxel: int,
        model_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M_np = M_matrix.cpu().numpy()
        M_inv = np.linalg.inv(M_np)

        img, mask, depth_map, pc_front = self._prepare_view_data(
            current_enriched_pc, M_np, transform_node, clean_node, proj_node,
            voxel_size, min_points_per_voxel, camera_type, horizontal_fov,
            width, height
        )

        # fill nan in depthmap with (-1)
        # outpaint missing regions
        hole_mask = (mask < 0.5).float()
        out_img = self._outpaint_missing_regions(
            img, hole_mask, # Pass hole_mask instead of the full mask
            outpaint_node, camera_type, horizontal_fov, width, height,
            patch_projection, patch_horiz_fov, patch_res,
            patch_phi, patch_theta, prompt,
            num_inference_steps, guidance_scale, mask_blur
        )
        # estimate and renormalize depth
        norm_depth, debug_depth_view = self._estimate_and_refine_depth(
            out_img, depth_map, mask, hole_mask,
            depth_node, zdepth_node, renorm_node,
            model_name, horizontal_fov
        )

        # back to pointcloud
        pc_world = self._convert_depth_to_world_pointcloud(
            out_img, norm_depth, hole_mask, M_inv,
            depth2pc_node, transform_node,
            camera_type, horizontal_fov
        )
        # enriched_pc is not rotated
        current_enriched_pc = torch.cat([current_enriched_pc, pc_world.to(device)], dim=0)
        return current_enriched_pc, out_img, debug_depth_view # Return out_img and the depth for this view

    def _convert_depth_to_world_pointcloud(
        self,
        out_img: torch.Tensor,
        norm_depth: torch.Tensor,
        hole_mask: torch.Tensor,
        M_inv: np.ndarray,
        depth2pc_node: DepthToPointCloud,
        transform_node: TransformPointCloud,
        camera_type: str,
        horizontal_fov: float,
    ) -> torch.Tensor:
        pc_new, = depth2pc_node.depth_to_pointcloud(
            out_img,
            camera_type,
            horizontal_fov,
            depth_scale=1.0,
            invert_depth=False,
            depthmap=norm_depth,
            mask=hole_mask,
        )
        pc_world, = transform_node.transform_pointcloud(pc_new, M_inv)
        return pc_world

    def _estimate_and_refine_depth(
        self,
        out_img: torch.Tensor,
        depth_map: torch.Tensor,
        original_mask: torch.Tensor, # Mask from projection
        hole_mask: torch.Tensor,
        depth_node: DepthEstimatorNode,
        zdepth_node: ZDepthToRayDepthNode,
        renorm_node: DepthRenormalizer,
        model_name: str,
        horizontal_fov: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nan_mask = torch.isnan(depth_map)
        depth_map[nan_mask] = 0  # In-place modification
        depth_map = torch.clamp(depth_map, 0, 1000.0)

        new_depth, = depth_node.estimate_depth(out_img, model_name, depth_scale=1.0)
        new_depth, = zdepth_node.depth_to_ray_depth(
            new_depth,
            horizontal_fov,
        )
        # renormalize depth
        # Use original_mask for depth_mask as it represents valid projected areas
        norm_depth, = renorm_node.renormalize_depth(
            new_depth,
            depth_map,
            depth_mask=(original_mask >= 0.5) * 1,
            guidance_mask=(hole_mask >= 0.5) * 1, # hole_mask is appropriate here
            use_inverse=False,
        )
        # median blur on depth
        k = 5
        d = norm_depth.permute(0,3,1,2)  # [B,1,H,W]
        pad = k//2
        pd = F.pad(d, (pad, pad, pad, pad), mode='reflect')
        patches = pd.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.contiguous().view(d.shape[0], d.shape[1], d.shape[2], d.shape[3], k*k)
        d_median, _ = patches.median(dim=-1) # Renamed to avoid conflict
        norm_depth_blurred = d_median.permute(0,2,3,1)  # [B,H,W,1]

        # Create debug_depth_view using the blurred normalized depth for holes
        # and the original depth_map for non-holes.
        debug_depth_view = norm_depth_blurred * hole_mask.unsqueeze(0).unsqueeze(-1) + \
                           depth_map * (1 - hole_mask.unsqueeze(0).unsqueeze(-1))
        
        return norm_depth_blurred, debug_depth_view


    def _outpaint_missing_regions(
        self,
        img: torch.Tensor,
        hole_mask: torch.Tensor, # Expects the specific hole_mask
        outpaint_node: OutpaintAnyProjection,
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
    ) -> torch.Tensor: # Returns only out_img, out_mask is not used later
        out_img, _ = outpaint_node.outpaint_any( # Assign out_mask to _
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
            mask                = hole_mask, # Use the passed hole_mask
            debug               = False,
        )
        return out_img

    def _prepare_view_data(
        self,
        current_enriched_pc: torch.Tensor,
        M_np: np.ndarray,
        transform_node: TransformPointCloud,
        clean_node: PointCloudCleaner,
        proj_node: ProjectPointCloud,
        voxel_size: float,
        min_points_per_voxel: int,
        camera_type: str,
        horizontal_fov: float,
        width: int,
        height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # transform and select front points
        rotated, = transform_node.transform_pointcloud(current_enriched_pc, M_np)
        pc_front = rotated[rotated[:, 2] > 0]

        # clean front points
        pc_front, = clean_node.clean_pointcloud(
            pc_front,
            voxel_size=voxel_size,
            min_points_per_voxel=min_points_per_voxel,
            width=4096, # Consider passing these as params if they vary
            height=4096, # Consider passing these as params if they vary
        )

        # project to image + depth
        img, mask, depth_map = proj_node.project_pointcloud(
            pc_front,
            camera_type,
            horizontal_fov,
            width,
            height,
            point_size=3,
            return_inverse_depth=False,
        )
        return img, mask, depth_map, pc_front

NODE_CLASS_MAPPINGS = {"FisheyeDepthEstimator": FisheyeDepthEstimator,
                       "PointcloudTrajectoryEnricher": PointcloudTrajectoryEnricher}
