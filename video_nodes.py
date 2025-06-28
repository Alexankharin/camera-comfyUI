import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from typing import Dict, Any, Tuple
from tqdm import tqdm  # Added tqdm import

# Import existing pointcloud nodes and projection definitions
from .pointcloud_nodes import DepthToPointCloud, TransformPointCloud, ProjectPointCloud, Projection, PointCloudCleaner
import folder_paths

# Ensure video_depth_anything is on path
_here = os.path.dirname(os.path.abspath(__file__))

# climb up 3 levels: camera-comfyUI → custom_nodes → ComfyUI
COMFYUI_ROOT = os.path.abspath(os.path.join(_here, os.pardir, os.pardir, os.pardir))

# point at metric_depth inside the Video-Depth-Anything clone at the ComfyUI root
video_depth_path = os.path.join(COMFYUI_ROOT, "Video-Depth-Anything", "metric_depth")

# insert at front so it always wins
if video_depth_path not in sys.path:
    sys.path.insert(0, video_depth_path)

try:
    from video_depth_anything.video_depth import VideoDepthAnything
    print("✅ video_depth_anything module loaded successfully.")
except ImportError as e:
    raise ImportError(
        f"❌ Could not load video_depth_anything from {video_depth_path!r}: {e}"
    )

class VideoCameraMotionSequence:
    """
    Takes a sequence of RGB frames and corresponding depth maps,
    converts each frame+depth to a pointcloud, interpolates a camera
    trajectory to match video length, cleans the pointcloud if needed,
    and outputs reprojected images, masks, and depth maps per frame.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # Sequence of frames: Tensor [T, H, W, 3]
                "frames": ("IMAGE", {"shape_hint": [None, None, None, 3]}),
                # Sequence of depth maps: Tensor [T, H, W] or [T, H, W, 1]
                "depth_seq": ("TENSOR", {"shape_hint": [None, None, None]}),
                # Camera trajectory waypoints: Tensor [K, 4, 4]
                "trajectory": ("TENSOR", {"shape_hint": [None, 4, 4]}),
                # Input projection parameters
                "input_projection": (Projection.PROJECTIONS, {}),
                "input_horizontal_fov": ("FLOAT", {"default": 90.0}),
                "depth_scale": ("FLOAT", {"default": 1.0}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                # Output projection parameters
                "output_projection": (Projection.PROJECTIONS, {}),
                "output_horizontal_fov": ("FLOAT", {"default": 90.0}),
                "output_width": ("INT", {"default": 512, "min": 1}),
                "output_height": ("INT", {"default": 512, "min": 1}),
                "point_size": ("INT", {"default": 1, "min": 1}),
                # Cleaning parameters
                "voxel_size": ("FLOAT", {"default": 1.0, "min": 1e-3}),
                "min_points_per_voxel": ("INT", {"default": 3, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("video_frames", "mask_frames", "depths")
    FUNCTION = "process_sequence"
    CATEGORY = "Camera/Video"

    def process_sequence(
        self,
        frames: torch.Tensor,
        depth_seq: torch.Tensor,
        trajectory: torch.Tensor,
        input_projection: str,
        input_horizontal_fov: float,
        depth_scale: float,
        invert_depth: bool,
        output_projection: str,
        output_horizontal_fov: float,
        output_width: int,
        output_height: int,
        point_size: int,
        voxel_size: float,
        min_points_per_voxel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # frames: [T, H, W, 3]
        # depth_seq: [T, H, W] or [T, H, W, 1]
        T, H, W, _ = frames.shape

        # Interpolate trajectory to match T
        K = trajectory.shape[0]
        if K < 2:
            interp_traj = trajectory.expand(T, 4, 4).clone()
        else:
            idxs = torch.linspace(0, K - 1, T, device=trajectory.device)
            lower = idxs.floor().long().clamp(max=K - 2)
            upper = lower + 1
            alpha = (idxs - lower.float()).unsqueeze(-1).unsqueeze(-1)
            traj_lower = trajectory[lower]
            traj_upper = trajectory[upper]
            interp_traj = traj_lower * (1 - alpha) + traj_upper * alpha

        out_frames = []
        out_masks = []
        out_depths = []

        # Add tqdm progress bar for the sequence
        for frame, depth, pose in tqdm(zip(frames, depth_seq, interp_traj), total=T, desc="Processing video frames"):
            if depth.dim() == 3 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            # to pointcloud
            pc, = DepthToPointCloud().depth_to_pointcloud(
                image=frame.permute(2, 0, 1),
                input_projection=input_projection,
                input_horizontal_fov=input_horizontal_fov,
                depth_scale=depth_scale,
                invert_depth=invert_depth,
                depthmap=depth,
                mask=None,
            )
            # optional cleaning
            if min_points_per_voxel > 1:
                pc, = PointCloudCleaner().clean_pointcloud(
                    pointcloud=pc,
                    width=output_width,
                    height=output_height,
                    voxel_size=voxel_size,
                    min_points_per_voxel=min_points_per_voxel,
                )
            # transform and project
            pc_t, = TransformPointCloud().transform_pointcloud(pc, pose)
            img_t, mask_t, depth_t = ProjectPointCloud().project_pointcloud(
                pointcloud=pc_t,
                output_projection=output_projection,
                output_horizontal_fov=output_horizontal_fov,
                output_width=output_width,
                output_height=output_height,
                point_size=point_size,
            )

            out_frames.append(img_t[0])
            out_masks.append(mask_t)
            out_depths.append(depth_t)

        return (
            torch.stack(out_frames, dim=0),  # [T, 3, H, W]
            torch.stack(out_masks, dim=0),   # [T, H, W]
            torch.stack(out_depths, dim=0),  # [T, H, W]
        )


class DepthFramesToVideo:
    """
    Converts a sequence of depth maps into video frame tensors for saving.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth_seq": ("TENSOR", {"shape_hint": [None, None, None]}),
                "mask_seq": ("MASK", {"shape_hint": [None, None, None]}),
                "normalize": ("BOOLEAN", {"default": True}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TENSOR", "IMAGE")
    RETURN_NAMES = ("video_frames", "depth_video")
    FUNCTION = "depth_to_video_frames"
    CATEGORY = "Camera/Video"

    def depth_to_video_frames(
        self,
        depth_seq: torch.Tensor,
        normalize: bool,
        invert_depth: bool,
        mask_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = depth_seq.clone().squeeze()
        if ds.dim() == 2:
            ds = ds.unsqueeze(0)  # [H, W] -> [1, H, W]
        if ds.dim() != 3:
            raise ValueError(f"Expected ds to be 3D [T, H, W], got shape {ds.shape}")
        if invert_depth:
            ds= 1.0 / (ds + 1e-8)  # Avoid division by zero
        if normalize:
            # Mask: only normalize where depth > 0
            mask = mask_seq>0.5
            if mask.any():
                #percentile first 10 percent min
                # sample 
                minv = ds[mask]
                # sample 10000 and find 10% quantile
                if minv.numel() > 10000:
                    minv = minv[torch.randperm(minv.numel())[:10000]]
                
                minv = minv.quantile(0.2)
                minv = minv if minv > 0.1 else 0.1  # Avoid division by zero
                #percentile last 10 percent max
                maxv = ds[mask]
                if maxv.numel() > 10000:
                    maxv = maxv[torch.randperm(maxv.numel())[:10000]]
                maxv = maxv.quantile(0.98)
                maxv = maxv if maxv < 100 else 100
                print(f"Normalizing depth: min={minv}, max={maxv}")
                ds_norm = (ds - minv) / (maxv - minv + 1e-8)
                ds = ds_norm.clamp(0, 1)  # torch.where(mask, ds_norm, ds)  # Only normalize valid values
            else:
                print("Warning: No valid depth values for normalization.")
        # expand to 3 channels: [T, H, W] -> [T, 3, H, W]
        raw = depth_seq.clone().squeeze()
        ds_u8 = (ds * 255.0).round().to(torch.uint8)
        raw_u8 = (raw.clamp(0, 255)).to(torch.uint8)  # if raw is already in a displayable range

        # expand to 3 channels and permute to HWC
        ds_color = ds_u8.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        raw_color = raw_u8.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        return raw_color, ds_color   # [T, 3, H, W] -> [T, H, W, 3]

class VideoMetricDepthEstimate:
    """
    Estimates metric depth for a sequence of frames using VideoDepthAnything.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # model files (.pth) in input directory
        model_dir = os.path.join(os.getcwd(), "models", "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        files = [f for f in os.listdir(model_dir) if f.lower().endswith(('.pth', '.ckpt', '.safetensors'))]
        return {
            "required": {
                "frames":     ("IMAGE", {"shape_hint": [None, None, None, 3]}),
                "model_checkpoint": (files,   {"file_chooser": True}),
                "input_size": ("INT",  {"default": 518, "min": 64, "max": 2048}),
                "max_fps":    ("INT",  {"default": 60,  "min": 1}),
            }
        }
    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("metric_depths", "fps")
    FUNCTION = "estimate_metric_depth"
    CATEGORY = "Camera/Video"

    def estimate_metric_depth(
        self,
        frames: torch.Tensor,
        model_checkpoint: str,
        input_size: int,
        max_fps: int,
    ) -> Tuple[torch.Tensor, float]:
        if VideoDepthAnything is None:
            raise ImportError("VideoDepthAnything library not found")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if max input<1.5 normalize to 0-255
        if frames.max() < 1.5:
            frames = (frames * 255)
        model = VideoDepthAnything(**{"encoder": "vitl", "features": 256, "out_channels": [256,512,1024,1024]})
        state = torch.load("/root/ComfyUI/models/checkpoints/{}".format(model_checkpoint), map_location='cpu')
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        np_frames = frames.cpu().numpy().astype(np.uint8)
        metric_depths, fps = model.infer_video_depth(np_frames, max_fps, input_size=input_size, device=device.type, fp32=False)
        return (torch.from_numpy(metric_depths), float(fps))

# Register nodes
NODE_CLASS_MAPPINGS = {
    "VideoCameraMotionSequence": VideoCameraMotionSequence,
    "VideoMetricDepthEstimate": VideoMetricDepthEstimate,
    "DepthFramesToVideo": DepthFramesToVideo,
}
