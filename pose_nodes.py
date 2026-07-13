"""Camera pose estimation nodes.

Provides:
- VideoPoseEstimator: VGGT-based per-frame camera pose + depth + intrinsics
  estimation from a video clip.
- TrajectoryInvert / TrajectoryCompose: small utility nodes for wiring
  trajectory tensors ([K, 4, 4] world-to-camera matrices) in graphs.

Coordinate convention (matches the rest of this repo): camera frame is
+X right, +Y down, +Z forward; trajectory matrices are 4x4 world-to-camera
(`cam_pts = world_pts @ R.T + t`). VGGT outputs OpenCV-convention
camera-from-world extrinsics, which match this convention directly.
"""

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import folder_paths
except ImportError:  # Allow notebook usage outside ComfyUI
    class _FolderPathsStub:
        def __getattr__(self, name):
            raise ModuleNotFoundError(
                "folder_paths is unavailable; this feature requires the ComfyUI runtime."
            )

    folder_paths = _FolderPathsStub()

_here = os.path.dirname(os.path.abspath(__file__))
# climb up 2 levels: camera-comfyUI -> custom_nodes -> ComfyUI
COMFYUI_ROOT = os.path.abspath(os.path.join(_here, os.pardir, os.pardir))

DEVICE_CHOICES = ["auto", "cpu", "cuda"]

# Module-level model cache: {device_str: model}
_VGGT_MODEL_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# SE(3) interpolation (contract C1). Prefer the shared implementation from
# pointcloud_nodes; fall back to a local copy so this file works standalone.
# --------------------------------------------------------------------------- #

def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert a single 3x3 rotation matrix to a wxyz quaternion."""
    R = R.to(torch.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return (q / q.norm().clamp(min=1e-12)).to(torch.float32)


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    q = q / q.norm().clamp(min=1e-12)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = q0 / q0.norm().clamp(min=1e-12)
    q1 = q1 / q1.norm().clamp(min=1e-12)
    dot = torch.dot(q0, q1)
    if dot < 0.0:  # take the short path on the quaternion hypersphere
        q1 = -q1
        dot = -dot
    dot = dot.clamp(-1.0, 1.0)
    if dot > 0.9995:  # nearly parallel: lerp + renormalize is numerically safer
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / q.norm().clamp(min=1e-12)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1.0 - alpha) * theta) / sin_theta
    w1 = torch.sin(alpha * theta) / sin_theta
    q = w0 * q0 + w1 * q1
    return q / q.norm().clamp(min=1e-12)


def _interpolate_se3_fallback(trajectory: torch.Tensor, num_steps: int) -> torch.Tensor:
    """trajectory [K,4,4] -> [num_steps,4,4]. Piecewise: quaternion SLERP on R, lerp on t.
    K==1 -> repeat. Returns valid (orthonormal) rotation matrices. Matches contract C1."""
    traj = torch.as_tensor(trajectory, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(f"Expected trajectory of shape [K,4,4], got {tuple(traj.shape)}")
    K = traj.shape[0]
    if K == 1:
        return traj.expand(num_steps, 4, 4).clone()
    quats = torch.stack([_matrix_to_quaternion(traj[i, :3, :3]) for i in range(K)])
    trans = traj[:, :3, 3]
    positions = torch.linspace(0.0, float(K - 1), num_steps)
    out = []
    for pos in positions:
        lower = int(torch.floor(pos).clamp(max=K - 2))
        upper = lower + 1
        alpha = float(pos) - lower
        q = _quat_slerp(quats[lower], quats[upper], alpha)
        t = (1.0 - alpha) * trans[lower] + alpha * trans[upper]
        M = torch.eye(4, dtype=torch.float32)
        M[:3, :3] = _quaternion_to_matrix(q)
        M[:3, 3] = t
        out.append(M)
    return torch.stack(out, dim=0)


try:
    from .pointcloud_nodes import interpolate_se3
except Exception:
    try:
        from pointcloud_nodes import interpolate_se3
    except Exception:
        interpolate_se3 = _interpolate_se3_fallback


# --------------------------------------------------------------------------- #
# VGGT lazy import helpers
# --------------------------------------------------------------------------- #

def _import_vggt() -> Tuple[Any, Any]:
    """Lazily import VGGT. Tries the pip package first, then a sibling clone
    at COMFYUI_ROOT/vggt (mirroring how video_nodes.py handles Video-Depth-Anything)."""
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        return VGGT, pose_encoding_to_extri_intri
    except ImportError:
        pass

    vggt_clone_path = os.path.join(COMFYUI_ROOT, "vggt")
    if os.path.isdir(vggt_clone_path) and vggt_clone_path not in sys.path:
        sys.path.insert(0, vggt_clone_path)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        return VGGT, pose_encoding_to_extri_intri
    except ImportError as exc:
        raise ModuleNotFoundError(
            "VGGT is not installed. Install it with `pip install vggt` (or "
            "`pip install git+https://github.com/facebookresearch/vggt.git`), or clone "
            f"https://github.com/facebookresearch/vggt into {vggt_clone_path!r}. "
            "It also requires `huggingface_hub` to download the facebook/VGGT-1B weights."
        ) from exc


def _get_vggt_model(device: torch.device) -> Any:
    """Load (and cache) the VGGT-1B model on the requested device."""
    key = str(device)
    if key not in _VGGT_MODEL_CACHE:
        VGGT, _ = _import_vggt()
        print(f"[pose_nodes] Loading facebook/VGGT-1B onto {key} (first call downloads ~5GB weights)...")
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = model.to(device).eval()
        _VGGT_MODEL_CACHE[key] = model
    return _VGGT_MODEL_CACHE[key]


def _vggt_preprocess(frames: torch.Tensor, resolution: int, device: torch.device) -> torch.Tensor:
    """[T,H,W,3] float 0..1 -> [1,T,3,Hp,Wp] with max dim == resolution (both dims
    divisible by 14, the VGGT patch size), aspect ratio preserved."""
    T, H, W, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    if imgs.max() > 1.5:  # defensively handle 0..255 inputs
        imgs = imgs / 255.0
    scale = float(resolution) / float(max(H, W))
    new_h = max(14, int(round(H * scale / 14.0)) * 14)
    new_w = max(14, int(round(W * scale / 14.0)) * 14)
    if (new_h, new_w) != (H, W):
        imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return imgs.clamp(0.0, 1.0).unsqueeze(0)  # [1,T,3,Hp,Wp]


class VideoPoseEstimator:
    """
    Estimates per-frame camera poses (world-to-camera [T,4,4]), metric-ish depth
    maps, depth confidence and the horizontal FOV from a video clip using
    facebook/VGGT-1B.

    VGGT extrinsics use the OpenCV camera convention (+X right, +Y down,
    +Z forward, camera-from-world), which matches this repo's trajectory
    convention, so the matrices are returned as-is (padded to 4x4).
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # Video frames: Tensor [T, H, W, 3] float 0..1
                "frames": ("IMAGE", {"shape_hint": [None, None, None, 3]}),
                "max_frames": ("INT", {
                    "default": 64, "min": 1, "max": 1024,
                    "tooltip": "If the clip has more frames than this, it is stride-subsampled "
                               "for VGGT and the poses are SE(3)-interpolated back to full length "
                               "(depth/confidence use nearest-frame fill).",
                }),
                "resolution": ("INT", {
                    "default": 518, "min": 98, "max": 1036,
                    "tooltip": "Max image dimension fed to VGGT (rounded to a multiple of 14).",
                }),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "FLOAT", "TENSOR")
    RETURN_NAMES = ("trajectory", "depths", "horizontal_fov", "confidence")
    FUNCTION = "estimate_poses"
    CATEGORY = "Camera/Pose"
    DESCRIPTION = (
        "VGGT camera pose + depth estimation. Outputs world-to-camera trajectory [T,4,4], "
        "depth maps [T,H,W] at the input resolution, mean horizontal FOV (degrees) and "
        "per-pixel depth confidence [T,H,W]."
    )

    def estimate_poses(
        self,
        frames: torch.Tensor,
        max_frames: int = 64,
        resolution: int = 518,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        if frames.dim() != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames of shape [T,H,W,3], got {tuple(frames.shape)}")
        T_full, H, W, _ = frames.shape

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available.")
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        # Stride-subsample overly long clips, keeping the frame mapping so that
        # poses can be interpolated back afterwards.
        if T_full > max_frames:
            sub_indices = torch.linspace(0, T_full - 1, max_frames).round().long().unique()
            print(
                f"[VideoPoseEstimator] WARNING: clip has {T_full} frames > max_frames={max_frames}; "
                f"running VGGT on {sub_indices.numel()} stride-subsampled frames. Poses are "
                "SE(3)-interpolated back to full length; depth/confidence use nearest-frame fill. "
                "Increase max_frames for exact per-frame estimates."
            )
            proc_frames = frames[sub_indices]
        else:
            sub_indices = None
            proc_frames = frames

        images = _vggt_preprocess(proc_frames, resolution, dev)  # [1,S,3,Hp,Wp]
        S, Hp, Wp = images.shape[1], images.shape[-2], images.shape[-1]

        _, pose_encoding_to_extri_intri = _import_vggt()
        model = _get_vggt_model(dev)

        try:
            with torch.no_grad():
                if dev.type == "cuda":
                    capability = torch.cuda.get_device_capability(dev)
                    amp_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        aggregated_tokens_list, ps_idx = model.aggregator(images)
                else:
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                # Camera + depth heads run in full precision (per the official VGGT example).
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"VGGT ran out of GPU memory on {S} frames at {Wp}x{Hp}. "
                "Lower max_frames and/or resolution, or set device='cpu' (slow)."
            ) from exc

        # ---- Trajectory: pad OpenCV world-to-camera [S,3,4] to [S,4,4] ---- #
        extrinsic = extrinsic.squeeze(0).to(torch.float32).cpu()  # [S,3,4]
        trajectory = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)
        trajectory[:, :3, :4] = extrinsic

        # ---- Horizontal FOV from intrinsics (resolution-invariant fx/W ratio) ---- #
        intrinsic = intrinsic.squeeze(0).to(torch.float32).cpu()  # [S,3,3]
        fx = intrinsic[:, 0, 0].clamp(min=1e-6)
        hfov_per_frame = 2.0 * torch.atan(0.5 * float(Wp) / fx)  # radians, at processing width
        # Aspect ratio is preserved during preprocessing, so fx/W is the same at
        # the original width and the FOV needs no conversion.
        horizontal_fov = float(torch.rad2deg(hfov_per_frame).mean())

        # ---- Depth + confidence, resized back to the input resolution ---- #
        depth = depth_map.squeeze(0).to(torch.float32).cpu()  # [S,Hp,Wp,1] (or [S,Hp,Wp])
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        conf = depth_conf.squeeze(0).to(torch.float32).cpu()  # [S,Hp,Wp]
        if conf.dim() == 4 and conf.shape[-1] == 1:
            conf = conf.squeeze(-1)

        # ---- Convert VGGT z-depth to RADIAL ray depth ---- #
        # VGGT's depth head predicts z-depth (its unprojection is
        # x = (u - cx) * d / fx, z = d), while every consumer in this repo
        # (pointcloud *_depth_to_XYZ helpers, MotionMaskFromDepth,
        # TracksToTrajectories, the GS4D helpers) multiplies unit ray directions
        # by depth, i.e. expects RADIAL distance. Multiply by the per-pixel ray
        # norm sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2) using the per-frame
        # intrinsics at the VGGT processing resolution.
        fx_pf = intrinsic[:, 0, 0].clamp(min=1e-6).view(-1, 1, 1)  # [S,1,1]
        fy_pf = intrinsic[:, 1, 1].clamp(min=1e-6).view(-1, 1, 1)
        cx_pf = intrinsic[:, 0, 2].view(-1, 1, 1)
        cy_pf = intrinsic[:, 1, 2].view(-1, 1, 1)
        uu = torch.arange(Wp, dtype=torch.float32).view(1, 1, -1)
        vv = torch.arange(Hp, dtype=torch.float32).view(1, -1, 1)
        xn = (uu - cx_pf) / fx_pf
        yn = (vv - cy_pf) / fy_pf
        depth = depth * torch.sqrt(1.0 + xn * xn + yn * yn)

        if (Hp, Wp) != (H, W):
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            conf = F.interpolate(conf.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        # ---- If subsampled, expand back to the full frame count ---- #
        if sub_indices is not None:
            # Subsample indices are (near-)uniform over [0, T_full-1], so uniform
            # SE(3) resampling reconstructs per-frame poses well.
            trajectory = interpolate_se3(trajectory, T_full)
            all_t = torch.arange(T_full).unsqueeze(1)  # [T_full,1]
            nearest = (sub_indices.unsqueeze(0) - all_t).abs().argmin(dim=1)  # [T_full]
            depth = depth[nearest]
            conf = conf[nearest]

        return (trajectory, depth, horizontal_fov, conf)


class TrajectoryInvert:
    """
    Inverts each 4x4 matrix in a trajectory tensor, converting between
    world-to-camera and camera-to-world conventions.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # Trajectory: Tensor [K, 4, 4] (a single [4, 4] matrix also works)
                "trajectory": ("TENSOR", {"shape_hint": [None, 4, 4]}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "invert"
    CATEGORY = "Camera/Pose"
    DESCRIPTION = "Inverts each 4x4 pose (world-to-camera <-> camera-to-world)."

    def invert(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor]:
        traj = torch.as_tensor(trajectory, dtype=torch.float32)
        squeeze = traj.dim() == 2
        if squeeze:
            traj = traj.unsqueeze(0)
        if traj.dim() != 3 or traj.shape[-2:] != (4, 4):
            raise ValueError(f"Expected trajectory of shape [K,4,4], got {tuple(trajectory.shape)}")
        # Rigid-body inverse: R -> R.T, t -> -R.T @ t (numerically stabler than
        # a generic matrix inverse for SE(3) poses).
        R = traj[:, :3, :3]
        t = traj[:, :3, 3:4]
        Rt = R.transpose(1, 2)
        inv = torch.eye(4, dtype=traj.dtype).unsqueeze(0).repeat(traj.shape[0], 1, 1)
        inv[:, :3, :3] = Rt
        inv[:, :3, 3:4] = -Rt @ t
        if squeeze:
            inv = inv.squeeze(0)
        return (inv,)


class TrajectoryCompose:
    """
    Composes two trajectories per frame: out_k = A_k @ B_k. Either input may be
    a single [4,4] matrix, which is broadcast against the other. Useful for
    retargeting novel camera paths relative to a source pose (e.g. compose a
    relative path with the inverse of source pose 0).
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # Left operand: Tensor [K, 4, 4] or [4, 4]
                "trajectory_a": ("TENSOR", {"shape_hint": [None, 4, 4]}),
                # Right operand: Tensor [K, 4, 4] or [4, 4]
                "trajectory_b": ("TENSOR", {"shape_hint": [None, 4, 4]}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "compose"
    CATEGORY = "Camera/Pose"
    DESCRIPTION = "Per-frame matrix product A @ B; a single 4x4 input broadcasts over the other."

    def compose(self, trajectory_a: torch.Tensor, trajectory_b: torch.Tensor) -> Tuple[torch.Tensor]:
        A = torch.as_tensor(trajectory_a, dtype=torch.float32)
        B = torch.as_tensor(trajectory_b, dtype=torch.float32)
        both_single = A.dim() == 2 and B.dim() == 2
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if B.dim() == 2:
            B = B.unsqueeze(0)
        if A.dim() != 3 or A.shape[-2:] != (4, 4):
            raise ValueError(f"Expected trajectory_a of shape [K,4,4] or [4,4], got {tuple(trajectory_a.shape)}")
        if B.dim() != 3 or B.shape[-2:] != (4, 4):
            raise ValueError(f"Expected trajectory_b of shape [K,4,4] or [4,4], got {tuple(trajectory_b.shape)}")
        if A.shape[0] != B.shape[0] and A.shape[0] != 1 and B.shape[0] != 1:
            raise ValueError(
                f"Trajectory lengths do not broadcast: {A.shape[0]} vs {B.shape[0]} "
                "(they must match, or one must be a single 4x4 matrix)."
            )
        out = torch.matmul(A, B)  # broadcasts [1,4,4] against [K,4,4]
        if both_single:
            out = out.squeeze(0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "VideoPoseEstimator": VideoPoseEstimator,
    "TrajectoryInvert": TrajectoryInvert,
    "TrajectoryCompose": TrajectoryCompose,
}
