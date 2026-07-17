"""4D (dynamic) Gaussian splat nodes for camera-comfyUI.

Implements the ``GaussianSplats4D`` container (ComfyUI type string "GSPLAT4D")
plus nodes to build, render, save and load dynamic splat scenes:

- MotionMaskFromDepth: geometric motion segmentation from a depth sequence.
- EstimateTracks: CoTracker3 point tracking (lazy torch.hub load).
- TracksToTrajectories: lift 2D tracks + depth to 3D world trajectories.
- SplitSplatsByMask: split a splat cloud into inside/outside a 2D mask.
- BuildSplats4D: bind canonical splats to track control points (kNN blend).
- RenderSplats4DFrame / RenderSplats4DVideo: render at a time / over a path.
- SaveSplats4D / LoadSplats4D: .npz persistence.

Conventions match the rest of the repo: camera frame is +X right, +Y down,
+Z forward; pose matrices are 4x4 WORLD-TO-CAMERA acting on row vectors as
``cam = world @ R.T + t``; depth maps store RADIAL distance (Euclidean norm of
the camera-frame point), matching pointcloud_nodes' depth_to_XYZ helpers.
"""

import math
import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
                "folder_paths is unavailable; SaveSplats4D/LoadSplats4D require the ComfyUI runtime."
            )

    folder_paths = _FolderPathsStub()

try:
    from . import GS_nodes as _gs_nodes
except Exception:
    import GS_nodes as _gs_nodes

GaussianSplats = _gs_nodes.GaussianSplats
_concat_splats = _gs_nodes._concat_splats
_match_sh_orders = _gs_nodes._match_sh_orders
splat_cloud_rotation = _gs_nodes.splat_cloud_rotation
_stitch_splats = _gs_nodes._stitch_splats
_write_ply_splats = _gs_nodes._write_ply_splats
_resolve_device_choice = _gs_nodes._resolve_device_choice
_xyz_to_pinhole = _gs_nodes._xyz_to_pinhole
_xyz_to_fisheye = _gs_nodes._xyz_to_fisheye
_xyz_to_equirect = _gs_nodes._xyz_to_equirect
DEVICE_CHOICES = _gs_nodes.DEVICE_CHOICES


class Projection:
    PROJECTIONS = ["PINHOLE", "FISHEYE", "EQUIRECTANGULAR"]


RENDER_MODES_4D = ["auto", "gsplat", "fast", "over"]


# --------------------------------------------------------------------------- #
# Lazy accessors for helpers written by concurrent work packages (contracts). #
# --------------------------------------------------------------------------- #

def _render_gaussians(*args, **kwargs) -> tuple:
    """Resolve GS_nodes.render_gaussians (contract C2) at call time.

    Returns (image [1,H,W,3] float 0..1, alpha/mask [H,W], disparity [1,H,W,1]).
    """
    fn = getattr(_gs_nodes, "render_gaussians", None)
    if fn is None:
        raise RuntimeError(
            "GS_nodes.render_gaussians is unavailable. The 4D splat nodes require the shared "
            "render_gaussians helper in GS_nodes.py — please update camera-comfyUI to a version "
            "that includes it."
        )
    return fn(*args, **kwargs)


_POINTCLOUD_MODULE = None


def _interpolate_se3(trajectory: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Resolve pointcloud_nodes.interpolate_se3 (contract C1) at call time."""
    global _POINTCLOUD_MODULE
    if _POINTCLOUD_MODULE is None:
        try:
            from . import pointcloud_nodes as _pc
        except Exception:
            try:
                import pointcloud_nodes as _pc
            except Exception as exc:
                raise RuntimeError(
                    f"pointcloud_nodes could not be imported (needed for interpolate_se3): {exc}"
                ) from exc
        _POINTCLOUD_MODULE = _pc
    fn = getattr(_POINTCLOUD_MODULE, "interpolate_se3", None)
    if fn is None:
        raise RuntimeError(
            "pointcloud_nodes.interpolate_se3 is unavailable — please update camera-comfyUI to a "
            "version that includes it."
        )
    return fn(trajectory, num_steps)


_COTRACKER_CACHE: Dict[str, Any] = {}


def _load_cotracker(device: torch.device):
    """Load (and cache) the CoTracker3 offline model via torch.hub."""
    key = str(device)
    model = _COTRACKER_CACHE.get(key)
    if model is not None:
        return model
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CoTracker3 via torch.hub. The first run needs internet access to "
            "download the facebookresearch/co-tracker repo and its checkpoint into the torch hub "
            "cache. If this machine is offline, pre-populate the cache on a connected machine "
            "with: python -c \"import torch; torch.hub.load('facebookresearch/co-tracker', "
            f"'cotracker3_offline')\". Original error: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    _COTRACKER_CACHE[key] = model
    return model


# --------------------------------------------------------------------------- #
# Small math / coercion helpers.                                              #
# --------------------------------------------------------------------------- #

def _coerce_matrix4x4(matrix, device: torch.device) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        return matrix.to(device=device, dtype=torch.float32).view(4, 4)
    return torch.tensor(matrix, device=device, dtype=torch.float32).view(4, 4)


def _coerce_trajectory(trajectory, device: torch.device) -> torch.Tensor:
    """Coerce a tensor / nested list to [K,4,4] float32 on device."""
    if isinstance(trajectory, torch.Tensor):
        traj = trajectory
    else:
        traj = torch.tensor(trajectory, dtype=torch.float32)
    traj = traj.to(device=device, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(f"Expected trajectory of shape [K,4,4], got {tuple(traj.shape)}")
    return traj


def _coerce_depth_seq(depth) -> torch.Tensor:
    """Coerce depth input to [T,H,W] float32 (accepts [H,W], [T,H,W,1], [T,1,H,W])."""
    d = depth if isinstance(depth, torch.Tensor) else torch.tensor(depth, dtype=torch.float32)
    d = d.float()
    if d.dim() == 2:
        d = d.unsqueeze(0)
    if d.dim() == 4 and d.shape[-1] == 1:
        d = d.squeeze(-1)
    elif d.dim() == 4 and d.shape[1] == 1:
        d = d.squeeze(1)
    if d.dim() != 3:
        raise ValueError(f"Expected depth sequence of shape [T,H,W], got {tuple(d.shape)}")
    return d


def _uv_grid(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized [-1,1] pixel-center grid matching pointcloud_nodes' linspace convention."""
    u = torch.linspace(-1.0, 1.0, width, device=device).unsqueeze(0).expand(height, width)
    v = torch.linspace(-1.0, 1.0, height, device=device).unsqueeze(1).expand(height, width)
    return u, v


def _uv_in_fov(u: torch.Tensor, v: torch.Tensor, projection: str) -> torch.Tensor:
    """True where normalized uv lies inside the projection's actual image region.

    For FISHEYE the [-1,1] square contains the corners beyond the image circle
    (r > 1, i.e. view angles beyond fov/2); pixels there carry no scene content
    (black corners / garbage depth) and must not be lifted, warped or tracked.
    """
    if projection == "FISHEYE":
        return (u * u + v * v) <= 1.0 + 1e-6
    return torch.ones_like(u, dtype=torch.bool)


def _uv_to_dirs(u: torch.Tensor, v: torch.Tensor, projection: str, horizontal_fov: float) -> torch.Tensor:
    """Unit ray directions [...,3] in camera frame for normalized uv in [-1,1].

    Inverse of GS_nodes' _xyz_to_pinhole/_xyz_to_fisheye/_xyz_to_equirect (which are what
    render_gaussians uses), so unprojection and splat rendering stay self-consistent.
    Multiply by RADIAL depth to obtain camera-frame points.
    """
    fov_rad = math.radians(horizontal_fov)
    if projection == "PINHOLE":
        f = 1.0 / math.tan(fov_rad / 2.0)
        dirs = torch.stack([u, v, torch.full_like(u, f)], dim=-1)
        return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    if projection == "FISHEYE":
        r = torch.sqrt(u * u + v * v).clamp(max=1.0)
        theta = r * (fov_rad / 2.0)
        phi = torch.atan2(v, u)
        sin_t = torch.sin(theta)
        return torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), torch.cos(theta)], dim=-1)
    if projection == "EQUIRECTANGULAR":
        lon = u * (fov_rad / 2.0)
        lat = v * (math.pi / 2.0)
        cos_lat = torch.cos(lat)
        return torch.stack([cos_lat * torch.sin(lon), torch.sin(lat), cos_lat * torch.cos(lon)], dim=-1)
    raise ValueError(f"Unsupported projection: {projection}")


def _project_xyz(
    X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, projection: str, horizontal_fov: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Camera-frame XYZ -> normalized uv in [-1,1] + radial depth (GS_nodes convention)."""
    if projection == "PINHOLE":
        return _xyz_to_pinhole(X, Y, Z, horizontal_fov)
    if projection == "FISHEYE":
        return _xyz_to_fisheye(X, Y, Z, horizontal_fov)
    if projection == "EQUIRECTANGULAR":
        return _xyz_to_equirect(X, Y, Z, horizontal_fov)
    raise ValueError(f"Unsupported projection: {projection}")


def _projection_valid(
    u: torch.Tensor, v: torch.Tensor, Z: torch.Tensor, projection: str
) -> torch.Tensor:
    """In-image validity for projected points; pinhole additionally requires Z>0,
    fisheye requires the point inside the image circle (r <= 1), not just the
    [-1,1] square — angles beyond fov/2 can otherwise land in the corners."""
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & (u >= -1.0)
        & (u <= 1.0)
        & (v >= -1.0)
        & (v <= 1.0)
    )
    if projection == "PINHOLE":
        valid = valid & (Z > 1e-6)
    elif projection == "FISHEYE":
        valid = valid & ((u * u + v * v) <= 1.0 + 1e-6)
    return valid


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, w: float) -> torch.Tensor:
    """Batched quaternion SLERP between [N,4] wxyz quats with scalar blend w in [0,1]."""
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near = sin_theta < 1e-5
    safe_sin = sin_theta.clamp(min=1e-12)
    w0 = torch.where(near, torch.full_like(theta, 1.0 - w), torch.sin((1.0 - w) * theta) / safe_sin)
    w1 = torch.where(near, torch.full_like(theta, w), torch.sin(w * theta) / safe_sin)
    out = w0 * q0 + w1 * q1
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _median_filter_time(traj: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Median-filter [T,...] over the time axis with replicate padding (kills depth spikes)."""
    T = traj.shape[0]
    if T < 3 or kernel < 3:
        return traj
    pad = kernel // 2
    first = traj[:1].expand(pad, *traj.shape[1:])
    last = traj[-1:].expand(pad, *traj.shape[1:])
    padded = torch.cat([first, traj, last], dim=0)
    windows = padded.unfold(0, kernel, 1)  # [T, ..., kernel]
    return windows.median(dim=-1).values


def _fill_invalid_linear(traj: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fill invalid timesteps of [T,M,3] trajectories by linear interp over time (hold ends).

    Fully vectorized (sort + searchsorted + gather): a per-track Python loop with
    np.interp would serialize M GPU syncs / host round-trips and stall for
    seconds-to-minutes at realistic track counts (grid_size 100-200).

    Returns (filled [T,M,3], track_ok [M] float — 0.0 for tracks with no valid sample).
    """
    T, M = valid.shape
    device = traj.device
    dtype = traj.dtype
    n_valid = valid.sum(dim=0)  # [M]
    track_ok = (n_valid > 0).to(dtype)
    if M == 0 or bool((n_valid == T).all()):
        return traj.clone(), track_ok

    # Sort each track's valid timesteps to the front (invalid -> sentinel T).
    tt = torch.arange(T, device=device).unsqueeze(1).expand(T, M)
    key = torch.where(valid, tt, torch.full_like(tt, T))  # [T,M]
    key_sorted, perm = key.sort(dim=0)  # valid times ascending, sentinels last
    vals_sorted = torch.gather(traj, 0, perm.unsqueeze(-1).expand(T, M, 3))  # [T,M,3]

    ks = key_sorted.transpose(0, 1).contiguous()  # [M,T]
    q = tt.transpose(0, 1).contiguous()  # [M,T] query times 0..T-1 per track
    # First valid-time index >= t per (track, time); clamp into the valid range
    # so out-of-range queries hold the first/last valid sample (np.interp-style).
    pos = torch.searchsorted(ks, q)  # [M,T]
    n_ix = (n_valid - 1).clamp(min=0).unsqueeze(1)  # [M,1]
    i1 = torch.minimum(pos.clamp(max=T - 1), n_ix)
    i0 = torch.minimum((pos - 1).clamp(min=0), n_ix)

    vals_mt = vals_sorted.permute(1, 0, 2)  # [M,T,3]
    t0 = torch.gather(ks, 1, i0).to(dtype)
    t1 = torch.gather(ks, 1, i1).to(dtype)
    v0 = torch.gather(vals_mt, 1, i0.unsqueeze(-1).expand(M, T, 3))
    v1 = torch.gather(vals_mt, 1, i1.unsqueeze(-1).expand(M, T, 3))
    denom = t1 - t0
    w = torch.where(denom > 0, (q.to(dtype) - t0) / denom.clamp(min=1e-12), torch.zeros_like(denom))
    w = w.clamp(0.0, 1.0)
    filled = (v0 + w.unsqueeze(-1) * (v1 - v0)).transpose(0, 1)  # [T,M,3]

    # Keep original samples at valid timesteps; zero tracks with no valid sample.
    out = torch.where(valid.unsqueeze(-1), traj, filled)
    out = torch.where((n_valid > 0).view(1, M, 1), out, torch.zeros_like(out))
    return out, track_ok


# --------------------------------------------------------------------------- #
# Contract C3: the 4D splat container.                                        #
# --------------------------------------------------------------------------- #

@dataclass
class GaussianSplats4D:
    """Dynamic Gaussian splat scene (ComfyUI type "GSPLAT4D").

    static: time-invariant splats in world frame (may be None).
    canonical: N dynamic splats at the reference time (world frame).
    trajectories: [T, N, 3] absolute world xyz per timestep.
    times: [T] float, monotonically increasing, normalized 0..1.
    rotations: optional [T, N, 4] wxyz quats (None -> use canonical rotations).
    """

    static: Optional[GaussianSplats]
    canonical: GaussianSplats
    trajectories: torch.Tensor
    times: torch.Tensor
    rotations: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "GaussianSplats4D":
        return GaussianSplats4D(
            static=self.static.to(device) if self.static is not None else None,
            canonical=self.canonical.to(device),
            trajectories=self.trajectories.to(device),
            times=self.times.to(device),
            rotations=self.rotations.to(device) if self.rotations is not None else None,
        )

    def at_time(self, t: float) -> GaussianSplats:
        """Evaluate the scene at time t (clamped to [times[0], times[-1]]).

        Linear interpolation of xyz (SLERP for rotations if present) between the
        bracketing timesteps, substituted into a clone of the canonical splats and
        concatenated with the static splats.
        """
        times = self.times.reshape(-1).float()
        T = int(times.shape[0])
        if self.trajectories.shape[0] != T:
            raise ValueError(
                f"trajectories has {self.trajectories.shape[0]} timesteps but times has {T}"
            )
        dyn = self.canonical.clone()
        if T == 1:
            xyz = self.trajectories[0]
            rot = self.rotations[0] if self.rotations is not None else None
        else:
            t_clamped = min(max(float(t), float(times[0])), float(times[-1]))
            probe = torch.tensor(t_clamped, dtype=times.dtype, device=times.device)
            hi = int(torch.searchsorted(times, probe, right=True).item())
            hi = min(max(hi, 1), T - 1)
            lo = hi - 1
            t0 = float(times[lo])
            t1 = float(times[hi])
            w = 0.0 if t1 <= t0 else (t_clamped - t0) / (t1 - t0)
            xyz = (1.0 - w) * self.trajectories[lo] + w * self.trajectories[hi]
            if self.rotations is not None:
                rot = _quat_slerp(self.rotations[lo], self.rotations[hi], w)
            else:
                rot = None
        dyn.xyz = xyz.to(device=dyn.xyz.device, dtype=dyn.xyz.dtype)
        if rot is not None:
            dyn.rotation = rot.to(device=dyn.rotation.device, dtype=dyn.rotation.dtype)
        if self.static is not None and len(self.static) > 0:
            static = self.static
            if static.xyz.device != dyn.xyz.device:
                static = static.to(dyn.xyz.device)
            return _concat_splats([static, dyn])
        return dyn


# --------------------------------------------------------------------------- #
# npz (de)serialization helpers.                                              #
# --------------------------------------------------------------------------- #

def _pack_splats_npz(arrays: Dict[str, np.ndarray], prefix: str, splats: GaussianSplats) -> None:
    arrays[f"{prefix}_xyz"] = splats.xyz.detach().cpu().float().numpy()
    arrays[f"{prefix}_scale"] = splats.scale.detach().cpu().float().numpy()
    arrays[f"{prefix}_rotation"] = splats.rotation.detach().cpu().float().numpy()
    arrays[f"{prefix}_opacity"] = splats.opacity.detach().cpu().float().numpy()
    arrays[f"{prefix}_f_dc"] = splats.f_dc.detach().cpu().float().numpy()
    arrays[f"{prefix}_f_rest"] = splats.f_rest.detach().cpu().float().numpy()
    arrays[f"{prefix}_sh_order"] = np.asarray(int(splats.sh_order))


def _unpack_splats_npz(data, prefix: str) -> Optional[GaussianSplats]:
    if f"{prefix}_xyz" not in data:
        return None
    return GaussianSplats(
        xyz=torch.from_numpy(np.asarray(data[f"{prefix}_xyz"], dtype=np.float32)),
        scale=torch.from_numpy(np.asarray(data[f"{prefix}_scale"], dtype=np.float32)),
        rotation=torch.from_numpy(np.asarray(data[f"{prefix}_rotation"], dtype=np.float32)),
        opacity=torch.from_numpy(np.asarray(data[f"{prefix}_opacity"], dtype=np.float32)),
        f_dc=torch.from_numpy(np.asarray(data[f"{prefix}_f_dc"], dtype=np.float32)),
        f_rest=torch.from_numpy(np.asarray(data[f"{prefix}_f_rest"], dtype=np.float32)),
        sh_order=int(np.asarray(data[f"{prefix}_sh_order"])),
    )


# --------------------------------------------------------------------------- #
# Nodes.                                                                      #
# --------------------------------------------------------------------------- #

class MotionMaskFromDepth:
    """Flags dynamic pixels by warping depth between frames and thresholding the residual."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth_seq": ("TENSOR", {"tooltip": "Depth sequence [T,H,W] (radial distance)."}),
                "trajectory": (
                    "TENSOR",
                    {"tooltip": "World-to-camera poses [T,4,4] (or [K,4,4]; interpolated to T)."},
                ),
                "input_projection": (Projection.PROJECTIONS, {}),
                "input_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Relative depth residual |d_proj - d_sampled|/d_sampled above which a pixel is dynamic.",
                    },
                ),
                "frame_gap": (
                    "INT",
                    {"default": 4, "min": 1, "max": 256, "tooltip": "Temporal gap for forward/backward reprojection checks."},
                ),
                "dilate": ("INT", {"default": 2, "min": 0, "max": 64, "tooltip": "Dilation radius (pixels) of the dynamic mask."}),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("motion_mask",)
    FUNCTION = "motion_mask"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Detects dynamic pixels from a depth+pose sequence (1.0 = moving)."

    @torch.no_grad()
    def motion_mask(
        self,
        depth_seq: torch.Tensor,
        trajectory: torch.Tensor,
        input_projection: str,
        input_horizontal_fov: float,
        threshold: float = 0.10,
        frame_gap: int = 4,
        dilate: int = 2,
        device: str = "auto",
    ) -> Tuple[torch.Tensor]:
        target_device = _resolve_device_choice(device)
        depth = _coerce_depth_seq(depth_seq).to(target_device)
        T, H, W = depth.shape
        poses = _coerce_trajectory(trajectory, target_device)
        if poses.shape[0] != T:
            poses = _coerce_trajectory(_interpolate_se3(poses, T), target_device)

        u, v = _uv_grid(H, W, target_device)
        dirs = _uv_to_dirs(u, v, input_projection, input_horizontal_fov)  # [H,W,3]
        in_fov = _uv_in_fov(u, v, input_projection)  # excludes fisheye corners
        gap = max(1, int(frame_gap))
        dynamic = torch.zeros((T, H, W), device=target_device)

        for t in tqdm(range(T), desc="MotionMaskFromDepth"):
            depth_t = depth[t]
            cam_pts = depth_t.unsqueeze(-1) * dirs  # [H,W,3]
            R_t = poses[t, :3, :3]
            tr_t = poses[t, :3, 3]
            # world = (cam - t) @ R  (inverse of cam = world @ R.T + t)
            world = (cam_pts - tr_t) @ R_t
            has_depth = (depth_t > 1e-6) & in_fov

            flagged = torch.zeros((H, W), dtype=torch.bool, device=target_device)
            for t2 in (t + gap, t - gap):
                if t2 < 0 or t2 >= T:
                    continue
                R2 = poses[t2, :3, :3]
                tr2 = poses[t2, :3, 3]
                cam2 = world @ R2.T + tr2
                X2, Y2, Z2 = cam2.unbind(-1)
                u2, v2, d2 = _project_xyz(X2, Y2, Z2, input_projection, input_horizontal_fov)
                valid = _projection_valid(u2, v2, Z2, input_projection) & has_depth
                grid = torch.stack(
                    [u2.nan_to_num(0.0), v2.nan_to_num(0.0)], dim=-1
                ).view(1, H, W, 2)
                sampled = F.grid_sample(
                    depth[t2].view(1, 1, H, W),
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=True,
                ).view(H, W)
                residual = (d2 - sampled).abs() / sampled.clamp(min=1e-6)
                flagged |= valid & (sampled > 1e-6) & (residual > threshold)
            dynamic[t] = flagged.float()

        if dilate > 0:
            kernel = 2 * int(dilate) + 1
            dynamic = F.max_pool2d(
                dynamic.unsqueeze(1), kernel_size=kernel, stride=1, padding=int(dilate)
            ).squeeze(1)
        return (dynamic,)


class EstimateTracks:
    """Dense grid point tracking with CoTracker3 (offline model, loaded via torch.hub)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video frames [T,H,W,3] float 0..1."}),
                "grid_size": (
                    "INT",
                    {"default": 20, "min": 1, "max": 200, "tooltip": "Tracks a grid_size x grid_size point grid."},
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("tracks", "visibility")
    FUNCTION = "estimate_tracks"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Runs CoTracker3 on a video; returns tracks [T,N,2] (pixels) and visibility [T,N]."

    @torch.no_grad()
    def estimate_tracks(
        self,
        frames: torch.Tensor,
        grid_size: int = 20,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_device = _resolve_device_choice(device)
        if frames.dim() != 4:
            raise ValueError(f"Expected frames [T,H,W,3], got {tuple(frames.shape)}")
        model = _load_cotracker(target_device)

        H, W = int(frames.shape[1]), int(frames.shape[2])
        video = frames[..., :3].permute(0, 3, 1, 2).float()  # [T,3,H,W], source device (usually CPU)
        # CoTracker3 internally resizes the clip to its interp_shape (~384x512)
        # anyway, so pre-resize BEFORE the device upload instead of shipping the
        # full-resolution video to the GPU as one [1,T,3,H,W] tensor (~5GB for
        # 200 frames at 1080p -> OOM before tracking starts). Track coordinates
        # are scaled back to the input resolution afterwards.
        interp = getattr(model, "interp_shape", (384, 512))
        interp_h, interp_w = int(interp[0]), int(interp[1])
        scale_x = scale_y = 1.0
        if H * W > interp_h * interp_w:
            video = F.interpolate(video, size=(interp_h, interp_w), mode="bilinear", align_corners=False)
            scale_x = float(W - 1) / float(max(interp_w - 1, 1))
            scale_y = float(H - 1) / float(max(interp_h - 1, 1))
        video = video.unsqueeze(0).to(target_device)
        if video.max().item() <= 1.0:
            video = video * 255.0

        pred_tracks, pred_visibility = model(video, grid_size=int(grid_size))
        tracks = pred_tracks[0].float()  # [T,N,2] pixel (x,y)
        if scale_x != 1.0 or scale_y != 1.0:
            tracks = tracks * torch.tensor([scale_x, scale_y], device=tracks.device, dtype=tracks.dtype)
        visibility = pred_visibility[0].float()  # [T,N]
        if visibility.dim() == 3:
            visibility = visibility.squeeze(-1)
        return (tracks, visibility)


class TracksToTrajectories:
    """Lifts 2D tracks + depth to 3D world-space trajectories."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TENSOR", {"tooltip": "Pixel tracks [T,N,2] (x,y) from EstimateTracks."}),
                "visibility": ("TENSOR", {"tooltip": "Track visibility [T,N] (0/1)."}),
                "depth_seq": ("TENSOR", {"tooltip": "Depth sequence [T,H,W] (radial distance)."}),
                "input_projection": (Projection.PROJECTIONS, {}),
                "input_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "min_visible_frac": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Drop tracks visible in fewer than this fraction of frames.",
                    },
                ),
            },
            "optional": {
                "trajectory": (
                    "TENSOR",
                    {"tooltip": "World-to-camera poses [T,4,4]. Default: identity (static camera)."},
                ),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("trajectories3d", "track_valid")
    FUNCTION = "tracks_to_trajectories"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Unprojects 2D tracks with depth and camera poses into world-space 3D trajectories [T,M,3]."

    @torch.no_grad()
    def tracks_to_trajectories(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        depth_seq: torch.Tensor,
        input_projection: str,
        input_horizontal_fov: float,
        min_visible_frac: float = 0.5,
        trajectory: Optional[torch.Tensor] = None,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_device = _resolve_device_choice(device)
        tracks = tracks.to(target_device).float()
        if tracks.dim() != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"Expected tracks [T,N,2], got {tuple(tracks.shape)}")
        visibility = visibility.to(target_device).float()
        if visibility.dim() == 3:
            visibility = visibility.squeeze(-1)
        depth = _coerce_depth_seq(depth_seq).to(target_device)
        T, N = tracks.shape[:2]
        if depth.shape[0] != T or visibility.shape != (T, N):
            raise ValueError(
                f"Shape mismatch: tracks [T={T},N={N}], visibility {tuple(visibility.shape)}, "
                f"depth {tuple(depth.shape)}"
            )
        H, W = depth.shape[1:]

        vis = visibility > 0.5
        frac = vis.float().mean(dim=0)
        keep = frac >= float(min_visible_frac)
        if not bool(keep.any()):
            raise ValueError(
                "No tracks meet min_visible_frac; lower the threshold or check the visibility input."
            )
        tracks = tracks[:, keep]
        vis = vis[:, keep]
        M = tracks.shape[1]

        # Pixel -> normalized [-1,1] (align_corners=True convention).
        u = 2.0 * tracks[..., 0] / max(W - 1, 1) - 1.0
        v = 2.0 * tracks[..., 1] / max(H - 1, 1) - 1.0

        # Sample depth at track locations (nearest to avoid mixing fg/bg at edges).
        grid = torch.stack([u, v], dim=-1).view(T, 1, M, 2)
        d = F.grid_sample(
            depth.view(T, 1, H, W),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).view(T, M)

        dirs = _uv_to_dirs(u, v, input_projection, input_horizontal_fov)  # [T,M,3]
        cam = d.unsqueeze(-1) * dirs

        if trajectory is None:
            poses = torch.eye(4, device=target_device).unsqueeze(0).expand(T, 4, 4).contiguous()
        else:
            poses = _coerce_trajectory(trajectory, target_device)
            if poses.shape[0] != T:
                poses = _coerce_trajectory(_interpolate_se3(poses, T), target_device)
        R = poses[:, :3, :3]
        tr = poses[:, :3, 3]
        # world = (cam - t) @ R  per frame.
        world = torch.bmm(cam - tr.unsqueeze(1), R)

        in_bounds = (
            (u >= -1.0) & (u <= 1.0) & (v >= -1.0) & (v <= 1.0)
            & _uv_in_fov(u, v, input_projection)
        )
        valid = vis & in_bounds & (d > 1e-6) & torch.isfinite(world).all(dim=-1)
        world = torch.where(valid.unsqueeze(-1), world, torch.zeros_like(world))

        filled, track_ok = _fill_invalid_linear(world, valid)
        filled = _median_filter_time(filled, kernel=3)
        return (filled, track_ok)


class SplitSplatsByMask:
    """Splits splats into (inside, outside) by projecting centers into a 2D mask."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats": ("GSPLAT",),
                "mask": ("MASK", {"tooltip": "Mask [H,W] (or [1,H,W]) in the camera view."}),
                "projection": (Projection.PROJECTIONS, {}),
                "horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "camera_matrix": (
                    "MAT_4X4",
                    {"tooltip": "World-to-camera matrix of the mask's view. Default: identity (splats already in camera frame)."},
                ),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT", "GSPLAT")
    RETURN_NAMES = ("inside_splats", "outside_splats")
    FUNCTION = "split_splats"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Projects splat centers into a mask; returns splats inside vs outside. Splats behind the camera go to outside."

    @torch.no_grad()
    def split_splats(
        self,
        splats: GaussianSplats,
        mask: torch.Tensor,
        projection: str,
        horizontal_fov: float,
        threshold: float = 0.5,
        camera_matrix=None,
        device: str = "auto",
    ) -> Tuple[GaussianSplats, GaussianSplats]:
        target_device = _resolve_device_choice(device)
        if splats.xyz.device != target_device:
            splats = splats.to(target_device)

        m = mask
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.float32)
        m = m.to(target_device).float()
        if m.dim() == 4:  # [B,H,W,C]
            m = m[0, ..., 0]
        elif m.dim() == 3:  # [B,H,W]
            m = m[0]
        if m.dim() != 2:
            raise ValueError(f"Expected mask [H,W] or [1,H,W], got {tuple(mask.shape)}")
        H, W = m.shape

        if camera_matrix is None:
            M = torch.eye(4, device=target_device)
        else:
            M = _coerce_matrix4x4(camera_matrix, target_device)
        R = M[:3, :3]
        t = M[:3, 3]
        cam = splats.xyz @ R.T + t
        X, Y, Z = cam.unbind(-1)
        u, v, _ = _project_xyz(X, Y, Z, projection, horizontal_fov)
        valid = _projection_valid(u, v, Z, projection)

        inside = torch.zeros((cam.shape[0],), dtype=torch.bool, device=target_device)
        if bool(valid.any()):
            uv_u = u[valid]
            uv_v = v[valid]
            px = ((uv_u * 0.5 + 0.5) * (W - 1)).round().long().clamp(0, W - 1)
            py = ((uv_v * 0.5 + 0.5) * (H - 1)).round().long().clamp(0, H - 1)
            inside[valid] = m[py, px] > float(threshold)
        return (splats[inside], splats[~inside])


class BuildSplats4D:
    """Binds canonical splats to 3D track control points via kNN linear blending."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "canonical": ("GSPLAT", {"tooltip": "Dynamic splats (world frame) at the reference timestep."}),
                "trajectories3d": ("TENSOR", {"tooltip": "Control-point trajectories [T,M,3] in world space."}),
                "reference_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100000, "tooltip": "Timestep the canonical splats correspond to."},
                ),
                "knn": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "Number of nearest control points per splat."}),
                "rbf_gamma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.1,
                        "tooltip": "0 = inverse-distance weights; >0 = RBF weights exp(-gamma*d^2).",
                    },
                ),
            },
            "optional": {
                "static": ("GSPLAT", {"tooltip": "Time-invariant splats (world frame)."}),
                "times": ("TENSOR", {"tooltip": "Timestamps [T], normalized 0..1. Default: linspace."}),
                "track_valid": ("TENSOR", {"tooltip": "Per-track validity [M] from TracksToTrajectories."}),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT4D",)
    RETURN_NAMES = ("splats4d",)
    FUNCTION = "build_splats4d"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Builds a 4D splat scene: each canonical splat follows a kNN blend of track control-point motions."

    @torch.no_grad()
    def build_splats4d(
        self,
        canonical: GaussianSplats,
        trajectories3d: torch.Tensor,
        reference_index: int = 0,
        knn: int = 4,
        rbf_gamma: float = 0.0,
        static: Optional[GaussianSplats] = None,
        times: Optional[torch.Tensor] = None,
        track_valid: Optional[torch.Tensor] = None,
        device: str = "auto",
    ) -> Tuple[GaussianSplats4D]:
        target_device = _resolve_device_choice(device)
        if canonical.xyz.device != target_device:
            canonical = canonical.to(target_device)

        traj = trajectories3d
        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj, dtype=torch.float32)
        traj = traj.to(device=target_device, dtype=torch.float32)
        if traj.dim() != 3 or traj.shape[-1] != 3:
            raise ValueError(f"Expected trajectories3d [T,M,3], got {tuple(traj.shape)}")

        if track_valid is not None:
            tv = track_valid
            if not isinstance(tv, torch.Tensor):
                tv = torch.tensor(tv)
            tv = tv.to(target_device).reshape(-1)
            if tv.shape[0] != traj.shape[1]:
                raise ValueError(
                    f"track_valid has {tv.shape[0]} entries but trajectories3d has {traj.shape[1]} tracks"
                )
            traj = traj[:, tv > 0.5]

        T, M = traj.shape[0], traj.shape[1]
        if M == 0:
            raise ValueError("No valid tracks remain after filtering; cannot build a 4D scene.")

        ref = int(min(max(int(reference_index), 0), T - 1))
        k = max(1, min(int(knn), M))
        ctrl_ref = traj[ref]  # [M,3]
        deltas = traj - ctrl_ref.unsqueeze(0)  # [T,M,3] motion relative to reference

        N = len(canonical)
        if N == 0:
            raise ValueError("Canonical splats are empty.")
        canon_xyz = canonical.xyz.to(torch.float32)
        # The eager [T,N,3] trajectory tensor can be huge (2.4GB for 2M splats at
        # T=100); keep it on the CPU so it does not pin VRAM for the lifetime of
        # the GSPLAT4D object. at_time() only moves the interpolated [N,3] slice
        # to the render device per frame.
        trajectories = torch.empty((T, N, 3), device="cpu", dtype=torch.float32)
        chunk = 16384
        for start in tqdm(range(0, N, chunk), desc="BuildSplats4D kNN binding"):
            end = min(N, start + chunk)
            xyz_c = canon_xyz[start:end]  # [n,3]
            dists = torch.cdist(xyz_c, ctrl_ref)  # [n,M]
            d_k, idx_k = torch.topk(dists, k, dim=1, largest=False)  # [n,k]
            if rbf_gamma > 0.0:
                # Subtract the per-row min in the exponent (softmax-style) so weights
                # never underflow to zero before normalization; the normalized result
                # is mathematically identical.
                sq = d_k * d_k
                w = torch.exp(-float(rbf_gamma) * (sq - sq.min(dim=1, keepdim=True).values))
            else:
                w = 1.0 / (d_k + 1e-8)
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
            delta_k = deltas[:, idx_k, :]  # [T,n,k,3]
            disp = torch.einsum("nk,tnkc->tnc", w, delta_k)  # [T,n,3]
            trajectories[:, start:end] = (xyz_c.unsqueeze(0) + disp).cpu()

        if times is None:
            times_t = torch.linspace(0.0, 1.0, T, device=target_device)
        else:
            times_t = times if isinstance(times, torch.Tensor) else torch.tensor(times)
            times_t = times_t.to(device=target_device, dtype=torch.float32).reshape(-1)
            if times_t.shape[0] != T:
                raise ValueError(f"times has {times_t.shape[0]} entries but trajectories3d has {T} timesteps")

        static_splats = None
        if static is not None:
            static_splats = static.to(target_device) if static.xyz.device != target_device else static
            if static_splats.sh_order != canonical.sh_order:
                # Harmonize SH orders now (zero-pad the lower-order cloud) so
                # at_time's _concat_splats does not fail with "All splats must
                # have the same SH order" at render/save time — e.g. an
                # sh_order-3 static from LoadPlySplat with an sh_order-0
                # SHARP-derived canonical.
                static_splats, canonical = _match_sh_orders(static_splats, canonical)

        splats4d = GaussianSplats4D(
            static=static_splats,
            canonical=canonical,
            trajectories=trajectories,
            times=times_t,
            rotations=None,
        )
        return (splats4d,)


class RenderSplats4DFrame:
    """Renders a 4D splat scene at a single time value."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats4d": ("GSPLAT4D",),
                "camera_matrix": ("MAT_4X4",),
                "camera_projection": (Projection.PROJECTIONS, {}),
                "camera_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "output_height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "render_mode": (RENDER_MODES_4D, {"default": "auto"}),
                "max_splats": ("INT", {"default": 0, "min": 0, "max": 10000000, "tooltip": "0 = unlimited."}),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("image", "mask", "disparity")
    FUNCTION = "render_frame"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Evaluates the 4D scene at a time value and renders it from the given camera."

    @torch.no_grad()
    def render_frame(
        self,
        splats4d: GaussianSplats4D,
        camera_matrix,
        camera_projection: str,
        camera_horizontal_fov: float,
        time: float = 0.0,
        output_width: int = 512,
        output_height: int = 512,
        render_mode: str = "auto",
        max_splats: int = 0,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        splats = splats4d.at_time(float(time))
        image, alpha, disparity = _render_gaussians(
            splats,
            camera_matrix,
            camera_projection,
            camera_horizontal_fov,
            output_width,
            output_height,
            max_splats=max_splats,
            render_mode=render_mode,
            device=device,
        )
        return (image, alpha, disparity)


class RenderSplats4DVideo:
    """Renders a 4D splat scene along a camera path over a time range."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats4d": ("GSPLAT4D",),
                "trajectory": ("TENSOR", {"tooltip": "Camera path [K,4,4] world-to-camera; interpolated to num_frames."}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 4096}),
                "time_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "camera_projection": (Projection.PROJECTIONS, {}),
                "camera_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "output_width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "output_height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "render_mode": (RENDER_MODES_4D, {"default": "auto"}),
            },
            "optional": {
                "max_splats": ("INT", {"default": 0, "min": 0, "max": 10000000, "tooltip": "0 = unlimited."}),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("images", "masks", "disparity")
    FUNCTION = "render_video"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Interpolates the camera path, sweeps time from time_start to time_end and renders each frame."

    @torch.no_grad()
    def render_video(
        self,
        splats4d: GaussianSplats4D,
        trajectory: torch.Tensor,
        num_frames: int = 49,
        time_start: float = 0.0,
        time_end: float = 1.0,
        camera_projection: str = "PINHOLE",
        camera_horizontal_fov: float = 90.0,
        output_width: int = 512,
        output_height: int = 512,
        render_mode: str = "auto",
        max_splats: int = 0,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = max(1, int(num_frames))
        poses = _coerce_trajectory(trajectory, torch.device("cpu"))
        if poses.shape[0] != num_frames:
            poses = _coerce_trajectory(_interpolate_se3(poses, num_frames), torch.device("cpu"))
        if num_frames == 1:
            time_values = [float(time_start)]
        else:
            time_values = torch.linspace(float(time_start), float(time_end), num_frames).tolist()

        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        disparities: List[torch.Tensor] = []
        for i in tqdm(range(num_frames), desc="RenderSplats4DVideo"):
            splats = splats4d.at_time(time_values[i])
            image, alpha, disparity = _render_gaussians(
                splats,
                poses[i],
                camera_projection,
                camera_horizontal_fov,
                output_width,
                output_height,
                max_splats=max_splats,
                render_mode=render_mode,
                device=device,
            )
            images.append(image.detach().cpu())  # [1,H,W,3]
            masks.append(alpha.detach().cpu())  # [H,W]
            disparities.append(disparity.detach().cpu())  # [1,H,W,1]

        images_out = torch.cat(images, dim=0)  # [F,H,W,3]
        masks_out = torch.stack(masks, dim=0)  # [F,H,W]
        disparity_out = torch.cat(disparities, dim=0)  # [F,H,W,1]
        return (images_out, masks_out, disparity_out)


class SaveSplats4D:
    """Saves a 4D splat scene to the ComfyUI output directory as .npz."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "splat4d"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats4d": ("GSPLAT4D",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUISplat4D",
                        "tooltip": "Prefix for the .npz file. You can include format-tokens like %date:yyyy-MM-dd%.",
                    },
                ),
                "export_ply_frames": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Also write one 3DGS .ply per timestep (evaluated via at_time)."},
                ),
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_splats4d"
    OUTPUT_NODE = True
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Saves the GSPLAT4D scene as an .npz archive (plus optional per-frame PLYs)."

    def save_splats4d(
        self,
        splats4d: GaussianSplats4D,
        filename_prefix: str,
        export_ply_frames: bool = False,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                0, 0
            )
        os.makedirs(full_output_folder, exist_ok=True)
        base_name = filename.replace("%batch_num%", "0")
        npz_name = f"{base_name}_{counter:05}.npz"
        npz_path = os.path.join(full_output_folder, npz_name)

        arrays: Dict[str, np.ndarray] = {}
        _pack_splats_npz(arrays, "canonical", splats4d.canonical)
        arrays["trajectories"] = splats4d.trajectories.detach().cpu().float().numpy()
        arrays["times"] = splats4d.times.detach().cpu().float().reshape(-1).numpy()
        if splats4d.rotations is not None:
            arrays["rotations"] = splats4d.rotations.detach().cpu().float().numpy()
        if splats4d.static is not None:
            _pack_splats_npz(arrays, "static", splats4d.static)
        np.savez_compressed(npz_path, **arrays)

        results = [{
            "filename": npz_name,
            "subfolder": subfolder,
            "type": self.type,
        }]
        if export_ply_frames:
            times = splats4d.times.detach().cpu().float().reshape(-1)
            for i in tqdm(range(times.shape[0]), desc="SaveSplats4D PLY frames"):
                ply_name = f"{base_name}_{counter:05}_t{i:04}.ply"
                _write_ply_splats(
                    os.path.join(full_output_folder, ply_name),
                    splats4d.at_time(float(times[i])),
                )
                results.append({
                    "filename": ply_name,
                    "subfolder": subfolder,
                    "type": self.type,
                })
        counter += 1
        return {"ui": {"splats4d": results}}


class LoadSplats4D:
    """Loads a 4D splat scene saved by SaveSplats4D from the ComfyUI input directory."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".npz")
        ]
        return {
            "required": {
                "splat4d_file": (
                    sorted(files),
                    {
                        "file_chooser": True,
                        "tooltip": "Select a Splats4D .npz archive from your input folder.",
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT4D",)
    RETURN_NAMES = ("splats4d",)
    FUNCTION = "load_splats4d"
    CATEGORY = "Camera/GSplat4D"
    DESCRIPTION = "Loads a GSPLAT4D scene from an .npz archive."

    def load_splats4d(self, splat4d_file: str, device: str = "auto"):
        path = folder_paths.get_annotated_filepath(splat4d_file)
        data = np.load(path)
        canonical = _unpack_splats_npz(data, "canonical")
        if canonical is None or "trajectories" not in data or "times" not in data:
            raise ValueError(
                f"{splat4d_file} is not a Splats4D archive (missing canonical_*/trajectories/times keys)."
            )
        trajectories = torch.from_numpy(np.asarray(data["trajectories"], dtype=np.float32))
        times = torch.from_numpy(np.asarray(data["times"], dtype=np.float32)).reshape(-1)
        rotations = None
        if "rotations" in data:
            rotations = torch.from_numpy(np.asarray(data["rotations"], dtype=np.float32))
        static = _unpack_splats_npz(data, "static")

        splats4d = GaussianSplats4D(
            static=static,
            canonical=canonical,
            trajectories=trajectories,
            times=times,
            rotations=rotations,
        )
        target_device = _resolve_device_choice(device)
        if target_device != torch.device("cpu"):
            splats4d = splats4d.to(target_device)
        return (splats4d,)

    @classmethod
    def IS_CHANGED(cls, splat4d_file: str, device: str = "auto"):
        path = folder_paths.get_annotated_filepath(splat4d_file)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, splat4d_file: str, device: str = "auto"):
        if not folder_paths.exists_annotated_filepath(splat4d_file):
            return f"Invalid splat4d file: {splat4d_file}"
        return True


NODE_CLASS_MAPPINGS = {
    "MotionMaskFromDepth": MotionMaskFromDepth,
    "EstimateTracks": EstimateTracks,
    "TracksToTrajectories": TracksToTrajectories,
    "SplitSplatsByMask": SplitSplatsByMask,
    "BuildSplats4D": BuildSplats4D,
    "RenderSplats4DFrame": RenderSplats4DFrame,
    "RenderSplats4DVideo": RenderSplats4DVideo,
    "SaveSplats4D": SaveSplats4D,
    "LoadSplats4D": LoadSplats4D,
}
