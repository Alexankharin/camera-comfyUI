import math
import os
import sys
import ssl
import shutil
import logging
import hashlib
import urllib.request
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:  # Allow notebook usage outside ComfyUI
    class _FolderPathsStub:
        def __getattr__(self, name):
            raise ModuleNotFoundError(
                "folder_paths is unavailable; LoadPlySplat requires ComfyUI runtime."
            )

    folder_paths = _FolderPathsStub()

try:
    from .reprojection_nodes import ReprojectImage
except Exception:
    try:
        from reprojection_nodes import ReprojectImage
    except Exception:
        ReprojectImage = None

_SHARP_AVAILABLE = False
_SHARP_IMPORT_ERROR: Optional[Exception] = None
_SHARP_DEFAULT_MODEL_URL = None
_SHARP_DEFAULT_CHECKPOINT_LABEL = "<download default>"
_SHARP_PREDICTOR_CACHE: Dict[Tuple[str, str], Any] = {}
try:
    _sharp_root = os.path.join(os.path.dirname(__file__), "submodules", "ml-sharpt", "src")
    if os.path.isdir(_sharp_root) and _sharp_root not in sys.path:
        sys.path.append(_sharp_root)

    from sharp.models import PredictorParams, create_predictor
    from sharp.cli.predict import predict_image as _sharp_predict_image
    from sharp.cli.predict import DEFAULT_MODEL_URL as _SHARP_DEFAULT_MODEL_URL
    from sharp.utils import color_space as _sharp_color_space
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics as _sharp_rgb_to_sh

    _SHARP_AVAILABLE = True
except Exception as exc:
    _SHARP_IMPORT_ERROR = exc


class Projection:
    PROJECTIONS = ["PINHOLE", "FISHEYE", "EQUIRECTANGULAR"]

DEVICE_CHOICES = ["auto", "cpu", "cuda"]
RENDER_MODES = ["fast", "over"]
RENDER_MODES_ALL = ["auto", "gsplat", "fast", "over"]
FUSE_MODES = ["smart", "average", "discard", "keep"]


def _infer_sh_order(f_rest_channels: int) -> int:
    if f_rest_channels == 0:
        return 0
    if f_rest_channels % 3 != 0:
        raise ValueError(f"f_rest channel count must be divisible by 3, got {f_rest_channels}")
    per_channel = f_rest_channels // 3
    total = per_channel + 1
    order = int(round(math.sqrt(total) - 1))
    if (order + 1) ** 2 != total:
        raise ValueError(f"Invalid f_rest channel count for SH: {f_rest_channels}")
    if order > 3:
        raise ValueError(f"SH order {order} is not supported (max 3)")
    return order


def _resolve_device_choice(device_choice: str, fallback: Optional[torch.device] = None) -> torch.device:
    if device_choice == "auto":
        if fallback is not None:
            return fallback
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class GaussianSplats:
    xyz: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    f_dc: torch.Tensor
    f_rest: torch.Tensor
    sh_order: Optional[int] = None

    def __post_init__(self) -> None:
        inferred = _infer_sh_order(self.f_rest.shape[1])
        if self.sh_order is None:
            self.sh_order = inferred
        elif self.sh_order != inferred:
            raise ValueError(f"sh_order={self.sh_order} does not match f_rest size ({self.f_rest.shape[1]})")

    def to(self, device: torch.device) -> "GaussianSplats":
        return GaussianSplats(
            xyz=self.xyz.to(device),
            scale=self.scale.to(device),
            rotation=self.rotation.to(device),
            opacity=self.opacity.to(device),
            f_dc=self.f_dc.to(device),
            f_rest=self.f_rest.to(device),
            sh_order=self.sh_order,
        )

    def clone(self) -> "GaussianSplats":
        return GaussianSplats(
            xyz=self.xyz.clone(),
            scale=self.scale.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity.clone(),
            f_dc=self.f_dc.clone(),
            f_rest=self.f_rest.clone(),
            sh_order=self.sh_order,
        )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __getitem__(self, index) -> "GaussianSplats":
        return self._select(index)

    def get_splat(self, index: int) -> "GaussianSplats":
        return self._select(index)

    def sh_coeffs(self) -> torch.Tensor:
        total = (self.sh_order + 1) ** 2
        expected_rest = (total - 1) * 3
        if self.f_rest.shape[1] != expected_rest:
            raise ValueError(f"Expected f_rest with {expected_rest} channels, got {self.f_rest.shape[1]}")
        coeffs = torch.cat([self.f_dc, self.f_rest], dim=1).view(-1, 3, total)
        return coeffs

    def _select(self, index) -> "GaussianSplats":
        def _slice(t: torch.Tensor) -> torch.Tensor:
            out = t[index]
            if isinstance(index, int):
                return out.unsqueeze(0)
            return out

        return GaussianSplats(
            xyz=_slice(self.xyz),
            scale=_slice(self.scale),
            rotation=_slice(self.rotation),
            opacity=_slice(self.opacity),
            f_dc=_slice(self.f_dc),
            f_rest=_slice(self.f_rest),
            sh_order=self.sh_order,
        )


# Real SH constants used in 3DGS/instant-ngp style evaluation.
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, 0.31539156525252005, 0.5462742152960396)
C3 = (0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277)


def _normalize_dirs(dirs: torch.Tensor) -> torch.Tensor:
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _sh_basis_l1(dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs.unbind(-1)
    return torch.stack(
        [
            -C1 * y,
            C1 * z,
            -C1 * x,
        ],
        dim=-1,
    )


def _sh_basis_l2(dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs.unbind(-1)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    yz = y * z
    xz = x * z
    return torch.stack(
        [
            C2[0] * xy,
            -C2[0] * yz,
            C2[1] * (3.0 * z2 - 1.0),
            -C2[0] * xz,
            C2[2] * (x2 - y2),
        ],
        dim=-1,
    )


def _sh_basis_l3(dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs.unbind(-1)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return torch.stack(
        [
            -C3[0] * y * (3.0 * x2 - y2),
            C3[1] * x * y * z,
            -C3[2] * y * (5.0 * z2 - 1.0),
            C3[3] * z * (5.0 * z2 - 3.0),
            -C3[2] * x * (5.0 * z2 - 1.0),
            C3[4] * z * (x2 - y2),
            -C3[0] * x * (x2 - 3.0 * y2),
        ],
        dim=-1,
    )


def _sh_basis(deg: int, dirs: torch.Tensor) -> torch.Tensor:
    dirs = _normalize_dirs(dirs)
    x, y, z = dirs.unbind(-1)
    basis = [torch.full_like(x, C0)]
    if deg >= 1:
        basis.append(-C1 * y)
        basis.append(C1 * z)
        basis.append(-C1 * x)
    if deg >= 2:
        x2 = x * x
        y2 = y * y
        z2 = z * z
        basis.append(C2[0] * x * y)
        basis.append(-C2[0] * y * z)
        basis.append(C2[1] * (3.0 * z2 - 1.0))
        basis.append(-C2[0] * x * z)
        basis.append(C2[2] * (x2 - y2))
    if deg >= 3:
        x2 = x * x
        y2 = y * y
        z2 = z * z
        basis.append(-C3[0] * y * (3.0 * x2 - y2))
        basis.append(C3[1] * x * y * z)
        basis.append(-C3[2] * y * (5.0 * z2 - 1.0))
        basis.append(C3[3] * z * (5.0 * z2 - 3.0))
        basis.append(-C3[2] * x * (5.0 * z2 - 1.0))
        basis.append(C3[4] * z * (x2 - y2))
        basis.append(-C3[0] * x * (x2 - 3.0 * y2))
    return torch.stack(basis, dim=-1)


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    if deg > 3:
        raise ValueError(f"SH degree {deg} is not supported (max 3)")
    basis = _sh_basis(deg, dirs)
    return (sh * basis.unsqueeze(-2)).sum(dim=-1)


def _make_rotation_support(l: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = 2 * l + 1
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1337 + l)
    for _ in range(1000):
        dirs = torch.randn((n, 3), generator=gen)
        dirs = _normalize_dirs(dirs)
        if l == 1:
            A = _sh_basis_l1(dirs)
        elif l == 2:
            A = _sh_basis_l2(dirs)
        else:
            A = _sh_basis_l3(dirs)
        A64 = A.double()
        if torch.linalg.matrix_rank(A64) == n:
            return dirs, torch.inverse(A64)
    raise RuntimeError(f"Failed to build SH rotation support for l={l}")


_SH_ROT_DIRS = {}
_SH_ROT_AINV = {}
for _l in (1, 2, 3):
    _dirs, _ainv = _make_rotation_support(_l)
    _SH_ROT_DIRS[_l] = _dirs
    _SH_ROT_AINV[_l] = _ainv


def _sh_rotation_matrix(l: int, rotation: torch.Tensor) -> torch.Tensor:
    device = rotation.device
    dtype = rotation.dtype
    dirs = _SH_ROT_DIRS[l].to(device=device, dtype=dtype)
    a_inv = _SH_ROT_AINV[l].to(device=device, dtype=dtype)
    rot = rotation
    dirs_rot = dirs @ rot
    if l == 1:
        B = _sh_basis_l1(dirs_rot)
    elif l == 2:
        B = _sh_basis_l2(dirs_rot)
    else:
        B = _sh_basis_l3(dirs_rot)
    return a_inv @ B


def rotate_sh_coeffs(sh_coeffs: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    total = sh_coeffs.shape[-1]
    order = int(round(math.sqrt(total) - 1))
    if (order + 1) ** 2 != total:
        raise ValueError(f"Invalid SH coefficient count: {total}")
    return rotate_sh_coeffs_ordered(sh_coeffs, rotation, order)


def rotate_sh_coeffs_ordered(sh_coeffs: torch.Tensor, rotation: torch.Tensor, order: int) -> torch.Tensor:
    expected = (order + 1) ** 2
    if sh_coeffs.shape[-1] != expected:
        raise ValueError(f"Expected {expected} SH coefficients per channel, got {sh_coeffs.shape[-1]}")
    if order == 0:
        return sh_coeffs
    parts = [sh_coeffs[..., 0:1]]
    T1 = _sh_rotation_matrix(1, rotation)
    parts.append(sh_coeffs[..., 1:4] @ T1.T)
    if order >= 2:
        T2 = _sh_rotation_matrix(2, rotation)
        parts.append(sh_coeffs[..., 4:9] @ T2.T)
    if order >= 3:
        T3 = _sh_rotation_matrix(3, rotation)
        parts.append(sh_coeffs[..., 9:16] @ T3.T)
    return torch.cat(parts, dim=-1)


def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    R = rotation
    m00 = R[0, 0]
    m11 = R[1, 1]
    m22 = R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat / quat.norm()


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def splat_cloud_rotation(splats: GaussianSplats, transform_matrix: torch.Tensor) -> GaussianSplats:
    device = splats.xyz.device
    if isinstance(transform_matrix, torch.Tensor):
        matrix = transform_matrix.to(device).view(4, 4).float()
    else:
        matrix = torch.tensor(transform_matrix, device=device, dtype=torch.float32).view(4, 4)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    coords = splats.xyz
    coords = coords @ rotation.T + translation
    quat_r = _rotation_matrix_to_quaternion(rotation)
    rot = splats.rotation
    rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rot = _quat_mul(quat_r, rot)
    coeffs = splats.sh_coeffs()
    coeffs = rotate_sh_coeffs_ordered(coeffs, rotation, splats.sh_order)
    f_dc = coeffs[:, :, 0]
    rest = (splats.sh_order + 1) ** 2 - 1
    if rest == 0:
        f_rest = torch.zeros((coeffs.shape[0], 0), device=coeffs.device, dtype=coeffs.dtype)
    else:
        f_rest = coeffs[:, :, 1:].reshape(coeffs.shape[0], rest * 3)
    return GaussianSplats(
        xyz=coords,
        scale=splats.scale.clone(),
        rotation=rot,
        opacity=splats.opacity.clone(),
        f_dc=f_dc,
        f_rest=f_rest,
        sh_order=splats.sh_order,
    )


def _xyz_to_pinhole(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2.0)
    depth = torch.sqrt(X * X + Y * Y + Z * Z)
    u = (X / Z) * f
    v = (Y / Z) * f
    return u, v, depth


def _xyz_to_fisheye(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fov_rad = math.radians(fov)
    depth = torch.sqrt(X * X + Y * Y + Z * Z)
    theta = torch.acos(Z / depth.clamp(min=1e-8))
    phi = torch.atan2(Y, X)
    r = theta / (fov_rad / 2.0)
    u = r * torch.cos(phi)
    v = r * torch.sin(phi)
    return u, v, depth


def _xyz_to_equirect(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fov_rad = math.radians(fov) / 2.0
    depth = torch.sqrt(X * X + Y * Y + Z * Z)
    lon = torch.atan2(X, Z)
    lat = torch.asin(Y / depth.clamp(min=1e-8))
    u = lon / fov_rad
    v = lat / (math.pi / 2.0)
    return u, v, depth


PLY_TYPES = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def _parse_ply_header(f) -> Tuple[str, int, List[Tuple[str, str]]]:
    fmt = None
    vertex_count = 0
    props: List[Tuple[str, str]] = []
    in_vertex = False
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        text = line.decode("ascii", errors="ignore").strip()
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element "):
            parts = text.split()
            element = parts[1]
            count = int(parts[2])
            in_vertex = element == "vertex"
            if in_vertex:
                vertex_count = count
        elif text.startswith("property ") and in_vertex:
            parts = text.split()
            if parts[1] == "list":
                continue
            props.append((parts[2], parts[1]))
        elif text == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing format")
    return fmt, vertex_count, props


def _read_ply_vertices(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        fmt, vertex_count, props = _parse_ply_header(f)
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                rows.append([float(x) for x in line.decode("ascii", errors="ignore").strip().split()])
            data = np.asarray(rows, dtype=np.float32)
            if data.shape[1] < len(props):
                raise ValueError("PLY vertex data does not match header properties")
            out = {}
            for idx, (name, _) in enumerate(props):
                out[name] = data[:, idx]
            return out
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")
        dtype = [(name, np.dtype(PLY_TYPES[ptype]).newbyteorder("<")) for name, ptype in props]
        data = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)
        return {name: data[name] for name, _ in props}


def _extract_f_rest(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
    keys = [k for k in data.keys() if k.startswith("f_rest_")]
    if not keys:
        return np.zeros((data["x"].shape[0], 0), dtype=np.float32), 0
    indices = sorted(int(k.split("_")[-1]) for k in keys)
    if indices != list(range(len(indices))):
        raise ValueError("f_rest indices must be contiguous starting at 0")
    f_rest = np.stack([data[f"f_rest_{i}"] for i in indices], axis=1).astype(np.float32)
    sh_order = _infer_sh_order(f_rest.shape[1])
    return f_rest, sh_order


def _ensure_sharp_available() -> None:
    if not _SHARP_AVAILABLE:
        raise ModuleNotFoundError(
            f"ml-sharpt is unavailable. Run this pack's install.py (ComfyUI-Manager does this "
            f"automatically) to fetch submodules/ml-sharpt and its dependencies (incl. gsplat). "
            f"Import error: {_SHARP_IMPORT_ERROR}"
        )


def _horizontal_fov_to_f_px(width: int, horizontal_fov: float) -> float:
    if horizontal_fov <= 0.0 or horizontal_fov >= 179.0:
        raise ValueError("horizontal_fov must be between 0 and 179 degrees.")
    fov_rad = math.radians(horizontal_fov)
    return (width / 2.0) / math.tan(fov_rad / 2.0)


def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    img = image
    if img.dim() == 4:
        img = img[0]
    if img.dim() == 3 and img.shape[-1] not in (3, 4) and img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img = img.detach().cpu().float()
    if img.numel() == 0:
        raise ValueError("Input image is empty.")
    if img.max().item() <= 1.0:
        img = img * 255.0
    img = img.clamp(0.0, 255.0).to(torch.uint8)
    return img.numpy()


def _list_sharp_checkpoint_choices() -> List[str]:
    input_dir = folder_paths.get_input_directory()
    checkpoint_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".pt")
    ]
    return [_SHARP_DEFAULT_CHECKPOINT_LABEL] + sorted(checkpoint_files)


def _build_rotation_matrix(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta_rad = math.radians(theta_deg)
    phi_rad = math.radians(phi_deg)
    r_theta = np.array(
        [
            [math.cos(phi_rad), 0.0, math.sin(phi_rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(phi_rad), 0.0, math.cos(phi_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    r_phi = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, math.cos(theta_rad), -math.sin(theta_rad), 0.0],
            [0.0, math.sin(theta_rad), math.cos(theta_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return r_theta @ r_phi


def _concat_splats(splats_list: List[GaussianSplats]) -> GaussianSplats:
    if not splats_list:
        raise ValueError("No splats provided to merge.")
    base = splats_list[0]
    for splats in splats_list[1:]:
        if splats.sh_order != base.sh_order or splats.f_rest.shape[1] != base.f_rest.shape[1]:
            raise ValueError("All splats must have the same SH order to merge.")
    return GaussianSplats(
        xyz=torch.cat([s.xyz for s in splats_list], dim=0),
        scale=torch.cat([s.scale for s in splats_list], dim=0),
        rotation=torch.cat([s.rotation for s in splats_list], dim=0),
        opacity=torch.cat([s.opacity for s in splats_list], dim=0),
        f_dc=torch.cat([s.f_dc for s in splats_list], dim=0),
        f_rest=torch.cat([s.f_rest for s in splats_list], dim=0),
        sh_order=base.sh_order,
    )


def _pad_sh_order(splats: GaussianSplats, sh_order: int) -> GaussianSplats:
    """Zero-pad a splat cloud's SH coefficients up to ``sh_order``.

    The SH decode used throughout this file (``sh_coeffs`` / ``eval_sh``) is
    ``cat([f_dc, f_rest], dim=1).view(-1, 3, total)`` — channel-major over the
    concatenated flat vector — so padding must reflow the existing
    ``(3, total_old)`` coefficient rows into a zeroed ``(3, total_new)`` block
    and re-flatten. Simply appending zeros to f_rest would shift the green/blue
    DC terms into the red channel's l>=1 slots and corrupt colors.
    """
    if splats.sh_order == sh_order:
        return splats
    if splats.sh_order > sh_order:
        raise ValueError("Cannot reduce SH order by zero-padding.")
    total_old = (splats.sh_order + 1) ** 2
    total_new = (sh_order + 1) ** 2
    n = splats.xyz.shape[0]
    old = torch.cat([splats.f_dc, splats.f_rest], dim=1).view(n, 3, total_old)
    coeffs = torch.zeros((n, 3, total_new), device=splats.f_dc.device, dtype=splats.f_dc.dtype)
    coeffs[:, :, :total_old] = old
    flat = coeffs.reshape(n, 3 * total_new)
    return GaussianSplats(
        xyz=splats.xyz,
        scale=splats.scale,
        rotation=splats.rotation,
        opacity=splats.opacity,
        f_dc=flat[:, :3],
        f_rest=flat[:, 3:],
        sh_order=sh_order,
    )


def _match_sh_orders(a: GaussianSplats, b: GaussianSplats) -> Tuple[GaussianSplats, GaussianSplats]:
    """Bring two splat clouds to a common (max) SH order via zero padding."""
    order = max(a.sh_order, b.sh_order)
    return _pad_sh_order(a, order), _pad_sh_order(b, order)


def _direction_bins(xyz: torch.Tensor, angle_deg: float) -> Tuple[torch.Tensor, int]:
    if angle_deg <= 0.0:
        raise ValueError("direction angle must be greater than 0 degrees.")
    step = math.radians(angle_deg)
    theta_bins = max(1, int(math.ceil(math.pi / step)))
    phi_bins = max(1, int(math.ceil(2.0 * math.pi / step)))
    dirs = xyz / xyz.norm(dim=1, keepdim=True).clamp(min=1e-8)
    theta = torch.acos(dirs[:, 2].clamp(-1.0, 1.0))
    phi = torch.atan2(dirs[:, 1], dirs[:, 0])
    theta_bin = torch.floor(theta / step).to(torch.int64).clamp(min=0, max=theta_bins - 1)
    phi_bin = torch.floor((phi + math.pi) / step).to(torch.int64).clamp(min=0, max=phi_bins - 1)
    return theta_bin * phi_bins + phi_bin, phi_bins


def _filter_overlapping_by_fov(
    other: GaussianSplats,
    horizontal_fov: float,
    padding_deg: float,
) -> GaussianSplats:
    if len(other) == 0:
        return other
    if horizontal_fov <= 0.0 or horizontal_fov >= 179.0:
        raise ValueError("horizontal_fov must be between 0 and 179 degrees.")
    half_fov = math.radians(horizontal_fov) * 0.5
    if padding_deg != 0.0:
        half_fov += math.radians(padding_deg)
    max_half = math.radians(89.9)
    half_fov = max(1e-6, min(half_fov, max_half))
    X, Y, Z = other.xyz.unbind(-1)
    in_front = Z > 1e-6
    x_angle = torch.atan2(X, Z)
    y_angle = torch.atan2(Y, Z)
    in_square = (x_angle.abs() <= half_fov) & (y_angle.abs() <= half_fov)
    keep = ~(in_front & in_square)
    return other[keep]


def _stitch_splats(
    splats_list: List[GaussianSplats],
    mode: str,
    voxel_size: float,
    direction_deg: float,
    pinhole_fov: Optional[float] = None,
    weights_list: Optional[List[float]] = None,
) -> GaussianSplats:
    """Merge multiple splat clouds, optionally reducing duplicates per voxel.

    weights_list: optional per-list weight multipliers (one float per entry of
    splats_list) applied to the per-splat weights before the voxel reduction.
    Only affects the "smart" and "average" modes; "keep", "discard" and
    "main_direction" ignore it.
    """
    if weights_list is not None and len(weights_list) != len(splats_list):
        raise ValueError(
            f"weights_list length ({len(weights_list)}) must match splats_list length ({len(splats_list)})."
        )
    if mode == "main_direction":
        if not splats_list:
            raise ValueError("No splats provided to merge.")
        if pinhole_fov is None:
            raise ValueError("pinhole_fov is required for main_direction stitching.")
        main = splats_list[0]
        filtered = [main]
        for splats in splats_list[1:]:
            filtered.append(_filter_overlapping_by_fov(splats, pinhole_fov, direction_deg))
        return _concat_splats(filtered)

    merged = _concat_splats(splats_list)
    if mode == "keep" or voxel_size <= 0.0 or len(merged) == 0:
        return merged

    device = merged.xyz.device
    dtype = merged.xyz.dtype
    voxel = torch.floor(merged.xyz / float(voxel_size)).to(torch.int64)
    unique, inv = torch.unique(voxel, dim=0, return_inverse=True)
    num_voxels = unique.shape[0]

    if mode == "discard":
        idx = torch.arange(len(merged), device=device, dtype=torch.long)
        min_idx = torch.full((num_voxels,), len(merged), device=device, dtype=torch.long)
        min_idx.scatter_reduce_(0, inv, idx, reduce="amin", include_self=True)
        keep = idx == min_idx[inv]
        return merged[keep]

    weights = torch.ones((len(merged),), device=device, dtype=dtype)
    if mode == "smart":
        opacity = torch.sigmoid(merged.opacity.squeeze(-1))
        sigma = torch.exp(merged.scale).mean(dim=1)
        weights = opacity / sigma.clamp(min=1e-6)
    if weights_list is not None:
        multipliers = torch.cat(
            [
                torch.full((len(s),), float(w), device=device, dtype=dtype)
                for s, w in zip(splats_list, weights_list)
            ]
        )
        weights = weights * multipliers.clamp(min=0.0)

    sum_w = torch.zeros((num_voxels,), device=device, dtype=dtype)
    sum_w.scatter_add_(0, inv, weights)
    # Voxels whose total weight is ~0 (e.g. FuseSplats with weight 0.0 for one
    # cloud, in voxels populated only by that cloud) would otherwise reduce to
    # degenerate splats at the origin (all-zero weighted sums divided by the
    # clamp); drop those voxels instead.
    nonzero_voxel = sum_w > 1e-8
    sum_w = sum_w.clamp(min=1e-8)

    def _weighted_sum(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros((num_voxels, values.shape[1]))
        out = torch.zeros((num_voxels, values.shape[1]), device=device, dtype=values.dtype)
        out.scatter_add_(0, inv[:, None].expand(-1, values.shape[1]), values * weights[:, None])
        return out

    xyz = _weighted_sum(merged.xyz) / sum_w[:, None]
    sigma = _weighted_sum(torch.exp(merged.scale)) / sum_w[:, None]
    scale = torch.log(sigma.clamp(min=1e-9))

    idx = torch.arange(len(merged), device=device, dtype=torch.long)
    min_idx = torch.full((num_voxels,), len(merged), device=device, dtype=torch.long)
    min_idx.scatter_reduce_(0, inv, idx, reduce="amin", include_self=True)
    ref = merged.rotation[min_idx]
    ref_per = ref[inv]
    dot = (merged.rotation * ref_per).sum(dim=1, keepdim=True)
    aligned = torch.where(dot < 0, -merged.rotation, merged.rotation)
    rot_sum = _weighted_sum(aligned)
    rotation = rot_sum / sum_w[:, None]
    rotation = rotation / rotation.norm(dim=1, keepdim=True).clamp(min=1e-8)

    f_dc = _weighted_sum(merged.f_dc) / sum_w[:, None]
    if merged.f_rest.shape[1] > 0:
        f_rest = _weighted_sum(merged.f_rest) / sum_w[:, None]
    else:
        f_rest = merged.f_rest.new_zeros((num_voxels, 0))

    opacity = torch.sigmoid(merged.opacity.squeeze(-1))
    opacity_sum = torch.zeros((num_voxels,), device=device, dtype=dtype)
    opacity_sum.scatter_add_(0, inv, opacity * weights)
    opacity_avg = (opacity_sum / sum_w).clamp(1e-6, 1.0 - 1e-6)
    opacity_logits = torch.log(opacity_avg / (1.0 - opacity_avg)).view(-1, 1)

    out = GaussianSplats(
        xyz=xyz,
        scale=scale,
        rotation=rotation,
        opacity=opacity_logits,
        f_dc=f_dc,
        f_rest=f_rest,
        sh_order=merged.sh_order,
    )
    if not bool(nonzero_voxel.all()):
        out = out[nonzero_voxel]
    return out


def _get_sharp_default_checkpoint_path() -> Optional[str]:
    if _SHARP_DEFAULT_MODEL_URL is None:
        return None
    filename = os.path.basename(_SHARP_DEFAULT_MODEL_URL)
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    return os.path.join(cache_dir, filename)

def _download_sharp_checkpoint(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        torch.hub.download_url_to_file(url, destination, progress=True)
        return
    except Exception:
        pass

    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, context=ctx) as response, open(destination, "wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as exc:
        if os.path.isfile(destination):
            try:
                os.remove(destination)
            except OSError:
                pass
        raise RuntimeError(
            "Failed to download the SHARP checkpoint. If your environment blocks SSL downloads, "
            "manually download the .pt file and select it from the input folder."
        ) from exc


def _load_sharp_predictor(
    checkpoint_path: Optional[str],
    device: torch.device,
):
    key = (checkpoint_path or "default", str(device))
    cached = _SHARP_PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached

    if checkpoint_path:
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path)
    else:
        if _SHARP_DEFAULT_MODEL_URL is None:
            raise RuntimeError("Default SHARP checkpoint URL is unavailable.")
        cached_path = _get_sharp_default_checkpoint_path()
        if cached_path is None:
            raise RuntimeError("Default SHARP checkpoint cache location is unavailable.")
        if not os.path.isfile(cached_path):
            _download_sharp_checkpoint(_SHARP_DEFAULT_MODEL_URL, cached_path)
        try:
            state_dict = torch.load(cached_path, weights_only=True)
        except TypeError:
            state_dict = torch.load(cached_path)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    _SHARP_PREDICTOR_CACHE[key] = predictor
    return predictor


def _write_ply_splats(path: str, splats: GaussianSplats) -> None:
    xyz = splats.xyz.detach().cpu().float().numpy()
    scale = splats.scale.detach().cpu().float().numpy()
    rotation = splats.rotation.detach().cpu().float().numpy()
    opacity = splats.opacity.detach().cpu().float().reshape(-1).numpy()
    f_dc = splats.f_dc.detach().cpu().float().numpy()
    f_rest = splats.f_rest.detach().cpu().float().numpy()

    props: List[Tuple[str, np.ndarray]] = [
        ("x", xyz[:, 0]),
        ("y", xyz[:, 1]),
        ("z", xyz[:, 2]),
        ("f_dc_0", f_dc[:, 0]),
        ("f_dc_1", f_dc[:, 1]),
        ("f_dc_2", f_dc[:, 2]),
        ("opacity", opacity),
        ("scale_0", scale[:, 0]),
        ("scale_1", scale[:, 1]),
        ("scale_2", scale[:, 2]),
        ("rot_0", rotation[:, 0]),
        ("rot_1", rotation[:, 1]),
        ("rot_2", rotation[:, 2]),
        ("rot_3", rotation[:, 3]),
    ]
    if f_rest.size > 0:
        for i in range(f_rest.shape[1]):
            props.append((f"f_rest_{i}", f_rest[:, i]))

    dtype = [(name, "<f4") for name, _ in props]
    data = np.empty(xyz.shape[0], dtype=dtype)
    for name, values in props:
        data[name] = values.astype(np.float32, copy=False)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {xyz.shape[0]}",
    ]
    for name, _ in props:
        header_lines.append(f"property float {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def _coerce_splats(splats: GaussianSplats, device: torch.device, dtype: torch.dtype) -> GaussianSplats:
    return GaussianSplats(
        xyz=splats.xyz.to(device=device, dtype=dtype),
        scale=splats.scale.to(device=device, dtype=dtype),
        rotation=splats.rotation.to(device=device, dtype=dtype),
        opacity=splats.opacity.to(device=device, dtype=dtype),
        f_dc=splats.f_dc.to(device=device, dtype=dtype),
        f_rest=splats.f_rest.to(device=device, dtype=dtype),
        sh_order=splats.sh_order,
    )


def _progress(iterable, desc: str = ""):
    """Wrap an iterable with tqdm if it is available, otherwise pass through."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


_GSPLAT_AVAILABLE_CACHE: Optional[bool] = None


def _gsplat_available() -> bool:
    """Return True if the gsplat package is importable (checked once, cached)."""
    global _GSPLAT_AVAILABLE_CACHE
    if _GSPLAT_AVAILABLE_CACHE is None:
        try:
            import importlib.util

            _GSPLAT_AVAILABLE_CACHE = importlib.util.find_spec("gsplat") is not None
        except Exception:
            _GSPLAT_AVAILABLE_CACHE = False
    return _GSPLAT_AVAILABLE_CACHE


def _import_gsplat():
    """Lazy-import gsplat with an actionable error message."""
    try:
        import gsplat
    except ImportError as exc:
        raise RuntimeError(
            "gsplat is required for render_mode='gsplat' and SplatPolish. "
            "Install it with: pip install gsplat (requires a CUDA-enabled PyTorch build). "
            f"Import error: {exc}"
        ) from exc
    return gsplat


def _quats_to_rotation_matrices(quats: torch.Tensor) -> torch.Tensor:
    """Convert [N,4] wxyz quaternions to [N,3,3] rotation matrices."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)


def _empty_render(output_width: int, output_height: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Black image, zero alpha and zero disparity for views with no visible splats."""
    img = torch.zeros((1, output_height, output_width, 3), device=device)
    mask = torch.zeros((output_height, output_width), device=device)
    disparity = torch.zeros((1, output_height, output_width, 1), device=device)
    return img, mask, disparity


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two [B,C,H,W] images with values in [0,1]."""
    channels = img1.shape[1]
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _coerce_trajectory(trajectory, num_frames: int, device: torch.device) -> torch.Tensor:
    """Coerce a trajectory input to a [num_frames,4,4] float tensor on device.

    Accepts [4,4] (broadcast to all frames), [1,4,4] or [num_frames,4,4].
    """
    if isinstance(trajectory, torch.Tensor):
        traj = trajectory.detach().float()
    else:
        traj = torch.tensor(trajectory, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(f"trajectory must be [T,4,4], got shape {tuple(traj.shape)}")
    if traj.shape[0] == 1 and num_frames > 1:
        traj = traj.expand(num_frames, 4, 4)
    if traj.shape[0] != num_frames:
        raise ValueError(
            f"trajectory has {traj.shape[0]} poses but {num_frames} frames were provided."
        )
    return traj.to(device)


def _normalize_map_sequence(seq, num_frames: int, name: str) -> torch.Tensor:
    """Coerce a per-frame map (depth/mask) input to [T,H,W] float ([1,H,W] broadcasts)."""
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq, dtype=torch.float32)
    seq = seq.float()
    if seq.dim() == 4 and seq.shape[-1] == 1:
        seq = seq[..., 0]
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)
    if seq.dim() != 3:
        raise ValueError(f"{name} must be [T,H,W] (or [H,W]), got shape {tuple(seq.shape)}")
    if seq.shape[0] not in (1, num_frames):
        raise ValueError(
            f"{name} has {seq.shape[0]} frames but the video has {num_frames}."
        )
    return seq


def _project_splats_to_pixels(
    xyz_cam: torch.Tensor,
    horizontal_fov: float,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-frame splat centers to pixel coordinates of the source pinhole image.

    Uses the same focal convention as SHARP/ImageToSplat (single f_px from the
    horizontal FOV over the width). Returns (px, py, z, in_bounds).
    """
    f_px = _horizontal_fov_to_f_px(width, horizontal_fov)
    X, Y, Z = xyz_cam.unbind(-1)
    zc = Z.clamp(min=1e-6)
    px = X / zc * f_px + (width - 1) / 2.0
    py = Y / zc * f_px + (height - 1) / 2.0
    in_bounds = (Z > 1e-6) & (px >= 0.0) & (px <= width - 1) & (py >= 0.0) & (py <= height - 1)
    return px, py, Z, in_bounds


def _sample_map_at_pixels(
    map_hw: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Nearest-neighbour sample a [Hm,Wm] map at pixel coords defined on a width x height image."""
    map_h, map_w = int(map_hw.shape[0]), int(map_hw.shape[1])
    if map_w == width and map_h == height:
        xi = px.round().long().clamp(0, map_w - 1)
        yi = py.round().long().clamp(0, map_h - 1)
    else:
        xi = (px / max(width - 1, 1) * (map_w - 1)).round().long().clamp(0, map_w - 1)
        yi = (py / max(height - 1, 1) * (map_h - 1)).round().long().clamp(0, map_h - 1)
    return map_hw[yi, xi]


def _render_gaussians_gsplat(
    splats: GaussianSplats,
    view_matrix: torch.Tensor,
    camera_horizontal_fov: float,
    output_width: int,
    output_height: int,
    max_splats: int,
    opacity_is_logit: bool,
    add_sh_bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA gsplat rasterization backend (PINHOLE only). Returns (image, alpha, disparity)."""
    gsplat = _import_gsplat()
    dev = splats.xyz.device
    if dev.type != "cuda":
        raise RuntimeError(
            "render_mode='gsplat' requires CUDA tensors. Set device='cuda' "
            "(or 'auto' on a CUDA machine), or use render_mode='fast'."
        )
    if len(splats) == 0:
        return _empty_render(output_width, output_height, dev)

    means = splats.xyz.float()
    quats = splats.rotation.float()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = torch.exp(splats.scale.float())
    opacity = splats.opacity.float().view(-1)
    if opacity_is_logit:
        opacity = torch.sigmoid(opacity)
    else:
        opacity = opacity.clamp(0.0, 1.0)
    f_dc = splats.f_dc.float()
    f_rest = splats.f_rest.float()

    if max_splats > 0 and means.shape[0] > max_splats:
        keep = torch.topk(opacity, k=max_splats).indices
        means = means[keep]
        quats = quats[keep]
        scales = scales[keep]
        opacity = opacity[keep]
        f_dc = f_dc[keep]
        f_rest = f_rest[keep]

    total = (splats.sh_order + 1) ** 2
    if add_sh_bias:
        # gsplat evaluates SH internally and adds the +0.5 bias itself.
        colors = torch.cat([f_dc, f_rest], dim=1).view(-1, 3, total).transpose(1, 2).contiguous()
        sh_degree: Optional[int] = int(splats.sh_order)
    else:
        # gsplat always adds the SH bias, so evaluate SH manually and pass raw colors.
        R = view_matrix[:3, :3]
        t = view_matrix[:3, 3]
        campos = -(R.transpose(0, 1) @ t)
        dirs = means - campos
        coeffs = torch.cat([f_dc, f_rest], dim=1).view(-1, 3, total)
        colors = eval_sh(splats.sh_order, coeffs, dirs).clamp(0.0, 1.0)
        sh_degree = None

    fov_rad = math.radians(camera_horizontal_fov)
    f_px = 0.5 * output_width / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [f_px, 0.0, output_width / 2.0],
            [0.0, f_px, output_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=dev,
        dtype=torch.float32,
    )
    renders, alphas, _meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=view_matrix.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(output_width),
        height=int(output_height),
        sh_degree=sh_degree,
        render_mode="RGB+ED",
    )
    rgb = renders[0, ..., :3].clamp(0.0, 1.0)
    depth = renders[0, ..., 3]
    alpha = alphas[0, ..., 0].clamp(0.0, 1.0)
    # gsplat's "ED" channel is expected z-depth; convert it to RADIAL ray depth
    # (multiply by the per-pixel ray norm) so the disparity semantics match the
    # "fast"/"over" backends, which use ||XYZ|| — otherwise render_mode="auto"
    # silently switches disparity meaning between CPU and CUDA machines.
    xs = (torch.arange(output_width, device=dev, dtype=torch.float32) + 0.5 - output_width / 2.0) / f_px
    ys = (torch.arange(output_height, device=dev, dtype=torch.float32) + 0.5 - output_height / 2.0) / f_px
    ray_norm = torch.sqrt(1.0 + xs.view(1, -1) ** 2 + ys.view(-1, 1) ** 2)
    depth = depth * ray_norm
    disparity = torch.where(depth > 1e-6, alpha / depth.clamp(min=1e-6), torch.zeros_like(depth))
    return rgb.unsqueeze(0), alpha, disparity.unsqueeze(0).unsqueeze(-1)


def render_gaussians(
    splats: "GaussianSplats",
    camera_matrix,
    camera_projection: str,
    camera_horizontal_fov: float,
    output_width: int,
    output_height: int,
    max_splats: int = 0,
    opacity_is_logit: bool = True,
    add_sh_bias: bool = True,
    render_mode: str = "auto",
    chunk_size: int = 256,
    max_radius: int = 32,
    device: str = "auto",
) -> tuple:
    """Render Gaussian splats from a world-to-camera 4x4 matrix.

    Returns (image [1,H,W,3] float 0..1, alpha/mask [H,W], disparity [1,H,W,1]).
    All three outputs are always present, even when no splat is visible.

    render_mode:
      - "auto": gsplat if importable, running on CUDA and projection is PINHOLE, else "fast".
      - "gsplat": CUDA gsplat rasterization (PINHOLE only, raises otherwise).
      - "fast": chunked torch splatting; anisotropic projected 2D covariance for PINHOLE,
        isotropic approximation for FISHEYE/EQUIRECTANGULAR.
      - "over": slow per-splat depth-sorted over-compositing (isotropic).
    """
    target_device = _resolve_device_choice(device)
    if splats.xyz.device != target_device:
        splats = splats.to(target_device)
    dev = splats.xyz.device

    mode = render_mode
    if mode == "auto":
        if (
            camera_projection == "PINHOLE"
            and torch.cuda.is_available()
            and dev.type == "cuda"
            and _gsplat_available()
        ):
            mode = "gsplat"
        else:
            mode = "fast"
    if mode not in ("gsplat", "fast", "over"):
        raise ValueError(f"Unknown render_mode: {render_mode}")

    if isinstance(camera_matrix, torch.Tensor):
        M = camera_matrix.to(dev).view(4, 4).float()
    else:
        M = torch.tensor(camera_matrix, device=dev, dtype=torch.float32).view(4, 4)

    if mode == "gsplat":
        if camera_projection != "PINHOLE":
            raise ValueError(
                f"render_mode='gsplat' supports only the PINHOLE projection (got {camera_projection}). "
                "Use render_mode='fast' or 'over' for FISHEYE/EQUIRECTANGULAR."
            )
        return _render_gaussians_gsplat(
            splats,
            M,
            camera_horizontal_fov,
            output_width,
            output_height,
            max_splats,
            opacity_is_logit,
            add_sh_bias,
        )

    R = M[:3, :3]
    t = M[:3, 3]

    coords = splats.xyz @ R.T + t
    z = coords[:, 2]
    in_front = z > 1e-6
    if not in_front.any():
        return _empty_render(output_width, output_height, dev)

    coords = coords[in_front]
    f_dc = splats.f_dc[in_front]
    f_rest = splats.f_rest[in_front]
    opacity = splats.opacity[in_front].squeeze(-1)
    scale = splats.scale[in_front]
    rotation = splats.rotation[in_front]

    if opacity_is_logit:
        opacity = torch.sigmoid(opacity)

    if max_splats > 0 and coords.shape[0] > max_splats:
        keep = torch.topk(opacity, k=max_splats).indices
        coords = coords[keep]
        f_dc = f_dc[keep]
        f_rest = f_rest[keep]
        opacity = opacity[keep]
        scale = scale[keep]
        rotation = rotation[keep]

    dirs = _normalize_dirs(coords)
    total = (splats.sh_order + 1) ** 2
    expected_rest = (total - 1) * 3
    if f_rest.shape[1] != expected_rest:
        raise ValueError(f"Expected f_rest with {expected_rest} channels, got {f_rest.shape[1]}")
    coeffs = torch.cat([f_dc, f_rest], dim=1).view(-1, 3, total)
    colors = eval_sh(splats.sh_order, coeffs, dirs)
    if add_sh_bias:
        colors = colors + 0.5
    colors = colors.clamp(0.0, 1.0)

    X, Y, Z = coords.unbind(-1)
    if camera_projection == "PINHOLE":
        u, v, depth = _xyz_to_pinhole(X, Y, Z, camera_horizontal_fov)
    elif camera_projection == "FISHEYE":
        u, v, depth = _xyz_to_fisheye(X, Y, Z, camera_horizontal_fov)
    else:
        u, v, depth = _xyz_to_equirect(X, Y, Z, camera_horizontal_fov)

    valid = (u >= -1.0) & (u <= 1.0) & (v >= -1.0) & (v <= 1.0)
    if not valid.any():
        return _empty_render(output_width, output_height, dev)

    coords = coords[valid]
    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    colors = colors[valid]
    opacity = opacity[valid]
    scale = scale[valid]
    rotation = rotation[valid]

    px = (u * 0.5 + 0.5) * (output_width - 1)
    py = (v * 0.5 + 0.5) * (output_height - 1)

    fov_rad = math.radians(camera_horizontal_fov)
    f = 1.0 / math.tan(fov_rad / 2.0)
    fx = f * (output_width - 1) / 2.0
    fy = f * (output_height - 1) / 2.0
    # Legacy isotropic footprint, used by "over" mode and by "fast" for
    # non-pinhole projections (kept for regression compatibility).
    scale_mean = scale.mean(dim=1)
    sigma_x = (scale_mean * fx / depth.clamp(min=1e-6)).clamp(min=0.5, max=512.0)
    sigma_y = (scale_mean * fy / depth.clamp(min=1e-6)).clamp(min=0.5, max=512.0)

    if mode == "fast":
        if camera_projection == "PINHOLE":
            # Anisotropic footprint: project the 3D covariance to the image plane.
            # Sigma3D = Rq S^2 Rq^T (world frame), rotated into the camera frame by
            # the view rotation W, then Sigma2D = J W Sigma3D W^T J^T with J the
            # perspective Jacobian, plus a 0.3px anti-alias blur.
            Rq = _quats_to_rotation_matrices(rotation)
            W3 = R.unsqueeze(0) @ Rq
            s2 = torch.exp(2.0 * scale)
            cov_cam = (W3 * s2.unsqueeze(1)) @ W3.transpose(1, 2)
            Xc, Yc, Zc = coords.unbind(-1)
            zc = Zc.clamp(min=1e-6)
            j00 = fx / zc
            j02 = -fx * Xc / (zc * zc)
            j11 = fy / zc
            j12 = -fy * Yc / (zc * zc)
            c00 = cov_cam[:, 0, 0]
            c01 = cov_cam[:, 0, 1]
            c02 = cov_cam[:, 0, 2]
            c11 = cov_cam[:, 1, 1]
            c12 = cov_cam[:, 1, 2]
            c22 = cov_cam[:, 2, 2]
            cov_a = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22
            cov_b = j00 * j11 * c01 + j00 * j12 * c02 + j02 * j11 * c12 + j02 * j12 * c22
            cov_c = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22
            # 0.3px low-pass blur and stability clamps (match the legacy sigma clamps).
            cov_a = (cov_a + 0.3).clamp(min=0.25, max=512.0 ** 2)
            cov_c = (cov_c + 0.3).clamp(min=0.25, max=512.0 ** 2)
            b_max = 0.99 * torch.sqrt(cov_a * cov_c)
            cov_b = torch.maximum(torch.minimum(cov_b, b_max), -b_max)
            det = (cov_a * cov_c - cov_b * cov_b).clamp(min=1e-8)
            conic_a = cov_c / det
            conic_b = -cov_b / det
            conic_c = cov_a / det
            rad_x_f = 3.0 * torch.sqrt(cov_a)
            rad_y_f = 3.0 * torch.sqrt(cov_c)
        else:
            # FISHEYE / EQUIRECTANGULAR: the pixel-space Jacobian of these
            # projections is strongly nonlinear and direction dependent (it
            # degenerates near the poles / image border), so we keep the legacy
            # isotropic approximation (mean scale / depth) instead of a
            # projected 2D covariance.
            conic_a = 1.0 / (sigma_x * sigma_x)
            conic_b = torch.zeros_like(sigma_x)
            conic_c = 1.0 / (sigma_y * sigma_y)
            rad_x_f = 3.0 * sigma_x
            rad_y_f = 3.0 * sigma_y

        max_radius = max(1, int(max_radius))
        chunk_size = max(1, int(chunk_size))
        total_px = output_height * output_width
        alpha_sum = torch.zeros((total_px,), device=dev)
        color_sum = torch.zeros((total_px, 3), device=dev)
        depth_sum = torch.zeros((total_px,), device=dev)
        n_splats = px.shape[0]
        rad_x_all = torch.ceil(rad_x_f).detach().to(torch.int64).clamp(min=1, max=max_radius)
        rad_y_all = torch.ceil(rad_y_f).detach().to(torch.int64).clamp(min=1, max=max_radius)

        def _splat_chunk(px_c, py_c, conic_a_c, conic_b_c, conic_c_c, opacity_c, colors_c, depth_c, rad_x, rad_y):
            """One chunk's scatter contributions: (idx, alpha, color, depth) flats."""
            conic_a_c = conic_a_c.view(-1, 1, 1)
            conic_b_c = conic_b_c.view(-1, 1, 1)
            conic_c_c = conic_c_c.view(-1, 1, 1)
            opacity_c = opacity_c.clamp(0.0, 1.0)
            max_rx = int(rad_x.max().detach().cpu().item())
            max_ry = int(rad_y.max().detach().cpu().item())
            empty = (
                torch.zeros((0,), device=dev, dtype=torch.int64),
                torch.zeros((0,), device=dev),
                torch.zeros((0, 3), device=dev),
                torch.zeros((0,), device=dev),
            )
            if max_rx <= 0 or max_ry <= 0:
                return empty

            # Window [floor(px - rad), floor(px - rad) + 2*max_r] covers the full
            # [px - rad, px + rad] footprint of every splat in the chunk. (The
            # previous arange(-max_r, max_r+1) offset from the left edge cut off
            # the right/bottom half of each footprint.)
            grid_x = torch.arange(0, 2 * max_rx + 1, device=dev)
            grid_y = torch.arange(0, 2 * max_ry + 1, device=dev)
            x0 = torch.floor(px_c.detach() - rad_x.float()).view(-1, 1, 1)
            y0 = torch.floor(py_c.detach() - rad_y.float()).view(-1, 1, 1)

            xs = x0 + grid_x.view(1, 1, -1)
            ys = y0 + grid_y.view(1, -1, 1)

            dx = xs - px_c.view(-1, 1, 1)
            dy = ys - py_c.view(-1, 1, 1)
            quad = conic_a_c * dx * dx + 2.0 * conic_b_c * dx * dy + conic_c_c * dy * dy
            weight = torch.exp(-0.5 * quad)

            xs_int = xs.to(torch.int64)
            ys_int = ys.to(torch.int64)
            valid_px = (
                (xs_int >= 0)
                & (xs_int < output_width)
                & (ys_int >= 0)
                & (ys_int < output_height)
                & (quad.detach() <= 9.0)
            )

            alpha = opacity_c.view(-1, 1, 1) * weight
            alpha = alpha * valid_px
            valid_flat = valid_px.expand(alpha.shape).reshape(-1)
            if not valid_flat.any():
                return empty

            idx = (ys_int * output_width + xs_int).expand(alpha.shape).reshape(-1)[valid_flat]
            alpha_flat = alpha.reshape(-1)[valid_flat]
            color_flat = (alpha.unsqueeze(-1) * colors_c.view(-1, 1, 1, 3)).reshape(-1, 3)[valid_flat]
            depth_flat = (alpha * depth_c.view(-1, 1, 1)).reshape(-1)[valid_flat]
            return idx, alpha_flat, color_flat, depth_flat

        # When gradients are required (e.g. SplatPolish's torch fallback),
        # gradient-checkpoint each chunk: otherwise autograd retains every
        # chunk's [chunk, 2r+1, 2r+1] intermediates (exp weights, alpha, color
        # products, ...) until backward, and memory scales with
        # n_splats x footprint — OOM at realistic splat counts. Checkpointing
        # recomputes the chunk during backward instead.
        needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (px, py, conic_a, conic_b, conic_c, opacity, colors, depth)
        )
        if needs_grad:
            from torch.utils.checkpoint import checkpoint as _torch_checkpoint

        for start in range(0, n_splats, chunk_size):
            end = min(n_splats, start + chunk_size)
            chunk_args = (
                px[start:end],
                py[start:end],
                conic_a[start:end],
                conic_b[start:end],
                conic_c[start:end],
                opacity[start:end],
                colors[start:end],
                depth[start:end],
                rad_x_all[start:end],
                rad_y_all[start:end],
            )
            if needs_grad:
                idx, alpha_flat, color_flat, depth_flat = _torch_checkpoint(
                    _splat_chunk, *chunk_args, use_reentrant=False
                )
            else:
                idx, alpha_flat, color_flat, depth_flat = _splat_chunk(*chunk_args)
            if idx.numel() == 0:
                continue

            alpha_sum.scatter_add_(0, idx, alpha_flat)
            color_sum.scatter_add_(0, idx.unsqueeze(-1).expand(-1, 3), color_flat)
            depth_sum.scatter_add_(0, idx, depth_flat)

        alpha_img = alpha_sum.view(output_height, output_width).clamp(max=1.0)
        color_img = color_sum.view(output_height, output_width, 3) / alpha_sum.view(output_height, output_width, 1).clamp(min=1e-6)
        depth_img = depth_sum.view(output_height, output_width) / alpha_sum.view(output_height, output_width).clamp(min=1e-6)
        disparity = (1.0 / depth_img.clamp(min=1e-6)) * alpha_img
        disparity = disparity.unsqueeze(0).unsqueeze(-1)
        return color_img.unsqueeze(0), alpha_img, disparity

    order = torch.argsort(depth)
    order_cpu = order.detach().cpu().tolist()
    px_cpu = px.detach().cpu().numpy()
    py_cpu = py.detach().cpu().numpy()
    sx_cpu = sigma_x.detach().cpu().numpy()
    sy_cpu = sigma_y.detach().cpu().numpy()

    img = torch.zeros((output_height, output_width, 3), device=dev)
    alpha_img = torch.zeros((output_height, output_width), device=dev)
    depth_acc = torch.zeros((output_height, output_width), device=dev)

    for idx in order_cpu:
        cx = float(px_cpu[idx])
        cy = float(py_cpu[idx])
        sx = float(sx_cpu[idx])
        sy = float(sy_cpu[idx])
        if sx <= 0.0 or sy <= 0.0:
            continue
        radius_x = int(math.ceil(3.0 * sx))
        radius_y = int(math.ceil(3.0 * sy))
        x0 = max(0, int(math.floor(cx - radius_x)))
        x1 = min(output_width - 1, int(math.ceil(cx + radius_x)))
        y0 = max(0, int(math.floor(cy - radius_y)))
        y1 = min(output_height - 1, int(math.ceil(cy + radius_y)))
        if x1 < x0 or y1 < y0:
            continue

        xs = torch.arange(x0, x1 + 1, device=dev)
        ys = torch.arange(y0, y1 + 1, device=dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = (xx - cx) / sx
        dy = (yy - cy) / sy
        weight = torch.exp(-0.5 * (dx * dx + dy * dy))
        alpha = opacity[idx].clamp(0.0, 1.0) * weight
        if alpha.max() <= 0.0:
            continue
        sub_alpha = alpha_img[y0 : y1 + 1, x0 : x1 + 1]
        trans = 1.0 - sub_alpha
        alpha = alpha.clamp(0.0, 1.0)
        sub_color = img[y0 : y1 + 1, x0 : x1 + 1]
        sub_color = sub_color + trans.unsqueeze(-1) * alpha.unsqueeze(-1) * colors[idx]
        sub_alpha = sub_alpha + trans * alpha
        sub_depth = depth_acc[y0 : y1 + 1, x0 : x1 + 1]
        sub_depth = sub_depth + trans * alpha * depth[idx]
        img[y0 : y1 + 1, x0 : x1 + 1] = sub_color
        alpha_img[y0 : y1 + 1, x0 : x1 + 1] = sub_alpha
        depth_acc[y0 : y1 + 1, x0 : x1 + 1] = sub_depth

    depth_img = depth_acc / alpha_img.clamp(min=1e-6)
    disparity = (1.0 / depth_img.clamp(min=1e-6)) * alpha_img
    disparity = disparity.unsqueeze(0).unsqueeze(-1)
    return img.unsqueeze(0), alpha_img, disparity


class LoadPlySplat:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".ply")
        ]
        return {
            "required": {
                "splat_file": (
                    sorted(files),
                    {
                        "file_chooser": True,
                        "tooltip": "Select a 3DGS .ply file to load from your input folder."
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("splats",)
    FUNCTION = "load_splats"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Loads a 3D Gaussian Splatting PLY file into a GSPLAT object."

    def load_splats(self, splat_file: str, device: str = "auto"):
        path = folder_paths.get_annotated_filepath(splat_file)
        data = _read_ply_vertices(path)

        required = [
            "x", "y", "z",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"PLY is missing required properties: {missing}")

        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        scale = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
        rotation = np.stack(
            [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]],
            axis=1,
        ).astype(np.float32)
        opacity = data["opacity"].astype(np.float32).reshape(-1, 1)
        f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)
        f_rest, sh_order = _extract_f_rest(data)

        splats = GaussianSplats(
            xyz=torch.from_numpy(xyz),
            scale=torch.from_numpy(scale),
            rotation=torch.from_numpy(rotation),
            opacity=torch.from_numpy(opacity),
            f_dc=torch.from_numpy(f_dc),
            f_rest=torch.from_numpy(f_rest),
            sh_order=sh_order,
        )
        target_device = _resolve_device_choice(device)
        if splats.xyz.device != target_device:
            splats = splats.to(target_device)
        return (splats,)

    @classmethod
    def IS_CHANGED(cls, splat_file: str):
        path = folder_paths.get_annotated_filepath(splat_file)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, splat_file: str):
        if not folder_paths.exists_annotated_filepath(splat_file):
            return f"Invalid splat file: {splat_file}"
        return True


class ImageToSplat:
    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_sharp_checkpoint_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_fov": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 1.0,
                        "max": 179.0,
                        "tooltip": "Horizontal field of view in degrees used to compute focal length.",
                    },
                ),
                "checkpoint": (
                    choices,
                    {
                        "default": _SHARP_DEFAULT_CHECKPOINT_LABEL,
                        "file_chooser": True,
                        "tooltip": "Select a .pt checkpoint from the input folder or download the default model.",
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("splats",)
    FUNCTION = "image_to_splat"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Predicts Gaussian splats from an image using SHARP."

    @torch.no_grad()
    def image_to_splat(
        self,
        image: torch.Tensor,
        horizontal_fov: float,
        checkpoint: str,
        device: str = "auto",
    ):
        _ensure_sharp_available()
        target_device = _resolve_device_choice(device)

        image_np = _tensor_image_to_numpy(image)
        height, width = image_np.shape[:2]
        if height < 2 or width < 2:
            raise ValueError("Input image is too small for SHARP.")

        f_px = _horizontal_fov_to_f_px(width, horizontal_fov)
        checkpoint_path = None
        if checkpoint and checkpoint != _SHARP_DEFAULT_CHECKPOINT_LABEL:
            checkpoint_path = folder_paths.get_annotated_filepath(checkpoint)

        predictor = _load_sharp_predictor(checkpoint_path, target_device)
        gaussians = _sharp_predict_image(predictor, image_np, float(f_px), target_device)

        mean_vectors = gaussians.mean_vectors[0] if gaussians.mean_vectors.dim() == 3 else gaussians.mean_vectors
        singular_values = gaussians.singular_values[0] if gaussians.singular_values.dim() == 3 else gaussians.singular_values
        quaternions = gaussians.quaternions[0] if gaussians.quaternions.dim() == 3 else gaussians.quaternions
        colors = gaussians.colors[0] if gaussians.colors.dim() == 3 else gaussians.colors
        opacities = gaussians.opacities[0] if gaussians.opacities.dim() == 2 else gaussians.opacities

        mean_vectors = mean_vectors.to(device=target_device, dtype=torch.float32)
        singular_values = singular_values.to(device=target_device, dtype=torch.float32)
        quaternions = quaternions.to(device=target_device, dtype=torch.float32)
        colors = colors.to(device=target_device, dtype=torch.float32)
        opacities = opacities.to(device=target_device, dtype=torch.float32)

        scale_logits = torch.log(singular_values.clamp(min=1e-9))
        opacity = opacities.clamp(1e-6, 1.0 - 1e-6).view(-1, 1)
        opacity_logits = torch.log(opacity / (1.0 - opacity))

        colors_srgb = _sharp_color_space.linearRGB2sRGB(colors.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        f_dc = _sharp_rgb_to_sh(colors_srgb).to(dtype=mean_vectors.dtype)
        f_rest = torch.zeros((mean_vectors.shape[0], 0), device=target_device, dtype=mean_vectors.dtype)

        splats = GaussianSplats(
            xyz=mean_vectors,
            scale=scale_logits,
            rotation=quaternions,
            opacity=opacity_logits,
            f_dc=f_dc,
            f_rest=f_rest,
            sh_order=0,
        )
        return (splats,)


class FisheyeToGaussian:
    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_sharp_checkpoint_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "fisheye_horizontal_fov": (
                    "FLOAT",
                    {
                        "default": 180.0,
                        "min": 1.0,
                        "max": 360.0,
                        "tooltip": "Horizontal field of view for the fisheye input.",
                    },
                ),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "checkpoint": (
                    choices,
                    {
                        "default": _SHARP_DEFAULT_CHECKPOINT_LABEL,
                        "file_chooser": True,
                        "tooltip": "Select a .pt checkpoint from the input folder or download the default model.",
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
                "pinhole_horizontal_fov": (
                    "FLOAT",
                    {"default": 90.0, "min": 1.0, "max": 179.0},
                ),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 512}),
                "stitch_mode": (
                    ["keep", "discard", "average", "smart", "main_direction"],
                    {"default": "smart"},
                ),
                "stitch_voxel_size": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 10.0},
                ),
                "stitch_direction_deg": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 45.0},
                ),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("splats",)
    FUNCTION = "fisheye_to_gaussian"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Reprojects fisheye views to multiple pinhole angles, predicts splats, rotates and merges them."

    @torch.no_grad()
    def fisheye_to_gaussian(
        self,
        image: torch.Tensor,
        fisheye_horizontal_fov: float,
        output_width: int,
        output_height: int,
        checkpoint: str,
        device: str = "auto",
        pinhole_horizontal_fov: float = 90.0,
        feathering: int = 0,
        stitch_mode: str = "smart",
        stitch_voxel_size: float = 0.01,
        stitch_direction_deg: float = 5.0,
    ):
        _ensure_sharp_available()
        if ReprojectImage is None:
            raise ModuleNotFoundError("ReprojectImage is unavailable; reprojection_nodes could not be imported.")

        image_tensor = image
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.dim() != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")

        _, height, width, _ = image_tensor.shape
        if output_width <= 0:
            output_width = int(width)
        if output_height <= 0:
            output_height = int(height)

        image_to_splat = ImageToSplat()
        reproject = ReprojectImage()

        view_angles = [
            (0.0, 0.0),
            (0.0, 45.0),
            (0.0, -45.0),
            (45.0, 0.0),
            (-45.0, 0.0),
        ]

        splats_list: List[GaussianSplats] = []
        for theta, phi in view_angles:
            transform = _build_rotation_matrix(theta, phi)
            reproj_image, _ = reproject.reproject_image(
                image_tensor,
                fisheye_horizontal_fov,
                pinhole_horizontal_fov,
                "FISHEYE",
                "PINHOLE",
                output_width,
                output_height,
                feathering,
                False,
                transform,
                None,
            )

            splats, = image_to_splat.image_to_splat(
                reproj_image,
                pinhole_horizontal_fov,
                checkpoint,
                device,
            )
            if theta != 0.0 or phi != 0.0:
                splats = splat_cloud_rotation(splats, transform)
            splats_list.append(splats)

        merged = _stitch_splats(
            splats_list,
            stitch_mode,
            stitch_voxel_size,
            stitch_direction_deg,
            pinhole_horizontal_fov,
        )
        return (merged,)


class RotateSplats:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats": ("GSPLAT",),
                "transform_matrix": ("MAT_4X4",),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("rotated_splats",)
    FUNCTION = "rotate_splats"
    CATEGORY = "Camera/GSplat"

    def rotate_splats(self, splats: GaussianSplats, transform_matrix: torch.Tensor, device: str = "auto"):
        target_device = _resolve_device_choice(device)
        if splats.xyz.device != target_device:
            splats = splats.to(target_device)
        return (splat_cloud_rotation(splats, transform_matrix),)


class MergeSplats:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats_a": ("GSPLAT",),
                "splats_b": ("GSPLAT",),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("merged_splats",)
    FUNCTION = "merge_splats"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Merges two GSPLAT objects into one."

    def merge_splats(self, splats_a: GaussianSplats, splats_b: GaussianSplats, device: str = "auto"):
        if splats_a.f_rest.shape[1] != splats_b.f_rest.shape[1] or splats_a.sh_order != splats_b.sh_order:
            raise ValueError(
                f"Splats must share the same SH order and f_rest size (got {splats_a.sh_order}/{splats_a.f_rest.shape[1]} vs "
                f"{splats_b.sh_order}/{splats_b.f_rest.shape[1]})."
            )

        if device == "auto":
            if splats_a.xyz.device == splats_b.xyz.device:
                target_device = splats_a.xyz.device
            else:
                target_device = _resolve_device_choice("auto")
        else:
            target_device = _resolve_device_choice(device)

        dtype = torch.promote_types(splats_a.xyz.dtype, splats_b.xyz.dtype)
        splats_a = _coerce_splats(splats_a, target_device, dtype)
        splats_b = _coerce_splats(splats_b, target_device, dtype)

        merged = GaussianSplats(
            xyz=torch.cat([splats_a.xyz, splats_b.xyz], dim=0),
            scale=torch.cat([splats_a.scale, splats_b.scale], dim=0),
            rotation=torch.cat([splats_a.rotation, splats_b.rotation], dim=0),
            opacity=torch.cat([splats_a.opacity, splats_b.opacity], dim=0),
            f_dc=torch.cat([splats_a.f_dc, splats_b.f_dc], dim=0),
            f_rest=torch.cat([splats_a.f_rest, splats_b.f_rest], dim=0),
            sh_order=splats_a.sh_order,
        )
        return (merged,)


class RenderSplat:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats": ("GSPLAT",),
                "camera_matrix": ("MAT_4X4",),
                "camera_projection": (Projection.PROJECTIONS, {}),
                "camera_horizontal_fov": ("FLOAT", {"default": 90.0}),
                "output_width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "output_height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "max_splats": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "Keep only the N most opaque splats. 0 = unlimited."}),
                "opacity_is_logit": ("BOOLEAN", {"default": True}),
                "add_sh_bias": ("BOOLEAN", {"default": True}),
                "render_mode": (RENDER_MODES_ALL, {"default": "auto", "tooltip": "auto = gsplat when available (CUDA + PINHOLE), otherwise the torch 'fast' splatter."}),
                "chunk_size": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "max_radius": ("INT", {"default": 32, "min": 1, "max": 512}),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("image", "mask", "disparity")
    FUNCTION = "render_splats"
    CATEGORY = "Camera/GSplat"

    def render_splats(
        self,
        splats: GaussianSplats,
        camera_matrix: torch.Tensor,
        camera_projection: str,
        camera_horizontal_fov: float,
        output_width: int,
        output_height: int,
        max_splats: int,
        opacity_is_logit: bool,
        add_sh_bias: bool = True,
        render_mode: str = "auto",
        chunk_size: int = 256,
        max_radius: int = 32,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return render_gaussians(
            splats,
            camera_matrix,
            camera_projection,
            camera_horizontal_fov,
            output_width,
            output_height,
            max_splats=max_splats,
            opacity_is_logit=opacity_is_logit,
            add_sh_bias=add_sh_bias,
            render_mode=render_mode,
            chunk_size=chunk_size,
            max_radius=max_radius,
            device=device,
        )


class SavePlySplat:
    """
    Save a Gaussian Splat PLY to the ComfyUI output directory.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "splat"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats": ("GSPLAT",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUIGSplat",
                        "tooltip": "Prefix for the .ply file. You can include format-tokens like %date:yyyy-MM-dd%."
                    }
                ),
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_splats"
    OUTPUT_NODE = True
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Saves the input GSPLAT to your ComfyUI output directory as a .ply file."

    def save_splats(self, splats: GaussianSplats, filename_prefix: str):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                0, 0
            )
        os.makedirs(full_output_folder, exist_ok=True)
        base_name = filename.replace("%batch_num%", "0")
        ply_name = f"{base_name}_{counter:05}.ply"
        ply_path = os.path.join(full_output_folder, ply_name)
        _write_ply_splats(ply_path, splats)
        counter += 1
        return {
            "ui": {
                "splats": [{
                    "filename": ply_name,
                    "subfolder": subfolder,
                    "type": self.type
                }]
            }
        }


class FuseSplats:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats_a": ("GSPLAT",),
                "splats_b": ("GSPLAT",),
                "voxel_size": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.001,
                        "tooltip": "Voxel edge length used to merge overlapping splats. 0 disables voxel merging.",
                    },
                ),
                "mode": (FUSE_MODES, {"default": "smart"}),
                "weight_a": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "tooltip": "Confidence/recency weight for splats_a (used by smart/average modes).",
                    },
                ),
                "weight_b": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "tooltip": "Confidence/recency weight for splats_b (used by smart/average modes).",
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("fused_splats",)
    FUNCTION = "fuse_splats"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = "Fuses two splat clouds with weighted voxel merging (weights bias the per-voxel reduction)."

    def fuse_splats(
        self,
        splats_a: GaussianSplats,
        splats_b: GaussianSplats,
        voxel_size: float,
        mode: str,
        weight_a: float,
        weight_b: float,
        device: str = "auto",
    ):
        if splats_a.f_rest.shape[1] != splats_b.f_rest.shape[1] or splats_a.sh_order != splats_b.sh_order:
            raise ValueError(
                f"Splats must share the same SH order and f_rest size (got {splats_a.sh_order}/{splats_a.f_rest.shape[1]} vs "
                f"{splats_b.sh_order}/{splats_b.f_rest.shape[1]})."
            )
        if device == "auto":
            if splats_a.xyz.device == splats_b.xyz.device:
                target_device = splats_a.xyz.device
            else:
                target_device = _resolve_device_choice("auto")
        else:
            target_device = _resolve_device_choice(device)
        dtype = torch.promote_types(splats_a.xyz.dtype, splats_b.xyz.dtype)
        a = _coerce_splats(splats_a, target_device, dtype)
        b = _coerce_splats(splats_b, target_device, dtype)
        fused = _stitch_splats(
            [a, b],
            mode,
            voxel_size,
            5.0,
            weights_list=[float(weight_a), float(weight_b)],
        )
        return (fused,)


class VideoToFusedSplats:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        choices = _list_sharp_checkpoint_choices()
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video frames [T,H,W,3]."}),
                "trajectory": (
                    "TENSOR",
                    {"tooltip": "[T,4,4] world-to-camera matrix per frame (a single [4,4] is broadcast)."},
                ),
                "horizontal_fov": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0}),
                "checkpoint": (
                    choices,
                    {
                        "default": _SHARP_DEFAULT_CHECKPOINT_LABEL,
                        "file_chooser": True,
                        "tooltip": "SHARP .pt checkpoint from the input folder, or download the default model.",
                    },
                ),
                "keyframe_stride": (
                    "INT",
                    {"default": 8, "min": 1, "max": 1000, "tooltip": "Run SHARP on every Nth frame."},
                ),
                "stitch_voxel_size": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.001}),
                "stitch_mode": (FUSE_MODES, {"default": "smart"}),
            },
            "optional": {
                "static_mask": (
                    "MASK",
                    {"tooltip": "[T,H,W], 1 = static/keep pixel. Splats whose source pixel has mask < 0.5 are dropped."},
                ),
                "depths": (
                    "TENSOR",
                    {"tooltip": "[T,H,W] metric depths. SHARP splats are scale-aligned per keyframe via a robust median disparity ratio."},
                ),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("splats",)
    FUNCTION = "video_to_fused_splats"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = (
        "Runs SHARP on video keyframes, optionally scale-aligns to metric depth and filters dynamic pixels, "
        "transforms each keyframe splat cloud to the world frame via the inverse camera pose, and fuses everything "
        "incrementally into a single world-frame splat cloud."
    )

    @torch.no_grad()
    def video_to_fused_splats(
        self,
        frames: torch.Tensor,
        trajectory,
        horizontal_fov: float,
        checkpoint: str,
        keyframe_stride: int = 8,
        stitch_voxel_size: float = 0.01,
        stitch_mode: str = "smart",
        static_mask: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
        device: str = "auto",
    ):
        _ensure_sharp_available()
        target_device = _resolve_device_choice(device)
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError("frames must be an IMAGE tensor [T,H,W,C].")
        num_frames = int(frames.shape[0])
        height = int(frames.shape[1])
        width = int(frames.shape[2])
        traj = _coerce_trajectory(trajectory, num_frames, target_device)
        depth_seq = _normalize_map_sequence(depths, num_frames, "depths") if depths is not None else None
        mask_seq = _normalize_map_sequence(static_mask, num_frames, "static_mask") if static_mask is not None else None

        keyframes = list(range(0, num_frames, max(1, int(keyframe_stride))))
        image_to_splat = ImageToSplat()
        keyframe_clouds: List[GaussianSplats] = []
        for i in _progress(keyframes, desc="VideoToFusedSplats"):
            splats, = image_to_splat.image_to_splat(frames[i : i + 1], horizontal_fov, checkpoint, device)
            if splats.xyz.device != target_device:
                splats = splats.to(target_device)
            if len(splats) == 0:
                continue

            if depth_seq is not None:
                depth_i = depth_seq[i if depth_seq.shape[0] > 1 else 0].to(target_device)
                px, py, z, ok = _project_splats_to_pixels(splats.xyz, horizontal_fov, width, height)
                d_ref = _sample_map_at_pixels(depth_i, px, py, width, height)
                ok = ok & (d_ref > 1e-6) & (z > 1e-6)
                if ok.any():
                    # Robust scale in the disparity domain:
                    # median((1/z_sharp) / (1/d_ref)) == median(d_ref / z_sharp).
                    s = torch.median(d_ref[ok] / z[ok])
                    if torch.isfinite(s) and float(s) > 1e-6:
                        splats = GaussianSplats(
                            xyz=splats.xyz * s,
                            scale=splats.scale + torch.log(s),
                            rotation=splats.rotation,
                            opacity=splats.opacity,
                            f_dc=splats.f_dc,
                            f_rest=splats.f_rest,
                            sh_order=splats.sh_order,
                        )

            if mask_seq is not None:
                mask_i = mask_seq[i if mask_seq.shape[0] > 1 else 0].to(target_device)
                px, py, _z, ok = _project_splats_to_pixels(splats.xyz, horizontal_fov, width, height)
                mask_values = _sample_map_at_pixels(mask_i, px, py, width, height)
                drop = ok & (mask_values < 0.5)
                splats = splats[~drop]
                if len(splats) == 0:
                    continue

            # trajectory is world-to-camera; the splats live in the camera frame,
            # so camera-to-world = inverse(pose) brings them into the world frame.
            cam_to_world = torch.linalg.inv(traj[i])
            splats_world = splat_cloud_rotation(splats, cam_to_world)
            keyframe_clouds.append(splats_world)
        if not keyframe_clouds:
            raise ValueError("No splats were produced from the provided frames.")
        # Fuse with a SINGLE voxel reduce over all keyframe clouds. Re-stitching
        # the whole accumulated cloud on every keyframe (the previous approach)
        # is O(keyframes x N) work and peak memory: each iteration re-copied and
        # re-unique-sorted the entire accumulated cloud, ballooning runtime and
        # OOMing on long clips.
        if len(keyframe_clouds) == 1:
            accumulated = keyframe_clouds[0]
        else:
            accumulated = _stitch_splats(
                keyframe_clouds,
                stitch_mode,
                stitch_voxel_size,
                5.0,
            )
        return (accumulated,)


class SplatPolish:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splats": ("GSPLAT",),
                "frames": ("IMAGE", {"tooltip": "Ground-truth frames [T,H,W,3]."}),
                "trajectory": ("TENSOR", {"tooltip": "[T,4,4] world-to-camera matrix per frame."}),
                "horizontal_fov": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0}),
                "iterations": ("INT", {"default": 300, "min": 1, "max": 100000}),
                "lr_xyz": ("FLOAT", {"default": 1.6e-4, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "lr_rest": (
                    "FLOAT",
                    {
                        "default": 2.5e-3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.0001,
                        "tooltip": "Base learning rate for non-position parameters (3DGS-style ratios applied per group).",
                    },
                ),
                "lambda_l1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0}),
                "lambda_dssim": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0}),
                "opacity_reg": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0}),
                "allow_torch_fallback": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Without gsplat+CUDA, optimize through the differentiable torch renderer at reduced resolution. EXTREMELY slow; expect minutes per 100 iterations.",
                    },
                ),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("polished_splats",)
    FUNCTION = "polish_splats"
    CATEGORY = "Camera/GSplat"
    DESCRIPTION = (
        "Optimizes an existing world-frame splat cloud against posed video frames "
        "(L1 + D-SSIM photometric loss) using gsplat's differentiable rasterizer."
    )

    def polish_splats(
        self,
        splats: GaussianSplats,
        frames: torch.Tensor,
        trajectory,
        horizontal_fov: float,
        iterations: int,
        lr_xyz: float,
        lr_rest: float,
        lambda_l1: float,
        lambda_dssim: float,
        opacity_reg: float,
        allow_torch_fallback: bool = False,
        device: str = "auto",
    ):
        target_device = _resolve_device_choice(device)
        use_gsplat = (
            target_device.type == "cuda"
            and torch.cuda.is_available()
            and _gsplat_available()
        )
        if not use_gsplat and not allow_torch_fallback:
            raise RuntimeError(
                "SplatPolish requires gsplat with CUDA (install with: pip install gsplat). "
                "Alternatively enable allow_torch_fallback to optimize through the pure-torch "
                "renderer at reduced resolution (extremely slow)."
            )

        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError("frames must be an IMAGE tensor [T,H,W,C].")
        frames = frames[..., :3].float()
        num_frames = int(frames.shape[0])
        frame_h = int(frames.shape[1])
        frame_w = int(frames.shape[2])
        traj = _coerce_trajectory(trajectory, num_frames, target_device)

        render_w, render_h = frame_w, frame_h
        if not use_gsplat:
            # The torch fallback renderer is O(pixels x splats); shrink the target.
            max_dim = 256
            scale_factor = min(1.0, max_dim / max(frame_w, frame_h))
            render_w = max(8, int(round(frame_w * scale_factor)))
            render_h = max(8, int(round(frame_h * scale_factor)))
        if (render_w, render_h) != (frame_w, frame_h):
            frames_chw = frames.permute(0, 3, 1, 2)
            frames_chw = F.interpolate(frames_chw, size=(render_h, render_w), mode="bilinear", align_corners=False)
            frames = frames_chw.permute(0, 2, 3, 1).contiguous()
        # Keep the ground-truth frames where they arrived (normally CPU): each
        # iteration samples a single random frame, so only that frame is moved
        # to the target device. Uploading the whole clip up front would pin
        # ~T*H*W*3*4 bytes of VRAM (about 5GB for 200 frames at 1080p) on top
        # of the rasterization buffers and optimizer state.
        frames = frames.contiguous()

        base = splats.to(target_device)
        xyz = base.xyz.detach().clone().float().requires_grad_(True)
        scale = base.scale.detach().clone().float().requires_grad_(True)
        rotation = base.rotation.detach().clone().float().requires_grad_(True)
        opacity = base.opacity.detach().clone().float().requires_grad_(True)
        f_dc = base.f_dc.detach().clone().float().requires_grad_(True)
        has_rest = base.f_rest.shape[1] > 0
        f_rest = base.f_rest.detach().clone().float()
        if has_rest:
            f_rest.requires_grad_(True)

        # Learning-rate ratios follow the standard 3DGS recipe, scaled by lr_rest.
        param_groups = [
            {"params": [xyz], "lr": lr_xyz},
            {"params": [f_dc], "lr": lr_rest},
            {"params": [opacity], "lr": lr_rest * 20.0},
            {"params": [scale], "lr": lr_rest * 2.0},
            {"params": [rotation], "lr": lr_rest * 0.4},
        ]
        if has_rest:
            param_groups.append({"params": [f_rest], "lr": lr_rest / 20.0})
        optimizer = torch.optim.Adam(param_groups, eps=1e-15)

        total_sh = (base.sh_order + 1) ** 2
        fov_rad = math.radians(horizontal_fov)
        f_px = 0.5 * render_w / math.tan(fov_rad / 2.0)
        K = torch.tensor(
            [
                [f_px, 0.0, render_w / 2.0],
                [0.0, f_px, render_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            device=target_device,
            dtype=torch.float32,
        )
        gsplat_mod = _import_gsplat() if use_gsplat else None

        for _ in _progress(range(int(iterations)), desc="SplatPolish"):
            frame_idx = int(torch.randint(0, num_frames, (1,)).item())
            target = frames[frame_idx].to(target_device)
            pose = traj[frame_idx]

            if use_gsplat:
                quats = rotation / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                sh = torch.cat([f_dc, f_rest], dim=1).view(-1, 3, total_sh).transpose(1, 2)
                renders, _alphas, _meta = gsplat_mod.rasterization(
                    means=xyz,
                    quats=quats,
                    scales=torch.exp(scale),
                    opacities=torch.sigmoid(opacity).view(-1),
                    colors=sh,
                    viewmats=pose.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=render_w,
                    height=render_h,
                    sh_degree=int(base.sh_order),
                    render_mode="RGB",
                )
                pred = renders[0, ..., :3].clamp(0.0, 1.0)
            else:
                current = GaussianSplats(
                    xyz=xyz,
                    scale=scale,
                    rotation=rotation,
                    opacity=opacity,
                    f_dc=f_dc,
                    f_rest=f_rest,
                    sh_order=base.sh_order,
                )
                img, _mask, _disp = render_gaussians(
                    current,
                    pose,
                    "PINHOLE",
                    horizontal_fov,
                    render_w,
                    render_h,
                    max_splats=0,
                    opacity_is_logit=True,
                    add_sh_bias=True,
                    render_mode="fast",
                    device=target_device.type,
                )
                pred = img[0]
            if not pred.requires_grad:
                continue  # nothing visible from this pose

            l1 = (pred - target).abs().mean()
            loss = lambda_l1 * l1
            if lambda_dssim > 0.0:
                ssim_val = _ssim(
                    pred.permute(2, 0, 1).unsqueeze(0),
                    target.permute(2, 0, 1).unsqueeze(0),
                )
                loss = loss + lambda_dssim * (1.0 - ssim_val)
            if opacity_reg > 0.0:
                loss = loss + opacity_reg * torch.sigmoid(opacity).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rotation.data = rotation.data / rotation.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                opacity.data.clamp_(-15.0, 15.0)
                scale.data.clamp_(-12.0, 6.0)

        polished = GaussianSplats(
            xyz=xyz.detach().clone(),
            scale=scale.detach().clone(),
            rotation=(rotation / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)).detach().clone(),
            opacity=opacity.detach().clone(),
            f_dc=f_dc.detach().clone(),
            f_rest=f_rest.detach().clone(),
            sh_order=base.sh_order,
        )
        return (polished,)


NODE_CLASS_MAPPINGS = {
    "LoadPlySplat": LoadPlySplat,
    "ImageToSplat": ImageToSplat,
    "FisheyeToGaussian": FisheyeToGaussian,
    "RotateSplats": RotateSplats,
    "MergeSplats": MergeSplats,
    "RenderSplat": RenderSplat,
    "SavePlySplat": SavePlySplat,
    "FuseSplats": FuseSplats,
    "VideoToFusedSplats": VideoToFusedSplats,
    "SplatPolish": SplatPolish,
}
