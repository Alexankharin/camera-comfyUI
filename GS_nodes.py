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
            f"ml-sharpt is unavailable. Ensure submodules/ml-sharpt is present and its dependencies are installed. "
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


def _filter_overlapping_by_direction(
    main: GaussianSplats,
    other: GaussianSplats,
    angle_deg: float,
) -> GaussianSplats:
    if len(other) == 0:
        return other
    main_bins, _ = _direction_bins(main.xyz, angle_deg)
    main_bins = torch.unique(main_bins)
    other_bins, _ = _direction_bins(other.xyz, angle_deg)
    main_bins_sorted, _ = torch.sort(main_bins)
    idx = torch.searchsorted(main_bins_sorted, other_bins)
    in_main = (idx < main_bins_sorted.numel()) & (main_bins_sorted[idx] == other_bins)
    keep = ~in_main
    return other[keep]


def _stitch_splats(
    splats_list: List[GaussianSplats],
    mode: str,
    voxel_size: float,
    direction_deg: float,
) -> GaussianSplats:
    if mode == "main_direction":
        if not splats_list:
            raise ValueError("No splats provided to merge.")
        main = splats_list[0]
        filtered = [main]
        for splats in splats_list[1:]:
            filtered.append(_filter_overlapping_by_direction(main, splats, direction_deg))
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

    sum_w = torch.zeros((num_voxels,), device=device, dtype=dtype)
    sum_w.scatter_add_(0, inv, weights)
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

    return GaussianSplats(
        xyz=xyz,
        scale=scale,
        rotation=rotation,
        opacity=opacity_logits,
        f_dc=f_dc,
        f_rest=f_rest,
        sh_order=merged.sh_order,
    )


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

        merged = _stitch_splats(splats_list, stitch_mode, stitch_voxel_size, stitch_direction_deg)
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
                "max_splats": ("INT", {"default": 20000, "min": 0, "max": 1000000}),
                "opacity_is_logit": ("BOOLEAN", {"default": True}),
                "add_sh_bias": ("BOOLEAN", {"default": True}),
                "render_mode": (RENDER_MODES, {"default": "fast"}),
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
        render_mode: str = "fast",
        chunk_size: int = 256,
        max_radius: int = 32,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_device = _resolve_device_choice(device)
        if splats.xyz.device != target_device:
            splats = splats.to(target_device)
        device = splats.xyz.device
        if isinstance(camera_matrix, torch.Tensor):
            M = camera_matrix.to(device).view(4, 4).float()
        else:
            M = torch.tensor(camera_matrix, device=device, dtype=torch.float32).view(4, 4)
        R = M[:3, :3]
        t = M[:3, 3]

        coords = splats.xyz @ R.T + t
        z = coords[:, 2]
        in_front = z > 1e-6
        if not in_front.any():
            img = torch.zeros((1, output_height, output_width, 3), device=device)
            mask = torch.zeros((output_height, output_width), device=device)
            return img, mask

        coords = coords[in_front]
        f_dc = splats.f_dc[in_front]
        f_rest = splats.f_rest[in_front]
        opacity = splats.opacity[in_front].squeeze(-1)
        scale = splats.scale[in_front]

        if opacity_is_logit:
            opacity = torch.sigmoid(opacity)

        if max_splats > 0 and coords.shape[0] > max_splats:
            keep = torch.topk(opacity, k=max_splats).indices
            coords = coords[keep]
            f_dc = f_dc[keep]
            f_rest = f_rest[keep]
            opacity = opacity[keep]
            scale = scale[keep]

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
            img = torch.zeros((1, output_height, output_width, 3), device=device)
            mask = torch.zeros((output_height, output_width), device=device)
            return img, mask

        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        colors = colors[valid]
        opacity = opacity[valid]
        scale = scale[valid]

        px = (u * 0.5 + 0.5) * (output_width - 1)
        py = (v * 0.5 + 0.5) * (output_height - 1)

        fov_rad = math.radians(camera_horizontal_fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        fx = f * (output_width - 1) / 2.0
        fy = f * (output_height - 1) / 2.0
        scale_mean = scale.mean(dim=1)
        sigma_x = (scale_mean * fx / depth.clamp(min=1e-6)).clamp(min=0.5, max=512.0)
        sigma_y = (scale_mean * fy / depth.clamp(min=1e-6)).clamp(min=0.5, max=512.0)

        if render_mode == "fast":
            max_radius = max(1, int(max_radius))
            chunk_size = max(1, int(chunk_size))
            total_px = output_height * output_width
            alpha_sum = torch.zeros((total_px,), device=device)
            color_sum = torch.zeros((total_px, 3), device=device)
            depth_sum = torch.zeros((total_px,), device=device)
            n_splats = px.shape[0]
            for start in range(0, n_splats, chunk_size):
                end = min(n_splats, start + chunk_size)
                px_c = px[start:end]
                py_c = py[start:end]
                sx_c = sigma_x[start:end].clamp(min=1e-4)
                sy_c = sigma_y[start:end].clamp(min=1e-4)
                opacity_c = opacity[start:end].clamp(0.0, 1.0)
                colors_c = colors[start:end]
                depth_c = depth[start:end]

                rad_x = torch.ceil(3.0 * sx_c).to(torch.int64).clamp(min=1, max=max_radius)
                rad_y = torch.ceil(3.0 * sy_c).to(torch.int64).clamp(min=1, max=max_radius)
                max_rx = int(rad_x.max().detach().cpu().item())
                max_ry = int(rad_y.max().detach().cpu().item())
                if max_rx <= 0 or max_ry <= 0:
                    continue

                grid_x = torch.arange(-max_rx, max_rx + 1, device=device)
                grid_y = torch.arange(-max_ry, max_ry + 1, device=device)
                x0 = torch.floor(px_c - rad_x.float()).view(-1, 1, 1)
                y0 = torch.floor(py_c - rad_y.float()).view(-1, 1, 1)

                xs = x0 + grid_x.view(1, 1, -1)
                ys = y0 + grid_y.view(1, -1, 1)

                dx = (xs - px_c.view(-1, 1, 1)) / sx_c.view(-1, 1, 1)
                dy = (ys - py_c.view(-1, 1, 1)) / sy_c.view(-1, 1, 1)
                weight = torch.exp(-0.5 * (dx * dx + dy * dy))

                xs_int = xs.to(torch.int64)
                ys_int = ys.to(torch.int64)
                valid = (
                    (xs_int >= 0)
                    & (xs_int < output_width)
                    & (ys_int >= 0)
                    & (ys_int < output_height)
                    & (dx.abs() <= 3.0)
                    & (dy.abs() <= 3.0)
                )

                alpha = opacity_c.view(-1, 1, 1) * weight
                alpha = alpha * valid
                valid_flat = valid.view(-1)
                if not valid_flat.any():
                    continue

                idx = (ys_int * output_width + xs_int).view(-1)[valid_flat]
                alpha_flat = alpha.view(-1)[valid_flat]
                color_flat = (alpha.unsqueeze(-1) * colors_c.view(-1, 1, 1, 3)).view(-1, 3)[valid_flat]
                depth_flat = (alpha * depth_c.view(-1, 1, 1)).view(-1)[valid_flat]

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

        img = torch.zeros((output_height, output_width, 3), device=device)
        alpha_img = torch.zeros((output_height, output_width), device=device)
        depth_acc = torch.zeros((output_height, output_width), device=device)

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

            xs = torch.arange(x0, x1 + 1, device=device)
            ys = torch.arange(y0, y1 + 1, device=device)
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


NODE_CLASS_MAPPINGS = {
    "LoadPlySplat": LoadPlySplat,
    "ImageToSplat": ImageToSplat,
    "FisheyeToGaussian": FisheyeToGaussian,
    "RotateSplats": RotateSplats,
    "MergeSplats": MergeSplats,
    "RenderSplat": RenderSplat,
    "SavePlySplat": SavePlySplat,
}
