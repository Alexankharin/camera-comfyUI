import math
import os
import logging
import hashlib
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


NODE_CLASS_MAPPINGS = {
    "LoadPlySplat": LoadPlySplat,
    "RotateSplats": RotateSplats,
    "RenderSplat": RenderSplat,
}
