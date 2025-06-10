import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, List
import math
import os
import folder_paths
import logging
import hashlib
from kornia.filters import median_blur

from tqdm import tqdm
# Try importing open3d and its visualization modules; log a warning if not found
try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    _open3d_import_error = None
except ImportError as e:
    o3d = None
    gui = None
    rendering = None
    _open3d_import_error = e
    logging.warning("[camera-comfyUI] open3d is not installed. Point cloud visualization features will be unavailable. Error: %s", e)

class Projection:
    """
    A class to define supported projection types.
    """
    PROJECTIONS = ["PINHOLE", "FISHEYE", "EQUIRECTANGULAR"]

# ==== Depth to pointcloud conversion functions ==== #
def pinhole_depth_to_XYZ(depth: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert depth map to XYZ coordinates using pinhole projection.
    """
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

def fisheye_depth_to_XYZ(depth: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert depth map to XYZ coordinates using fisheye projection.
    """
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

def equirect_depth_to_XYZ(depth: torch.Tensor, *_: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert depth map to XYZ coordinates using equirectangular projection.
    """
    # full 360°×180° equirectangular
    H, W = depth.shape
    lon = torch.linspace(-math.pi, math.pi, W, device=depth.device).unsqueeze(0).expand(H, W)
    lat = torch.linspace( math.pi/2, -math.pi/2, H, device=depth.device).unsqueeze(1).expand(H, W)
    X = depth * torch.cos(lat) * torch.cos(lon)
    Y = depth * torch.sin(lat)
    Z = depth * torch.cos(lat) * torch.sin(lon)
    return X, Y, Z


# ==== XYZ→Normalized UV + depth ====

def XYZ_to_pinhole(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert XYZ coordinates to normalized UV and depth using pinhole projection.
    """
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2)
    depth = torch.sqrt(X**2 + Y**2 + Z**2)
    phi   = torch.atan2(Y, X)
    theta = torch.acos(Z / depth)
    r     = f * torch.tan(theta)
    u     = r * torch.cos(phi)
    v     = r * torch.sin(phi)
    return u, v, depth

def XYZ_to_fisheye(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert XYZ coordinates to normalized UV and depth using fisheye projection.
    """
    # equidistant fisheye: u = (θ/(fov/2))·cosφ, etc.
    fov_rad = math.radians(fov)
    depth   = torch.sqrt(X**2 + Y**2 + Z**2)
    theta   = torch.acos(Z / depth)
    phi     = torch.atan2(Y, X)
    r       = theta / (fov_rad / 2)
    u       = r * torch.cos(phi)
    v       = r * torch.sin(phi)
    return u, v, depth

def XYZ_to_equirect(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, fov: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert XYZ coordinates to normalized UV and depth using equirectangular projection.
    """
    # full 360°×180°  
    fov_rad = math.radians(fov) / 2
    depth = torch.sqrt(X**2 + Y**2 + Z**2)
    lon   = torch.atan2(X, Z)            # –π → +π
    lat   = torch.asin(Y / depth)        # –π/2 → +π/2
    u     = lon / fov_rad                # –1 → +1 across width
    v     = lat / (math.pi / 2)          # –1 → +1 down height
    return u, v, depth

def project_first_hit(volume_sparse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project the first hit in a sparse volume to an RGBA image and mask.
    """
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
    Convert an (optional) depth map and RGB(A) image into a single pointcloud
    tensor of shape (N,7) [X,Y,Z,R,G,B,A], optionally masking out points.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image":               ("IMAGE",),
                "input_projection":    (Projection.PROJECTIONS, {"tooltip": "projection type of depth map"}),
                "input_horizontal_fov":("FLOAT", {
                    "default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Horizontal field of view in degrees"}),
                "depth_scale":         ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "Scale factor for depth values"}),
                "invert_depth":        ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the depth map values"}),
            },
            "optional": {
                "depthmap": ("TENSOR",),
                "mask":     ("MASK",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "depth_to_pointcloud"
    CATEGORY = "Camera/PointCloud"

    def depth_to_pointcloud(
        self,
        image: torch.Tensor,
        input_projection: str,
        input_horizontal_fov: float,
        depth_scale: float,
        invert_depth: bool,
        depthmap: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        # --- Prepare image (C,H,W) ---
        img = image
        if img.dim() == 4:                      # [1,H,W,C] or [B,H,W,C]
            img = img.squeeze(0)
        if img.dim() == 3 and img.shape[-1] in (3,4):  # [H,W,C]
            img = img.permute(2,0,1)
        C, H, W = img.shape

        # --- Prepare depth (H,W) ---
        if depthmap is None:
            depth = torch.ones((H, W), device=img.device)
        else:
            d = depthmap
            if d.dim() == 4:                  # [1,H,W,1] or [B,H,W,1]
                d = d.squeeze(0).squeeze(-1)
            elif d.dim() == 3 and d.shape[-1] == 1:  # [H,W,1]
                d = d.squeeze(-1)
            elif d.dim() == 3:               # [H,W,C]
                d = d.mean(dim=-1)
            # now d is [H_d, W_d]
            H_d, W_d = d.shape
            if (H_d, W_d) != (H, W):
                d = F.interpolate(
                    d.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            depth = d

        # invert & scale
        if invert_depth:
            depth = 1.0 / depth.clamp(min=1e-6)
        depth = depth * depth_scale

        # --- Depth → XYZ ---
        if input_projection == "PINHOLE":
            X, Y, Z = pinhole_depth_to_XYZ(depth, input_horizontal_fov)
        elif input_projection == "FISHEYE":
            X, Y, Z = fisheye_depth_to_XYZ(depth, input_horizontal_fov)
        else:
            X, Y, Z = equirect_depth_to_XYZ(depth, input_horizontal_fov)
        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

        # --- Extract colors RGBA → (H,W,4) then flatten ---
        rgba = img.permute(1,2,0).float()  # [H,W,C]
        if C == 3:
            alpha = torch.ones((H, W, 1), device=rgba.device)
            rgba = torch.cat([rgba, alpha], dim=2)
        colors = rgba.reshape(-1, 4)

        # --- Apply mask if provided ---
        if mask is not None:
            m = mask
            # collapse any batch/channel dims
            if m.dim() == 4:
                m = m.squeeze(0).mean(dim=-1)
            elif m.dim() == 3:
                m = m.mean(dim=-1)
            # resize to [H,W]
            if m.shape[-2:] != (H, W):
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            m_bool = m > 0.5
            keep = m_bool.reshape(-1)
            coords = coords[keep]
            colors = colors[keep]

        # --- Build pointcloud [N,7] ---
        pointcloud = torch.cat([coords, colors], dim=1)
        return (pointcloud,)

class TransformPointCloud:
    """
    Apply a 4×4 transform to a point cloud tensor (N,7) -> (N,7).
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for the node.
        """
        return {
            "required": {
                "pointcloud": ("TENSOR",),
                "transform_matrix": ("MAT_4X4",),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("transformed pointcloud",)
    FUNCTION = "transform_pointcloud"
    CATEGORY = "Camera/PointCloud"
    
    def transform_pointcloud(
        self,
        pointcloud: torch.Tensor,
        transform_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Apply a transformation matrix to a point cloud.
        """
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
    Projects a point cloud (N,7) into an image, mask, and depth tensor using GPU-efficient
    z-buffering. Supports splatting dilation (point_size > 1) via average pooling and mask normalization,
    filling gaps with preference given to back-facing points.

    Returns:
      - image: [1,H,W,3]
      - mask: [H,W]
      - depth: [1,H,W,1]
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud":            ("TENSOR",),
                "output_projection":     (Projection.PROJECTIONS, {}),
                "output_horizontal_fov": ("FLOAT", {"default": 90.0}),
                "output_width":          ("INT",   {"default": 512, "min": 1, "max": 16384}),
                "output_height":         ("INT",   {"default": 512, "min": 1, "max": 16384}),
                "point_size":            ("INT",   {"default": 1, "min": 1}),
                "return_inverse_depth":  ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("image", "mask", "depth")
    FUNCTION = "project_pointcloud"
    CATEGORY = "Camera/PointCloud"

    def project_pointcloud(
        self,
        pointcloud:    torch.Tensor,
        output_projection: str,
        output_horizontal_fov: float,
        output_width:  int,
        output_height: int,
        point_size:    int = 1,
        return_inverse_depth: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = pointcloud.device
        coords = pointcloud[:, :3]
        colors = pointcloud[:, 3:].float()

        # 1) Filter points in front of the camera
        mask_front = coords[:, 2] > 0
        coords = coords[mask_front]
        colors = colors[mask_front]

        # 2) Project to normalized UV + depth
        X, Y, Z = coords.unbind(1)
        if output_projection == "PINHOLE":
            u, v, depth = XYZ_to_pinhole(X, Y, Z, output_horizontal_fov)
        elif output_projection == "FISHEYE":
            u, v, depth = XYZ_to_fisheye(X, Y, Z, output_horizontal_fov)
        else:
            u, v, depth = XYZ_to_equirect(X, Y, Z, output_horizontal_fov)

        # 3) Rasterize to pixel indices
        px = (u * (output_width - 1) / 2) + (output_width - 1) / 2
        py = (v * (output_height - 1) / 2) + (output_height - 1) / 2
        ix = px.round().clamp(0, output_width - 1).long()
        iy = py.round().clamp(0, output_height - 1).long()
        pix = iy * output_width + ix
        M   = output_width * output_height

        # —— NEW: drop any invalid / NaN→int_min projections ——
        valid = (pix >= 0) & (pix < M)
        depth  = depth[valid]
        colors = colors[valid]
        pix    = pix[valid]
        order  = torch.arange(depth.size(0), device=device)
        # rebuild your "order" to match

        # 4) Allocate or reuse buffers
        if not hasattr(self, '_z_front') or self._z_front.numel() != M:
            self._z_front = torch.empty((M,), device=device)
            self._z_back  = torch.empty((M,), device=device)
            self._idx     = torch.full((M,), -1, dtype=torch.long, device=device)
            self._flat    = torch.zeros((M, 4), device=device)
        z_front = self._z_front
        z_back  = self._z_back
        idxbuf  = self._idx
        flat    = self._flat

        # 5) Front z-buffer pass (nearest)
        z_front.fill_(float('inf'))
        z_front.scatter_reduce_(0, pix, depth, reduce='amin', include_self=True)
        sel_front = depth == z_front[pix]
        order = torch.arange(depth.size(0), device=device)
        order_m = torch.where(sel_front, order, depth.size(0))
        idxbuf.fill_(depth.size(0))
        idxbuf.scatter_reduce_(0, pix, order_m, reduce='amin', include_self=True)
        win_front = order == idxbuf[pix]

        flat.fill_(0)
        flat[pix[win_front]] = colors[win_front]
        img4 = flat.view(output_height, output_width, 4)
        rgb   = img4[..., :3].clamp(0, 255)
        alpha = (img4[..., 3] > 0).float()
        rgb  *= alpha.unsqueeze(-1)
        depth_img = z_front.view(output_height, output_width)
        rgb_HR = rgb

        # 6) Back z-buffer pass (farthest) for hole-filling
        if point_size > 1:
            z_back.fill_(-float('inf'))
            z_back.scatter_reduce_(0, pix, depth, reduce='amax', include_self=True)
            sel_back = depth == z_back[pix]
            order_m = torch.where(sel_back, order, -1)
            idxbuf.fill_(-1)
            idxbuf.scatter_reduce_(0, pix, order_m, reduce='amax', include_self=True)
            win_back = idxbuf[pix] >= 0

            flat.fill_(0)
            flat[pix[win_back]] = colors[win_back]
            back4 = flat.view(output_height, output_width, 4)
            rgb_back   = back4[..., :3].clamp(0,255)
            alpha_back = (back4[..., 3] > 0).float()

            # fill holes where front missed
            hole = (alpha == 0) & (alpha_back > 0)
            rgb[hole]       = rgb_back[hole]
            alpha[hole]     = 1.0
            depth_img[hole] = z_back.view(output_height, output_width)[hole]

            # 7) Median-filter _only_ in hole regions
            if hole.any():
                # prepare for kornia median_blur: [B,C,H,W]
                rgb_t   = rgb.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
                # apply median filter
                rgb_med = median_blur(rgb_t, (point_size, point_size))
                # back to HWC
                rgb_med = rgb_med.squeeze(0).permute(1,2,0)
                # merge only at hole locations
                rgb[hole] = rgb_med[hole]
                # alpha already set to 1.0 for holes

        # 8) Pack and return with original script shapes
        img       = rgb.unsqueeze(0)  # [1,H,W,3]
        mask_out  = alpha             # [H,W]
        depth4    = depth_img.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
        if return_inverse_depth:
            depth4 = 1.0 / depth4.clamp(min=1e-6)
        depth4 = depth4 * mask_out.unsqueeze(0).unsqueeze(-1)
        return img, mask_out, depth4

class PointCloudUnion:
    """
    Combine two point clouds into one.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud1": ("TENSOR",),
                "pointcloud2": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES =("merged pointcloud",) 
    FUNCTION = "union_pointclouds"
    CATEGORY = "Camera/PointCloud"

    def union_pointclouds(
        self,
        pointcloud1: torch.Tensor,
        pointcloud2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Combine two point clouds into one.
        """
        return (torch.cat([pointcloud1, pointcloud2], dim=0),)


class LoadPointCloud:
    """
    Load a PLY or NumPy .npy point‐cloud file from your ComfyUI input directory into a (N,7) tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and (f.lower().endswith(".ply") or f.lower().endswith(".npy"))
        ]
        return {
            "required": {
                "pointcloud_file": (
                    sorted(files),
                    {
                        "file_chooser": True,
                        "tooltip": "Select a .ply or .npy point‐cloud file to load from your input folder."
                    }
                ),
            }
        }

    CATEGORY = "Camera/PointCloud"
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("loaded pointcloud",)
    FUNCTION = "load_pointcloud"
    DESCRIPTION = "Loads a .ply or .npy point‐cloud file into a tensor of shape (N,7)."

    def load_pointcloud(self, pointcloud_file: str):
        file_path = folder_paths.get_annotated_filepath(pointcloud_file)
        if pointcloud_file.lower().endswith(".npy"):
            arr = np.load(file_path)
            tensor_pc = torch.from_numpy(arr)
            return (tensor_pc,)

        if o3d is None:
            logging.warning("[camera-comfyUI] open3d is not installed. Falling back to manual PLY parser.")
            coords = []
            colors = []
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                while not line.startswith("end_header"):
                    line = f.readline().strip()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    x, y, z = map(float, parts[0:3])
                    r, g, b, a = map(float, parts[3:7])
                    coords.append((x, y, z))
                    colors.append((r, g, b, a))
            np_coords = np.array(coords, dtype=np.float32)
            np_colors = np.array(colors, dtype=np.float32)
            # if colors are > 1, normalize them to [0,1]
            if np_colors.max() > 1.0:
                np_colors = np_colors / 255.0
        else:
            pc = o3d.t.io.read_point_cloud(file_path)
            np_coords = pc.point["positions"].numpy().astype(np.float32)
            if "colors" in pc.point:
                cols = pc.point["colors"].numpy().astype(np.float32)
            else:
                cols = np.ones((np_coords.shape[0], 3), dtype=np.float32)
            if "alpha" in pc.point:
                alpha = pc.point["alpha"].numpy().astype(np.float32)
            else:
                alpha = np.ones((np_coords.shape[0], 1), dtype=np.float32)
            np_colors = np.concatenate([cols, alpha], axis=1)
            if np_colors.max() > 1.0:
                np_colors = np_colors / 255.0
        # combine coords and colors into a single tensor
        combined = np.concatenate([np_coords, np_colors], axis=1)
        tensor_pc = torch.from_numpy(combined)
        return (tensor_pc,)

    @classmethod
    def IS_CHANGED(cls, pointcloud_file: str):
        path = folder_paths.get_annotated_filepath(pointcloud_file)
        m = hashlib.sha256()
        with open(path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, pointcloud_file: str):
        if not folder_paths.exists_annotated_filepath(pointcloud_file):
            return f"Invalid point‐cloud file: {pointcloud_file}"
        return True

class SavePointCloud:
    """
    Save a point cloud tensor (N,7) to a file in PLY format or as a NumPy .npy array,
    resolving into your ComfyUI output directory with a filename prefix.
    """
    def __init__(self):
        # exactly like SaveImage
        self.output_dir    = folder_paths.get_output_directory()
        self.type          = "pointcloud"
        self.prefix_append = ""   # you can set e.g. a suffix in the UI if you like

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud":     ("TENSOR",),
                "filename_prefix":(
                    "STRING",
                    {
                        "default": "ComfyUIPointCloud",
                        "tooltip": "Prefix for the .ply/.npy file. You can include format-tokens like %date:yyyy-MM-dd%."
                    }
                ),
                "save_as": (["ply", "npy"], {"default": "ply", "tooltip": "Choose file format to save: PLY or NumPy .npy"}),
            },
            "hidden": {}
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_pointcloud"
    OUTPUT_NODE  = True
    CATEGORY     = "Camera/PointCloud"
    DESCRIPTION  = "Saves the input point cloud to your ComfyUI output directory as .ply or .npy."

    def save_pointcloud(self, pointcloud: torch.Tensor, filename_prefix: str, save_as: str = "ply"):
        # apply any suffix from the node UI
        filename_prefix += self.prefix_append

        # ---- exactly the same pattern as SaveImage uses ----
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                0, 0
            )
        os.makedirs(full_output_folder, exist_ok=True)
        base_name = filename.replace("%batch_num%", "0")
        if save_as == "ply":
            ply_name = f"{base_name}_{counter:05}.ply"
            ply_path = os.path.join(full_output_folder, ply_name)
            coords = pointcloud[:, :3].cpu().numpy().astype(np.float32)
            colors = pointcloud[:, 3:].cpu().numpy().clip(0, 1).astype(np.float32)

            if o3d is None:
                logging.warning("[camera-comfyUI] open3d is not installed. Falling back to manual ASCII PLY writer.")
                with open(ply_path, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {coords.shape[0]}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property float red\n")
                    f.write("property float green\n")
                    f.write("property float blue\n")
                    f.write("property float alpha\n")
                    f.write("end_header\n")
                    for (x, y, z), (r, g, b, a) in zip(coords, colors):
                        f.write(f"{x} {y} {z} {r} {g} {b} {a}\n")
            else:
                pc = o3d.t.geometry.PointCloud()
                pc.point["positions"] = o3d.core.Tensor(coords, o3d.core.float32)
                pc.point["colors"] = o3d.core.Tensor(colors[:, :3], o3d.core.float32)
                if colors.shape[1] > 3:
                    pc.point["alpha"] = o3d.core.Tensor(colors[:, 3:], o3d.core.float32)
                else:
                    pc.point["alpha"] = o3d.core.Tensor(np.ones((coords.shape[0], 1), dtype=np.float32), o3d.core.float32)
                o3d.t.io.write_point_cloud(ply_path, pc)
            file_name = ply_name
        else:
            npy_name = f"{base_name}_{counter:05}.npy"
            npy_path = os.path.join(full_output_folder, npy_name)
            np.save(npy_path, pointcloud.cpu().numpy())
            file_name = npy_name
        counter += 1
        return {
            "ui": {
                "pointclouds": [{
                    "filename": file_name,
                    "subfolder": subfolder,
                    "type":      self.type
                }]
            }
        }

class CameraMotionNode:
    """
    Renders a pointcloud sequence by interpolating along a trajectory.
    Accepts:
      • trajectory: (K,4,4)
      • n_points:   INT frames per segment
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pointcloud":          ("TENSOR",),
            "trajectory":          ("TENSOR",),   # (K,4,4)
            "n_points":            ("INT",    {"default":10, "min":2, "step":1}),
            "output_projection":   (Projection.PROJECTIONS, {}),
            "output_horizontal_fov": ("FLOAT", {"default":90.0}),
            "output_width":        ("INT",    {"default":512, "min":8, "max":16384}),
            "output_height":       ("INT",    {"default":512, "min":8, "max":16384}),
            "point_size":          ("INT",    {"default":1,   "min":1}),
            "widen_mask":         ("INT",    {"default":0, "min":0, "max":64}),
            "invert_mask":        ("BOOLEAN", {"default": False}),
            "points_to_mask":     ("BOOLEAN", {"default": False, "tooltip": "Output mask frames of projected points"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("motion_frames", "mask_frames")
    FUNCTION      = "generate_motion_frames"
    CATEGORY      = "Camera/Trajectory"

    def generate_motion_frames(
        self,
        pointcloud:            torch.Tensor,
        trajectory:            torch.Tensor,
        n_points:              int,
        output_projection:     str,
        output_horizontal_fov: float,
        output_width:          int,
        output_height:         int,
        point_size:            int = 1,
        widen_mask:           int = 0,
        invert_mask:          bool = False,
        points_to_mask:       bool = False
    ) -> Tuple[torch.Tensor]:
        # validate trajectory shape
        if trajectory.dim() != 3 or trajectory.shape[1:] != (4,4):
            raise ValueError(f"trajectory must be (K,4,4), got {trajectory.shape}")
        K = trajectory.shape[0]
        if K < 2:
            raise ValueError("trajectory must contain at least two poses")

        # build full list of interpolated extrinsics
        all_mats = []
        for i in range(K-1):
            A = trajectory[i]
            B = trajectory[i+1]
            # uniform interpolation for this segment
            ts = np.linspace(0.0, 1.0, n_points, endpoint=False)
            for t in ts:
                all_mats.append(A * (1.0 - t) + B * t)
        # finally append the very last pose
        all_mats.append(trajectory[-1])

        full_traj = torch.stack(all_mats, dim=0)  # (T,4,4)

        # render each pose
        proj_node      = ProjectPointCloud()
        transform_node = TransformPointCloud()
        frames = []
        masks  = []
        for M in tqdm(full_traj):
            pc_t, = transform_node.transform_pointcloud(pointcloud, M)
            img, mask, _ = proj_node.project_pointcloud(
                pc_t,
                output_projection,
                output_horizontal_fov,
                output_width,
                output_height,
                point_size
            )
            if widen_mask > 0:
                k = 2 * widen_mask + 1
                pad = widen_mask
                mask = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0).squeeze(0)
            if invert_mask:
                mask = 1.0 - mask
            masks.append(mask)
            if points_to_mask:
                img = mask.unsqueeze(-1).repeat(1,1,1,3)
            frames.append(img[0])

        # output as (T,H,W,3)
        return (torch.stack(frames, dim=0), torch.stack(masks, dim=0))

class CameraInterpolationNode:
    """
    Wrap two 4×4 poses into a trajectory tensor. 
    Outputs only `trajectory` (shape 2×4×4).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_matrix": ("MAT_4X4",),
                "final_matrix":   ("MAT_4X4",),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "interpolate"
    CATEGORY = "Camera/Trajectory"

    def interpolate(
        self,
        initial_matrix: torch.Tensor,
        final_matrix:   torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # stack into a (2,4,4) trajectory
        # convert to tensor if needed
        if isinstance(initial_matrix, np.ndarray):
            initial_matrix = torch.from_numpy(initial_matrix).float()
        if isinstance(final_matrix, np.ndarray):
            final_matrix = torch.from_numpy(final_matrix).float()
        traj = torch.stack([initial_matrix, final_matrix], dim=0)
        return (traj,)


class CameraTrajectoryNode:
    """
    Interactive tool to walk inside a pointcloud and select camera poses.
    Outputs a trajectory tensor (K×4×4).
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {"required": {
            "pointcloud":    ("TENSOR",),
        },
        "optional": {
            "initial_matrix": ("MAT_4X4"),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "build_trajectory"
    CATEGORY = "Camera/Trajectory"

    def build_trajectory(
        self,
        pointcloud:     torch.Tensor,
        initial_matrix: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        # Default camera extrinsic = identity (origin, looking +Z)
        if initial_matrix is None:
            initial_matrix = torch.eye(4, device=pointcloud.device).float()
        if isinstance(initial_matrix, np.ndarray):
            initial_matrix = torch.from_numpy(initial_matrix).float()
        traj_list = launch_trajectory_editor(pointcloud, initial_matrix)
        traj = torch.stack([torch.from_numpy(m).float() for m in traj_list], dim=0)
        return (traj,)


def launch_trajectory_editor(
    pointcloud: torch.Tensor,
    initial_matrix: torch.Tensor,
    interp_steps: int = 10
) -> List[np.ndarray]:
    """
    Launches an Open3D VisualizerWithKeyCallback window to navigate the pointcloud.
    Returns interpolated list of extrinsic matrices.
    """
    # Prepare pointcloud
    pts = pointcloud[:, :3].cpu().numpy()  # [m] to [cm]
    # clip each coordinate to 90 percentile of the whole pointcloud
    # (this is a bit arbitrary, but it works well for most pointclouds)
    clip = np.percentile(pts, 90, axis=0)
    pts = np.clip(pts, -clip, clip)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if pointcloud.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:,3:6].cpu().numpy())

    # Compute centroid
    centroid = pcd.get_center()

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            "Trajectory Editor | WSAD: pan | Z/X: zoom | P: record pose | Q: quit"
        ),
        width=1024, height=768
    )
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()

    # 1) Center camera at the point-cloud centroid
    ctr.set_lookat(centroid)
    # 2) Look straight along +Z
    ctr.set_front((0.0, 0.0, 1.0))
    # 3) Choose a conventional 'up' vector
    ctr.set_up((0.0, -1.0, 0.0))
    # 4) Zoom very close (smaller = closer)
    ctr.set_zoom(0.2)

    # If you still want to apply a user-provided initial_matrix, you can
    # uncomment these two lines—but they’ll override the “look‐down‐Z”
    # params above.
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = initial_matrix.cpu().numpy().copy(); ctr.convert_from_pinhole_camera_parameters(params)

    # controls: slower pan and zoom
    pan_step  = 0.1   # meters per keypress
    zoom_step = 0.95   # factor <1 = zoom in, >1 = zoom out

    traj_list: List[np.ndarray] = []

    # Define callbacks
    def pan_forward(vis):  vis.get_view_control().translate(0,  pan_step);      return False
    def pan_backward(vis): vis.get_view_control().translate(0, -pan_step);      return False
    def pan_left(vis):     vis.get_view_control().translate( pan_step, 0);      return False
    def pan_right(vis):    vis.get_view_control().translate(-pan_step, 0);      return False
    def zoom_in(vis):      vis.get_view_control().scale(zoom_step);             return False
    def zoom_out(vis):     vis.get_view_control().scale(1/zoom_step);           return False

    def record_pose(vis):
        extr = vis.get_view_control() \
                  .convert_to_pinhole_camera_parameters() \
                  .extrinsic.copy()
        traj_list.append(extr)
        print(f"[Trajectory Editor] Recorded pose {len(traj_list)}")
        return False

    def close(vis):
        vis.destroy_window()
        return True

    # Register keys
    vis.register_key_callback(ord('W'), pan_forward)
    vis.register_key_callback(ord('S'), pan_backward)
    vis.register_key_callback(ord('A'), pan_left)
    vis.register_key_callback(ord('D'), pan_right)
    vis.register_key_callback(ord('Z'), zoom_in)
    vis.register_key_callback(ord('X'), zoom_out)
    vis.register_key_callback(ord('P'), record_pose)
    vis.register_key_callback(ord('Q'), close)

    # Run the window
    vis.run()

    # Ensure at least two poses
    if not traj_list:
        raise ValueError("No waypoints recorded. Please record at least one with 'P'.")
    if len(traj_list) == 1:
        traj_list.append(traj_list[0].copy())

    # Interpolate between each pair
    interp: List[np.ndarray] = []
    for A, B in zip(traj_list[:-1], traj_list[1:]):
        for t in np.linspace(0.0, 1.0, interp_steps, endpoint=False):
            interp.append((1 - t) * A + t * B)
    interp.append(traj_list[-1])

    return interp


class PointCloudCleaner:
    """
    Projects a pointcloud through an identity matrix into fisheye-180° normalized UV and depth,
    then inverts and normalizes depth and performs voxel-based cleaning in (px,py,inv_depth) space.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pointcloud":             ("TENSOR",),
                "width":                  ("INT",   {"default":1024, "min":1, "max":16384}),
                "height":                 ("INT",   {"default":1024, "min":1, "max":16384}),
                "voxel_size":             ("FLOAT", {"default":1.0,   "min":1e-3}),
                "min_points_per_voxel":   ("INT",   {"default":3,     "min":1}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("cleaned_pointcloud",)
    FUNCTION = "clean_pointcloud"
    CATEGORY = "Camera/PointCloud"

    def clean_pointcloud(
        self,
        pointcloud: torch.Tensor,
        width: int,
        height: int,
        voxel_size: float,
        min_points_per_voxel: int
    ) -> Tuple[torch.Tensor]:
        device = pointcloud.device
        N = pointcloud.shape[0]

        # 1) Project through identity (camera at origin) --> just pts in camera space
        coords = pointcloud[:, :3]

        # 2) Fisheye-180 projection to normalized uv and raw depth
        X, Y, Z = coords.unbind(1)
        u, v, depth = XYZ_to_fisheye(X, Y, Z, 180.0)

        # 3) Convert normalized uv -> pixel coordinates
        px = (u * (width  - 1) / 2.0) + ((width  - 1) / 2.0)
        py = (v * (height - 1) / 2.0) + ((height - 1) / 2.0)

        # 4) Invert depth and normalize so that max(inv_depth) == width/2
        inv_depth = 1.0 / (depth + 1e-6)
        # make 95 percentile range. Sample down to 1e6 points since quantile is limited
        inv = inv_depth.view(-1)
        N   = inv.numel()
        # sample up to 1e6 points
        k   = min(N, 1000000)
        idx = torch.randperm(N, device=inv.device)[:k]
        sample = inv[idx]
        q95    = torch.quantile(sample, 0.95)
        inv_norm = inv_depth / q95 * (width/8)

        # 5) Build voxel indices in [px,py,inv_norm] space
        uvz = torch.stack([px, py, inv_norm], dim=1)
        voxel_idx = torch.floor(uvz / voxel_size).to(torch.int64)

        # 6) Unique voxels & counts
        _, inverse_idx, counts = torch.unique(
            voxel_idx, return_inverse=True, return_counts=True, dim=0
        )

        # 7) Filter points in sparse voxels
        keep_mask = counts[inverse_idx] >= min_points_per_voxel
        cleaned = pointcloud[keep_mask]
        return (cleaned,)

class ProjectAndClean:
    """
    Projects a point cloud through a 4×4 matrix, records per-pixel point indices,
    applies morphological cleaning on the mask (defining occluded pixels to remove),
    and returns the point cloud with those occlusions retained.
    Buffers for depth and index are reused to reduce peak memory.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        odd_kernel_sizes = [i for i in range(1, 100, 2)]
        return {
            "required": {
                "pointcloud":      ("TENSOR",),
                "matrix":          ("MAT_4X4",),
                "projection":      (Projection.PROJECTIONS, {}),
                "fov":             ("FLOAT", {"default": 90.0}),
                "width":           ("INT",   {"default": 512, "min": 1, "max": 16384}),
                "height":          ("INT",   {"default": 512, "min": 1, "max": 16384}),
                "mode":            (["erode", "open"], {"default": "erode"}),
                "num_iterations":  ("INT",   {"default": 1, "min": 1, "max": 10}),
                "kernel_size":     ("INT",   {"default": 3, "min": 1, "max": 99, "step": 2}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("cleaned_pointcloud",)
    FUNCTION = "project_and_clean"
    CATEGORY = "Camera/PointCloud"

    def project_and_clean(
        self,
        pointcloud: torch.Tensor,
        matrix: torch.Tensor,
        projection: str,
        fov: float,
        width: int,
        height: int,
        mode: str,
        num_iterations: int,
        kernel_size: int,
    ) -> Tuple[torch.Tensor]:
        device = pointcloud.device

        # Prepare transformation matrix on-device
        if isinstance(matrix, torch.Tensor):
            M = matrix.to(device).view(4, 4)
        else:
            M = torch.tensor(matrix, device=device, dtype=torch.float32).view(4, 4)

        total_px = width * height
        # Preallocate buffers
        depth_buf = torch.empty((total_px,), device=device)
        idx_buf   = torch.full((total_px,), -1, device=device, dtype=torch.long)
        pad = kernel_size // 2

        for _ in range(num_iterations):
            N = pointcloud.shape[0]
            # 1) Transform points
            coords = pointcloud[:, :3]
            homo = torch.cat([coords, torch.ones((N, 1), device=device)], dim=1)
            coords_t = (M @ homo.T).T[:, :3]

            # 2) Project
            X, Y, Z = coords_t.unbind(1)
            if projection == "PINHOLE":
                u, v, depth = XYZ_to_pinhole(X, Y, Z, fov)
            elif projection == "FISHEYE":
                u, v, depth = XYZ_to_fisheye(X, Y, Z, fov)
            else:
                u, v, depth = XYZ_to_equirect(X, Y, Z, fov)

            # 3) Rasterize
            px = (u * (width - 1) / 2) + (width - 1) / 2
            py = (v * (height - 1) / 2) + (height - 1) / 2
            ix = px.round().clamp(0, width - 1).long()
            iy = py.round().clamp(0, height - 1).long()
            pix_idx = iy * width + ix  # [N]

            # 4) Z-buffer first hit
            depth_buf.fill_(float('inf'))
            depth_buf.scatter_reduce_(0, pix_idx, depth, reduce='amin', include_self=True)
            is_first = depth == depth_buf[pix_idx]

            # 5) Build index buffer of first hits correctly
            idx_buf.fill_(-1)
            indices = torch.arange(N, device=device)
            first_pixels = pix_idx[is_first]
            first_pts   = indices[is_first]
            idx_buf[first_pixels] = first_pts

            # 6) Mask and morphological clean
            mask_flat = idx_buf >= 0
            mask = mask_flat.view(height, width).float()
            inv = (1.0 - mask).unsqueeze(0).unsqueeze(0)
            if mode == 'erode':
                inv_e = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=pad)
                clean_mask = (1.0 - inv_e).squeeze() > 0.5
            else:
                inv_e = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=pad)
                eroded = (1.0 - inv_e)
                dil = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad)
                clean_mask = dil.squeeze() > 0.5

            # 7) Remove points in cleaned-away pixels
            clean_flat = clean_mask.view(-1)
            removal = (~clean_flat) & (idx_buf >= 0)
            remove_idxs = torch.unique(idx_buf[removal])
            keep = torch.ones(N, dtype=torch.bool, device=device)
            keep[remove_idxs] = False
            pointcloud = pointcloud[keep]

        return (pointcloud,)

class SaveTrajectory:
    """
    Save a trajectory tensor (N,4,4) to a .npy file in the ComfyUI output directory.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "trajectory"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "trajectory": ("TENSOR",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUITrajectory",
                        "tooltip": "Prefix for the .npy file. You can include format-tokens like %date:yyyy-MM-dd%."
                    }
                ),
            },
            "hidden": {}
        }

    RETURN_TYPES = ()
    FUNCTION = "save_trajectory"
    OUTPUT_NODE = True
    CATEGORY = "Camera/Trajectory"
    DESCRIPTION = "Saves the input trajectory tensor (N,4,4) to your ComfyUI output directory as .npy."

    def save_trajectory(self, trajectory: torch.Tensor, filename_prefix: str):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                0, 0
            )
        os.makedirs(full_output_folder, exist_ok=True)
        base_name = filename.replace("%batch_num%", "0")
        npy_name = f"{base_name}_{counter:05}.npy"
        npy_path = os.path.join(full_output_folder, npy_name)
        np.save(npy_path, trajectory.cpu().numpy())
        counter += 1
        return {
            "ui": {
                "trajectories": [{
                    "filename": npy_name,
                    "subfolder": subfolder,
                    "type": self.type
                }]
            }
        }

class LoadTrajectory:
    """
    Load a trajectory tensor (N,4,4) from a .npy file in the ComfyUI input directory.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".npy")
        ]
        return {
            "required": {
                "trajectory_file": (
                    sorted(files),
                    {
                        "file_chooser": True,
                        "tooltip": "Select a .npy trajectory file to load from your input folder."
                    }
                ),
            }
        }

    CATEGORY = "Camera/Trajectory"
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("loaded_trajectory",)
    FUNCTION = "load_trajectory"
    DESCRIPTION = "Loads a .npy trajectory file into a tensor of shape (N,4,4)."

    def load_trajectory(self, trajectory_file: str):
        file_path = folder_paths.get_annotated_filepath(trajectory_file)
        arr = np.load(file_path)
        tensor_traj = torch.from_numpy(arr).float()
        return (tensor_traj,)

    @classmethod
    def IS_CHANGED(cls, trajectory_file: str):
        path = folder_paths.get_annotated_filepath(trajectory_file)
        m = hashlib.sha256()
        with open(path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, trajectory_file: str):
        if not folder_paths.exists_annotated_filepath(trajectory_file):
            return f"Invalid trajectory file: {trajectory_file}"
        return True

NODE_CLASS_MAPPINGS = {
    "DepthToPointCloud": DepthToPointCloud,
    "TransformPointCloud": TransformPointCloud,
    "ProjectPointCloud": ProjectPointCloud,
    "ProjectAndClean": ProjectAndClean,
    "PointCloudUnion": PointCloudUnion,
    "LoadPointCloud": LoadPointCloud,
    "SavePointCloud": SavePointCloud,
    "CameraTrajectoryNode": CameraTrajectoryNode,
    "CameraMotionNode": CameraMotionNode,
    "CameraInterpolationNode": CameraInterpolationNode,
    "PointCloudCleaner": PointCloudCleaner,
    "SaveTrajectory": SaveTrajectory,
    "LoadTrajectory": LoadTrajectory,
}