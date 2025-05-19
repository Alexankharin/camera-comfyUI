# Organize imports
from PIL import Image
import numpy as np
import torch
import time
from typing import Dict, Tuple, Any

class Projection:
    """
    A class to define supported projection types.
    """
    PROJECTIONS = ["PINHOLE", "FISHEYE", "EQUIRECTANGULAR"]

def map_grid(
    grid_torch: torch.Tensor, 
    input_projection: str, 
    output_projection: str, 
    input_horizontal_fov: float, 
    output_horizontal_fov: float, 
    rotation_matrix: torch.Tensor = None
) -> np.ndarray:
    """
    Maps a 2D grid from one projection type to another.

    Args:
        grid_torch (torch.Tensor): A 2D array of shape (height, width, 2) with x and y coordinates normalized to [-1, 1].
        input_projection (str): The input projection type ("PINHOLE", "FISHEYE", "EQUIRECTANGULAR").
        output_projection (str): The output projection type ("PINHOLE", "FISHEYE", "EQUIRECTANGULAR").
        input_horizontal_fov (float): Horizontal field of view for the input projection in degrees.
        output_horizontal_fov (float): Horizontal field of view for the output projection in degrees.
        rotation_matrix (torch.Tensor, optional): A 4x4 rotation matrix. Defaults to identity matrix if not provided.

    Returns:
        np.ndarray: A 2D array of shape (height, width, 2) with the mapped x and y coordinates.
    """
    with torch.no_grad():
        if rotation_matrix is None:
            rotation_matrix = torch.eye(4)  # Identity matrix
        rotation_matrix = rotation_matrix.float().to(grid_torch.device)
        # convert all floats to tensors
        input_horizontal_fov = torch.tensor(input_horizontal_fov, device=grid_torch.device).float()
        output_horizontal_fov = torch.tensor(output_horizontal_fov, device=grid_torch.device).float()

        # Calculate vertical field of view for input and output projections
        output_vertical_fov = output_horizontal_fov  # Assuming square aspect ratio
        input_vertical_fov = input_horizontal_fov * (grid_torch.shape[0] / grid_torch.shape[1])

        # Normalize the grid for vertical FOV adjustment
        normalized_grid = grid_torch.clone()
        normalized_grid[..., 1] *= (grid_torch.shape[0] / grid_torch.shape[1])

        # Step 1: Map each pixel to its location on the sphere for the output projection
        if output_projection == "PINHOLE":
            D = 1.0 / torch.tan(torch.deg2rad(output_horizontal_fov) / 2)
            radius_to_center = torch.sqrt(normalized_grid[..., 0]**2 + normalized_grid[..., 1]**2)
            phi = torch.atan2(normalized_grid[..., 1], normalized_grid[..., 0])
            theta = torch.atan2(radius_to_center, D)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
        elif output_projection == "FISHEYE":
            phi = torch.atan2(normalized_grid[..., 1], normalized_grid[..., 0])
            radius = torch.sqrt(normalized_grid[..., 0]**2 + normalized_grid[..., 1]**2)
            theta = radius * torch.deg2rad(output_horizontal_fov) / 2
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
        elif output_projection == "EQUIRECTANGULAR":
            phi = grid_torch[..., 0] * torch.deg2rad(output_horizontal_fov) / 2
            theta = grid_torch[..., 1] * torch.deg2rad(output_vertical_fov) / 2
            y = torch.sin(theta)
            x = torch.cos(theta) * torch.sin(phi)
            z = torch.cos(theta) * torch.cos(phi)
        else:
            raise ValueError(f"Unsupported output projection: {output_projection}")

        # Step 2: Apply rotation matrix for yaw and pitch
        coords = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        coords_homogeneous = torch.cat([coords, torch.ones((coords.shape[0], 1), device=coords.device)], dim=-1)
        coords_rotated = torch.matmul(rotation_matrix, coords_homogeneous.T).T
        coords = coords_rotated[..., :3]  # Extract x, y, z after rotation

        # Step 3: Map rotated points back to the input projection
        if input_projection == "PINHOLE":
            D = 1.0 / torch.tan(torch.deg2rad(input_horizontal_fov) / 2)
            theta = torch.atan2(torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2), coords[..., 2])
            phi = torch.atan2(coords[..., 1], coords[..., 0])
            radius = D * torch.tan(theta)
            x = radius * torch.cos(phi)
            y = radius * torch.sin(phi)
            mask = coords[..., 2] > 0  # Only keep points where z > 0
            x[~mask] = 100
            y[~mask] = 100
        elif input_projection == "FISHEYE":
            theta = torch.atan2(torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2), coords[..., 2])
            phi = torch.atan2(coords[..., 1], coords[..., 0])
            radius = theta / (torch.deg2rad(input_horizontal_fov) / 2)
            x = radius * torch.cos(phi)
            y = radius * torch.sin(phi)
        elif input_projection == "EQUIRECTANGULAR":
            theta = torch.asin(coords[..., 1])
            phi = torch.atan2(coords[..., 0], coords[..., 2])
            x = phi / (torch.deg2rad(input_horizontal_fov) / 2)
            y = theta / (torch.deg2rad(input_vertical_fov) / 2)
        else:
            raise ValueError(f"Unsupported input projection: {input_projection}")

        x = x.view(grid_torch.shape[0], grid_torch.shape[1])
        y = y.view(grid_torch.shape[0], grid_torch.shape[1])
        output_grid = torch.zeros_like(grid_torch)
        output_grid[..., 0] = x
        output_grid[..., 1] = y

    return output_grid


class ReprojectImage:
    """
    A node to reproject an image from one projection type to another.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for the node.

        Returns:
            dict: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "input_horiszontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "output_horiszontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "input_projection": (Projection.PROJECTIONS, {"tooltip": "input projection type"}),
                "output_projection": (Projection.PROJECTIONS, {"tooltip": "output projection type"}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "inverse": ("BOOLEAN", {"default": False}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 16384, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "transform_matrix": ("MAT_4X4", {"default": None}),
            }
        }

    RETURN_TYPES: Tuple[str, str] = ("IMAGE", "MASK")
    RETURN_NAMES = ("reprojected image", "reprojected mask")
    FUNCTION: str = "reproject_image"
    CATEGORY: str = "Camera/reproject"

    def reproject_image(
        self,
        image: torch.Tensor,
        input_horiszontal_fov: float,
        output_horiszontal_fov: float,
        input_projection: str,
        output_projection: str,
        output_width: int,
        output_height: int,
        feathering: int,
        inverse: bool=False,
        transform_matrix: np.ndarray=None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reproject an image from one projection type to another.

        Args:
            image (torch.Tensor): The input image tensor.
            input_horiszontal_fov (float): The horizontal field of view of the input image.
            output_horiszontal_fov (float): The horizontal field of view of the output image.
            input_projection (str): The projection type of the input image.
            output_projection (str): The projection type of the output image.
            output_width (int): The width of the output image.
            output_height (int): The height of the output image.
            transform_matrix (np.ndarray): The transformation matrix.
            inverse (bool): Whether to invert the transformation matrix.
            feathering (int): The feathering value for blending.
            mask (torch.Tensor, optional): The input mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The reprojected image (BHWC) and mask (HW), both normalized to 0-1.
        """
        if transform_matrix is None:
            transform_matrix = np.eye(4)
        transform_matrix=torch.from_numpy(transform_matrix).to(image.device)
        if transform_matrix.shape != (4, 4):
            transform_matrix = transform_matrix.view(4, 4)
        transform_matrix = transform_matrix.float()
        if inverse:
            transform_matrix = torch.inverse(transform_matrix)
        # Create output grid
        d1, input_height, input_width, d4 = image.size()  # Batch, height, width, channels
        image_tensor = image.permute(0, 3, 1, 2)  # Convert BHWC to BCHW

        # Add alpha channel if missing
        if d4 != 4:
            alpha_channel = torch.ones((d1, 1, input_height, input_width), dtype=image_tensor.dtype, device=image_tensor.device)
            image_tensor = torch.cat((image_tensor, alpha_channel), dim=1)

        # Add 1-pixel border with alpha = 0
        image_tensor[:, -1, :, 0] = 0
        image_tensor[:, -1, :, -1] = 0
        image_tensor[:, -1, 0, :] = 0
        image_tensor[:, -1, -1, :] = 0
        # pad till square with alpha = 0
        if input_height != input_width:
            if input_height < input_width:
                pad_top = (input_width - input_height) // 2
                pad_bottom = input_width - input_height - pad_top
                pad_left = 0
                pad_right = 0
            else:
                pad_top = 0
                pad_bottom = 0
                pad_left = (input_height - input_width) // 2
                pad_right = input_height - input_width - pad_left
            image_tensor = torch.nn.functional.pad(
                image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0
            )
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, output_width, device=image_tensor.device),
            torch.linspace(-1, 1, output_height, device=image_tensor.device),
            indexing="ij"
        )
        grid_init = torch.stack((grid_x, grid_y), dim=-1)
        grid = map_grid(
            grid_init, input_projection, output_projection,
            input_horiszontal_fov, output_horiszontal_fov, transform_matrix
        )

        # Sample input image using the grid
        grid = grid.unsqueeze(0)  # Add batch dimension
        sampled_tensor = torch.nn.functional.grid_sample(
            image_tensor, grid, mode='nearest', padding_mode='border', align_corners=False
        )

        # Extract and normalize mask
        if mask is None:
            mask = sampled_tensor[:, -1:, :, :]  # Extract alpha channel
            sampled_tensor = sampled_tensor * mask + 0.5 * (1 - mask)  # Blend with gray background

            mask = torch.nn.functional.avg_pool2d(
                mask, kernel_size=feathering * 2 + 1, stride=1, padding=feathering
            )
            mask = mask * (mask > 0)  # Feathering on border of image
            mask = (1 - mask).clamp(0, 1).squeeze(0).squeeze(0)  # Normalize to 0-1 range
        else:
            # Reproject the provided mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)  # Add batch dimension

            sampled_mask = torch.nn.functional.grid_sample(
                mask, grid, mode='nearest', padding_mode='border', align_corners=False
            )

            sampled_mask = torch.nn.functional.avg_pool2d(
                sampled_mask, kernel_size=feathering * 2 + 1, stride=1, padding=feathering
            )
            sampled_mask = sampled_mask * (sampled_mask > 0)  # Apply feathering

            mask = sampled_mask.clamp(0, 1)  # Normalize to 0-1 range

        # Normalize image to 0-1 range and convert to BHWC
        
        image = sampled_tensor[:, :-1, :, :].permute(0, 2, 3, 1)  # BCHW to BHWC
        if output_projection == "FISHEYE":
            # add circular mask to image
            mask_circle = torch.sqrt(grid_x**2 + grid_y**2) <= 1.0
            mask=mask*(mask_circle*1)
            image = image * mask_circle.unsqueeze(0).unsqueeze(-1)
        return image, mask[None, ...]  # add batch dimension to mask


class TransformToMatrix:
    """
    A node to convert shiftX, shiftY, shiftZ, theta, and phi into a 4x4 transformation matrix.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for the node.

        Returns:
            dict: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "shiftX": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "shiftY": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "shiftZ": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "theta": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "phi": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("MAT_4X4",)
    RETURN_NAMES = ("transformation matrix",)
    FUNCTION: str = "generate_matrix"
    CATEGORY: str = "Camera/reproject"

    def generate_matrix(
        self,
        shiftX: float,
        shiftY: float,
        shiftZ: float,
        theta: float,
        phi: float,
    ) -> np.ndarray:
        """
        Generate a 4x4 transformation matrix based on the inputs.

        Args:
            shiftX (float): Translation along the X-axis.
            shiftY (float): Translation along the Y-axis.
            shiftZ (float): Translation along the Z-axis.
            theta (float): Rotation angle around the Y-axis in degrees.
            phi (float): Rotation angle around the X-axis in degrees.
            inverse (bool): Whether to output the inverse matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        # Translation matrix
        T = np.eye(4)
        T[0, 3] = shiftX
        T[1, 3] = shiftY
        T[2, 3] = shiftZ

        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        R_theta = np.array([
            [np.cos(phi_rad), 0, np.sin(phi_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(phi_rad), 0, np.cos(phi_rad), 0],
            [0, 0, 0, 1]
        ])

        R_phi = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad), 0],
            [0, np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 0, 1]
        ])

        # Combine transformations
        M = np.matmul(T, np.matmul(R_theta, R_phi))
        return M[None, ...]  # Add batch dimension

# module to manually set rotation matrix
class TransformToMatrixManual:
    """
    A node to manually set a 4x4 transformation matrix using 16 individual inputs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for the node.

        Returns:
            dict: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "m00": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m01": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m02": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m03": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m10": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m11": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m12": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m13": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m20": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m21": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m22": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m23": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m30": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m31": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m32": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "m33": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("MAT_4X4",)
    RETURN_NAMES = ("transformation matrix",)
    FUNCTION: str = "generate_matrix"
    CATEGORY: str = "Camera/reproject"

    def generate_matrix(
        self,
        m00: float, m01: float, m02: float, m03: float,
        m10: float, m11: float, m12: float, m13: float,
        m20: float, m21: float, m22: float, m23: float,
        m30: float, m31: float, m32: float, m33: float,
    ) -> np.ndarray:
        """
        Generate a 4x4 transformation matrix based on the 16 inputs.

        Args:
            m00, m01, ..., m33 (float): Individual elements of the 4x4 matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix with batch dimension added.
        """
        matrix = np.array([
            [m00, m01, m02, m03],
            [m10, m11, m12, m13],
            [m20, m21, m22, m23],
            [m30, m31, m32, m33],
        ], dtype=np.float32)
        return matrix[None, ...]  # Add batch dimension

class ReprojectDepth:
    """
    A node to reproject a depth tensor from one projection type to another,
    and also return a mask of valid pixels.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth": ("TENSOR",),
                "input_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "output_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "input_projection": (Projection.PROJECTIONS, {"tooltip": "input projection type"}),
                "output_projection": (Projection.PROJECTIONS, {"tooltip": "output projection type"}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "transform_matrix": ("MAT_4X4", {"default": None}),
                "inverse": ("BOOLEAN", {"default": False}),
            }
        }

    # Now returns both depth and mask
    RETURN_TYPES: Tuple[str, str] = ("TENSOR", "MASK")
    RETURN_NAMES = ("reprojected_depth", "reprojected_mask")
    FUNCTION: str = "reproject_depth"
    CATEGORY: str = "Camera/reproject"

    def reproject_depth(
        self,
        depth: torch.Tensor,
        input_horizontal_fov: float,
        output_horizontal_fov: float,
        input_projection: str,
        output_projection: str,
        output_width: int,
        output_height: int,
        transform_matrix: np.ndarray = None,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reproject a depth tensor and return a mask of valid pixels.

        Returns:
            reprojected_depth: torch.Tensor shaped [H_out, W_out] (or batch if provided)
            reprojected_mask: torch.Tensor same shape, 1.0 where valid, 0.0 where out-of-bounds
        """
        # Prepare transform
        device, dtype = depth.device, depth.dtype
        if transform_matrix is None:
            T = torch.eye(4, device=device, dtype=dtype)
        else:
            T = torch.from_numpy(transform_matrix).to(device=device, dtype=dtype).view(4, 4)
        if inverse:
            T = torch.inverse(T)

        # Normalize dims
        x = depth
        added_batch = added_channel = False
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0); added_batch = added_channel = True
        elif x.dim() == 3:
            x = x.unsqueeze(0); added_batch = True
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported depth tensor with {x.dim()} dims")

        B, C, H_in, W_in = x.shape
        H_out = output_height if output_height > 0 else H_in
        W_out = output_width  if output_width  > 0 else W_in

        # Build grid and reproject
        ys = torch.linspace(-1, 1, H_out, device=device)
        xs = torch.linspace(-1, 1, W_out, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = map_grid(grid, input_projection, output_projection,
                        input_horizontal_fov, output_horizontal_fov, T)
        grid = grid.unsqueeze(0)  # add batch

        # Sample depth
        sampled = torch.nn.functional.grid_sample(
            x, grid, mode='nearest', padding_mode='border', align_corners=False
        )

        # Generate validity mask by sampling an all-ones tensor with zero padding
        mask_in = torch.ones_like(x)
        sampled_mask = torch.nn.functional.grid_sample(
            mask_in, grid, mode='nearest', padding_mode='zeros', align_corners=False
        )

        # Squeeze back to remove added dims
        if added_batch:
            sampled = sampled.squeeze(0)
            sampled_mask = sampled_mask.squeeze(0)
        if added_channel:
            sampled = sampled.squeeze(0)
            sampled_mask = sampled_mask.squeeze(0)

        # Convert depth back to original format
        # sampled shape: [C, H_out, W_out] or [H_out, W_out] if single channel
        reprojected_depth = sampled
        reprojected_mask  = (sampled_mask > 0.5).float()
        # if fisheye, add circular mask
        if output_projection == "FISHEYE":
            mask_circle = (grid_x**2 + grid_y**2) <= 1.0
            reprojected_mask = reprojected_mask * mask_circle.unsqueeze(0).unsqueeze(0)
        return reprojected_depth.permute(0,2,3,1), reprojected_mask


NODE_CLASS_MAPPINGS = {
    "ReprojectImage": ReprojectImage,
    "TransformToMatrix": TransformToMatrix,
    "TransformToMatrixManual": TransformToMatrixManual,
    "ReprojectDepth": ReprojectDepth
}
