# Organize imports
from PIL import Image
import numpy as np
import torch
import time
from typing import Dict, Tuple, Any

# Local imports
import comfy.utils
import comfy.projections

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
                "input_projection": (comfy.projections.Projection.PROJECTIONS, {"tooltip": "input projection type"}),
                "output_projection": (comfy.projections.Projection.PROJECTIONS, {"tooltip": "output projection type"}),
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
    FUNCTION: str = "reproject_image"
    CATEGORY: str = "image"

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
        grid = comfy.projections.map_grid(
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

            mask = sampled_mask.squeeze(0).clamp(0, 1)  # Normalize to 0-1 range

        # Normalize image to 0-1 range and convert to BHWC
        image = sampled_tensor[:, :-1, :, :].permute(0, 2, 3, 1)  # BCHW to BHWC

        return image, mask


class ReprojectMask:
    """
    A node to reproject a mask from one projection type to another.
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
                "mask": ("MASK",),
                "input_horiszontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "output_horiszontal_fov": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "input_projection": (comfy.projections.Projection.PROJECTIONS, {"tooltip": "input projection type"}),
                "output_projection": (comfy.projections.Projection.PROJECTIONS, {"tooltip": "output projection type"}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "inverse": ("BOOLEAN", {"default": False}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 16384, "step": 1}),
            },
            "optional": {
                "transform_matrix": ("MAT_4X4", {"default": None}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("MASK",)
    FUNCTION: str = "reproject_mask"
    CATEGORY: str = "mask"

    def reproject_mask(
        self,
        mask: torch.Tensor,
        input_horiszontal_fov: float,
        output_horiszontal_fov: float,
        input_projection: str,
        output_projection: str,
        output_width: int,
        output_height: int,
        feathering: int,
        transform_matrix: np.ndarray=None,
        inverse: bool=False,
    ) -> torch.Tensor:
        """
        Reproject a mask from one projection type to another.

        Args:
            mask (torch.Tensor): The input mask tensor.
            input_horiszontal_fov (float): The horizontal field of view of the input mask.
            output_horiszontal_fov (float): The horizontal field of view of the output mask.
            input_projection (str): The projection type of the input mask.
            output_projection (str): The projection type of the output mask.
            output_width (int): The width of the output mask.
            output_height (int): The height of the output mask.
            transform_matrix (torch.Tensor): The transformation matrix.
            inverse (bool): Whether to invert the transformation matrix.
            feathering (int): The feathering value for blending.

        Returns:
            torch.Tensor: The reprojected mask (HW), normalized to 0-1.
        """
        if transform_matrix is None:
            transform_matrix = np.eye(4)
        transform_matrix = torch.from_numpy(transform_matrix).to(mask.device)
        if inverse:
            transform_matrix = torch.inverse(transform_matrix)

        # Add batch and channel dimensions if missing
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)  # Add batch dimension

        # Create output grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, output_width, device=mask.device),
            torch.linspace(-1, 1, output_height, device=mask.device),
            indexing="ij"
        )
        grid_init = torch.stack((grid_x, grid_y), dim=-1)
        grid = comfy.projections.map_grid(
            grid_init, input_projection, output_projection,
            input_horiszontal_fov, output_horiszontal_fov, transform_matrix
        )

        # Sample input mask using the grid
        grid = grid.unsqueeze(0)  # Add batch dimension
        sampled_mask = torch.nn.functional.grid_sample(
            mask, grid, mode='nearest', padding_mode='border', align_corners=False
        )

        # Feathering on the border of the mask
        sampled_mask = torch.nn.functional.avg_pool2d(
            sampled_mask, kernel_size=feathering*2+1, stride=1, padding=feathering
        )
        sampled_mask = sampled_mask * (sampled_mask > 0)  # Apply feathering

        # Normalize mask to 0-1 range and remove batch and channel dimensions
        mask = sampled_mask.squeeze(0).clamp(0, 1)

        return mask


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
                "shiftX": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "shiftY": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "shiftZ": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "theta": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "phi": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("MAT_4X4",)
    FUNCTION: str = "generate_matrix"
    CATEGORY: str = "transformation"

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
            [np.cos(theta_rad), 0, np.sin(theta_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad), 0],
            [0, 0, 0, 1]
        ])

        R_phi = np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi_rad), -np.sin(phi_rad), 0],
            [0, np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 0, 1]
        ])

        # Combine transformations
        M = np.matmul(T, np.matmul(R_theta, R_phi))
        return M[None, ...]  # Add batch dimension


NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
    "ReprojectImage": ReprojectImage,
    "ReprojectMask": ReprojectMask,
    "TransformToMatrix": TransformToMatrix,
}
