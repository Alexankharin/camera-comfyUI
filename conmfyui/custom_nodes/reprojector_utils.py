# Organize imports
import numpy as np
import torch

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

        # Calculate vertical field of view for input and output projections
        output_vertical_fov = output_horizontal_fov  # Assuming square aspect ratio
        input_vertical_fov = input_horizontal_fov * (grid_torch.shape[0] / grid_torch.shape[1])

        # Normalize the grid for vertical FOV adjustment
        normalized_grid = grid_torch.clone()
        normalized_grid[..., 1] = grid_torch[..., 1] * (grid_torch.shape[0] / grid_torch.shape[1])

        # Step 1: Map each pixel to its location on the sphere for the output projection
        if output_projection == "PINHOLE":
            D = 1.0 / torch.tan(torch.deg2rad(torch.tensor(output_horizontal_fov)) / 2)
            radius_to_center = torch.sqrt(normalized_grid[..., 0]**2 + normalized_grid[..., 1]**2)
            phi = torch.atan2(normalized_grid[..., 1], normalized_grid[..., 0])
            theta = torch.atan2(radius_to_center, D)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
        elif output_projection == "FISHEYE":
            # Use normalized_grid for both x and y coordinates
            phi = torch.atan2(normalized_grid[..., 1], normalized_grid[..., 0])
            radius = torch.sqrt(normalized_grid[..., 0]**2 + normalized_grid[..., 1]**2)
            theta = radius * torch.deg2rad(torch.tensor(output_horizontal_fov)) / 2
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
        elif output_projection == "EQUIRECTANGULAR":
            phi = grid_torch[..., 0] * torch.deg2rad(torch.tensor(output_horizontal_fov)) / 2
            theta = grid_torch[..., 1] * torch.deg2rad(torch.tensor(output_vertical_fov)) / 2
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
            D = 1.0 / torch.tan(torch.deg2rad(torch.tensor(input_horizontal_fov)) / 2)
            theta = torch.atan2(torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2), coords[..., 2])
            phi = torch.atan2(coords[..., 1], coords[..., 0])
            radius = D * torch.tan(theta)
            x = radius * torch.cos(phi)
            y = radius * torch.sin(phi)
            # Exclude points on the back hemisphere
            mask = coords[..., 2] > 0  # Only keep points where z > 0
            x[~mask] = 100
            y[~mask] = 100
        elif input_projection == "FISHEYE":
            theta = torch.atan2(torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2), coords[..., 2])
            phi = torch.atan2(coords[..., 1], coords[..., 0])
            radius = theta / (torch.deg2rad(torch.tensor(input_horizontal_fov)) / 2)
            x = radius * torch.cos(phi)
            y = radius * torch.sin(phi)
        elif input_projection == "EQUIRECTANGULAR":
            # In the forward equirectangular mapping, we used:
            #   y = sin(theta) and x = cos(theta) * sin(phi), z = cos(theta) * cos(phi)
            # The inverse is then:
            theta = torch.asin(coords[..., 1])
            phi = torch.atan2(coords[..., 0], coords[..., 2])
            x = phi / (torch.deg2rad(torch.tensor(input_horizontal_fov)) / 2)
            y = theta / (torch.deg2rad(torch.tensor(input_vertical_fov)) / 2)
        else:
            raise ValueError(f"Unsupported input projection: {input_projection}")

        x = x.view(grid_torch.shape[0], grid_torch.shape[1])
        y = y.view(grid_torch.shape[0], grid_torch.shape[1])
        output_grid = torch.zeros_like(grid_torch)
        output_grid[..., 0] = x
        output_grid[..., 1] = y

    return output_grid
