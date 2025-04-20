# camera-comfyUI

## Overview
This repository contains custom nodes for ComfyUI, designed to handle various projection models, camera movements, and point cloud manipulations. These nodes can be used to enhance workflows, create dynamic visual outputs, and perform continuous reprojection tasks.

## Features
- **Reprojection Nodes**: Enable continuous reprojection between different camera models and projections.
- **Point Cloud Nodes**: Work with point clouds, including depth-to-point cloud conversion, transformations, and rendering.
- Support for multiple projection models: Pinhole, Fisheye, and Equirectangular.
- Customizable camera movement and transformation nodes.
- Example workflows to demonstrate functionality.

## Installation
1. Clone this repository.
2. Move the `custom_nodes` folder to your ComfyUI installation directory.

## Node Categories

### Reprojection Nodes
These nodes allow for continuous reprojection between different camera models and projections. They include:
- **ReprojectImage**: Reprojects an image from one projection type to another.
- **TransformToMatrix**: Converts translation and rotation parameters into a 4x4 transformation matrix.
- **TransformToMatrixManual**: Allows manual input of a 4x4 transformation matrix.

### Point Cloud Nodes
These nodes enable working with point clouds and depth maps. They include:
- **DepthToPointCloud**: Converts a depth map and optional RGB(A) image into a point cloud.
- **TransformPointCloud**: Applies a 4x4 transformation matrix to a point cloud.
- **ProjectPointCloud**: Projects a point cloud back into an image and mask using z-buffering.
- **DepthAwareInpainter**: Inpaints holes in an RGB image using depth-aware propagation over a provided mask.

## Usage
1. **Reprojection Nodes**:
   - Use the `ReprojectImage` node to reproject images between different camera models and projections.
   - Use the `TransformToMatrix` or `TransformToMatrixManual` nodes to create transformation matrices for reprojection.

2. **Point Cloud Nodes**:
   - Use the `DepthToPointCloud` node to convert depth maps into point clouds.
   - Apply transformations to point clouds using the `TransformPointCloud` node.
   - Render point clouds back into images using the `ProjectPointCloud` node.
   - Use the `DepthAwareInpainter` node to fill holes in images with depth-aware inpainting.

## Example Workflows
Example workflows are provided in the `workflows/` folder to help you get started:
- `demo_camera_workflow.json`: Demonstrates camera movement and reprojection.
- `outpainting_fisheye.json` and `outpainting_fisheye_kitchen.json`: Showcases fisheye outpainting workflows.
- `PointCloud.json`: Demonstrates point cloud manipulation and rendering.

## Contributing
Feel free to submit issues or pull requests to improve this repository. Contributions are welcome!

## Notes
- Place `node_typing` into the `comfyui/custom_nodes/` folder.
- Place the rest of the files into the `comfyui/custom_nodes/camera/` folder.