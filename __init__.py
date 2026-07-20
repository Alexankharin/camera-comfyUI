from .pointcloud_nodes import NODE_CLASS_MAPPINGS as NCM1
from .reprojection_nodes import NODE_CLASS_MAPPINGS as NCM2
from .metric_depth_nodes import NODE_CLASS_MAPPINGS as NCM3
from .flux_fisheye_filling_nodes import NODE_CLASS_MAPPINGS as NCM4
from .complex_nodes import NODE_CLASS_MAPPINGS as NCM5
from .video_nodes import NODE_CLASS_MAPPINGS as NCM6
from .GS_nodes import NODE_CLASS_MAPPINGS as NCM7

# Optional node packs: a missing/broken optional dependency must never kill the
# whole extension (mirrors how video_nodes degrades when video_depth_anything
# is unavailable).
try:
    from .GS4D_nodes import NODE_CLASS_MAPPINGS as NCM8
except Exception as _exc:
    print(f"[camera-comfyUI] Warning: GS4D_nodes could not be loaded, 4D splat nodes disabled: {_exc}")
    NCM8 = {}
try:
    from .pose_nodes import NODE_CLASS_MAPPINGS as NCM9
except Exception as _exc:
    print(f"[camera-comfyUI] Warning: pose_nodes could not be loaded, pose estimation nodes disabled: {_exc}")
    NCM9 = {}
try:
    from .world_nodes import NODE_CLASS_MAPPINGS as NCM10
except Exception as _exc:
    print(f"[camera-comfyUI] Warning: world_nodes could not be loaded, world-building nodes disabled: {_exc}")
    NCM10 = {}
try:
    from .video4d_nodes import NODE_CLASS_MAPPINGS as NCM11
except Exception as _exc:
    print(f"[camera-comfyUI] Warning: video4d_nodes could not be loaded, per-frame 4D splat nodes disabled: {_exc}")
    NCM11 = {}

NODE_CLASS_MAPPINGS = {**NCM1, **NCM2, **NCM3, **NCM4, **NCM5, **NCM6, **NCM7, **NCM8, **NCM9, **NCM10, **NCM11}

__all__ = ["NODE_CLASS_MAPPINGS"]
