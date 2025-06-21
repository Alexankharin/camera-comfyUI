from .pointcloud_nodes import NODE_CLASS_MAPPINGS as NCM1
from .reprojection_nodes import NODE_CLASS_MAPPINGS as NCM2
from .metric_depth_nodes import NODE_CLASS_MAPPINGS as NCM3
from .flux_fisheye_filling_nodes import NODE_CLASS_MAPPINGS as NCM4
from .complex_nodes import NODE_CLASS_MAPPINGS as NCM5
from .video_nodes import NODE_CLASS_MAPPINGS as NCM6
NODE_CLASS_MAPPINGS = {**NCM1, **NCM2, **NCM3, **NCM4, **NCM5, **NCM6}

__all__ = ["NODE_CLASS_MAPPINGS"]