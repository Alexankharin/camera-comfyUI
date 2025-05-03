from .pointcloud_nodes import NODE_CLASS_MAPPINGS as NCM1
from .reprojection_nodes import NODE_CLASS_MAPPINGS as NCM2
from .metric_depth_nodes import NODE_CLASS_MAPPINGS as NCM3
from .flux_fisheye_filling_nodes import NODE_CLASS_MAPPINGS as NCM4
NODE_CLASS_MAPPINGS = {**NCM1, **NCM2, **NCM3, **NCM4}

__all__ = ["NODE_CLASS_MAPPINGS"]