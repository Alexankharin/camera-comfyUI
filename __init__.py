from .pointcloud_nodes import NODE_CLASS_MAPPINGS as NCM1
from .reprojection_nodes import NODE_CLASS_MAPPINGS as NCM2
from .metric_depth_nodes import NODE_CLASS_MAPPINGS as NCM3
NODE_CLASS_MAPPINGS = {**NCM1, **NCM2, **NCM3}

__all__ = ["NODE_CLASS_MAPPINGS"]