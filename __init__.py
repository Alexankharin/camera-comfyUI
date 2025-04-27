from .pointcloud_nodes import NODE_CLASS_MAPPINGS as NCM1
from .reprojection_nodes import NODE_CLASS_MAPPINGS as NCM2
NODE_CLASS_MAPPINGS = {**NCM1, **NCM2}

__all__ = ["NODE_CLASS_MAPPINGS"]