"""ReScene AI — utils package."""

from utils.image_utils import (
    ensure_rgb, resize_long_edge, resize_to_multiple,
    numpy_to_pil, pil_to_numpy, load_image, save_image, dilate_mask,
)
from utils.depth_utils import (
    normalise_depth, depth_to_uint8, depth_colourmap,
    compute_floor_plane_mask, sample_depth_region, scale_object_by_depth,
)
from utils.composite_utils import (
    alpha_composite, paste_with_mask, feather_mask,
    poisson_blend, create_checkerboard,
)

__all__ = [
    "ensure_rgb", "resize_long_edge", "resize_to_multiple",
    "numpy_to_pil", "pil_to_numpy", "load_image", "save_image", "dilate_mask",
    "normalise_depth", "depth_to_uint8", "depth_colourmap",
    "compute_floor_plane_mask", "sample_depth_region", "scale_object_by_depth",
    "alpha_composite", "paste_with_mask", "feather_mask",
    "poisson_blend", "create_checkerboard",
]
