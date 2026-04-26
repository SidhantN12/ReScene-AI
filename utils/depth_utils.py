"""ReScene AI — depth map utility functions."""

from __future__ import annotations

import cv2
import numpy as np


def normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Normalise depth map to [0, 1] float32."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth → uint8 [0, 255] for visualisation (near=bright)."""
    norm = normalise_depth(depth)
    # Invert so near objects appear bright
    return ((1.0 - norm) * 255).astype(np.uint8)


def depth_colourmap(depth: np.ndarray) -> np.ndarray:
    """Return H×W×3 uint8 RGB false-colour depth map using TURBO colourmap."""
    grey = depth_to_uint8(depth)
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)


def compute_floor_plane_mask(
    depth: np.ndarray,
    percentile: float = 20.0,
) -> np.ndarray:
    """Heuristic: return a mask of pixels likely belonging to the floor.

    The floor is the largest near-camera planar region in the bottom third of
    the image.  Returns uint8 mask, 255 = floor region.
    """
    h = depth.shape[0]
    bottom_strip = depth[int(h * 0.6):, :]
    threshold = np.percentile(bottom_strip, percentile)
    floor_mask = np.zeros(depth.shape, dtype=np.uint8)
    floor_mask[int(h * 0.6):][bottom_strip < threshold] = 255
    return floor_mask


def sample_depth_region(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    reduction: str = "median",
) -> float:
    """Return a scalar depth representative of a bounding box region.

    Parameters
    ----------
    bbox : (x0, y0, x1, y1)
    reduction : "median", "mean", or "min"
    """
    x0, y0, x1, y1 = bbox
    region = depth[y0:y1, x0:x1]
    if region.size == 0:
        return float(depth.mean())
    if reduction == "median":
        return float(np.median(region))
    elif reduction == "min":
        return float(region.min())
    return float(region.mean())


def scale_object_by_depth(
    object_size_px: tuple[int, int],
    src_depth: float,
    dst_depth: float,
    reference_depth: float = 2.5,
) -> tuple[int, int]:
    """Compute the pixel size an object should have at *dst_depth*.

    Objects that move closer (smaller dst_depth) grow; farther shrink.

    Parameters
    ----------
    object_size_px : (width, height) of the object at *src_depth*
    src_depth      : depth at original position (metres)
    dst_depth      : depth at target position (metres)
    reference_depth: normalisation constant (doesn't affect relative scaling)

    Returns
    -------
    (new_width, new_height) in pixels
    """
    if dst_depth < 1e-3:
        dst_depth = 1e-3
    if src_depth < 1e-3:
        src_depth = 1e-3
    scale = src_depth / dst_depth
    w = max(1, int(object_size_px[0] * scale))
    h = max(1, int(object_size_px[1] * scale))
    return w, h
