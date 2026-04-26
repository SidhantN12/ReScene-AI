"""ReScene AI — compositing utility functions."""

from __future__ import annotations

import cv2
import numpy as np


def alpha_composite(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    position: tuple[int, int],
) -> np.ndarray:
    """Alpha-blend *foreground_rgba* onto *background* at *position*.

    Parameters
    ----------
    background     : H×W×3 uint8 RGB
    foreground_rgba: H_f×W_f×4 uint8 RGBA
    position       : (x, y) — top-left corner of foreground in background coords

    Returns
    -------
    composited : H×W×3 uint8 RGB
    """
    result = background.copy()
    bh, bw = background.shape[:2]
    fh, fw = foreground_rgba.shape[:2]
    x0, y0 = position

    # Compute valid intersection
    x1, y1 = x0 + fw, y0 + fh
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x1, bw), min(y1, bh)
    if bx0 >= bx1 or by0 >= by1:
        return result  # no overlap

    fx0 = bx0 - x0
    fy0 = by0 - y0
    fx1 = fx0 + (bx1 - bx0)
    fy1 = fy0 + (by1 - by0)

    fg_rgb = foreground_rgba[fy0:fy1, fx0:fx1, :3].astype(np.float32)
    fg_alpha = foreground_rgba[fy0:fy1, fx0:fx1, 3:4].astype(np.float32) / 255.0
    bg_region = result[by0:by1, bx0:bx1].astype(np.float32)

    blended = fg_rgb * fg_alpha + bg_region * (1.0 - fg_alpha)
    result[by0:by1, bx0:bx1] = blended.clip(0, 255).astype(np.uint8)
    return result


def paste_with_mask(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    position: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Composite *foreground_rgba* onto *background* and return the object mask.

    The *position* is the **centre** of the foreground (not top-left corner).

    Returns
    -------
    (composited_image, object_mask) where object_mask is H×W uint8
    """
    bh, bw = background.shape[:2]
    fh, fw = foreground_rgba.shape[:2]
    cx, cy = position

    x0 = cx - fw // 2
    y0 = cy - fh // 2

    composited = alpha_composite(background, foreground_rgba, (x0, y0))

    # Build object mask in background coordinate space
    obj_mask = np.zeros((bh, bw), dtype=np.uint8)
    x1, y1 = x0 + fw, y0 + fh
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x1, bw), min(y1, bh)
    if bx0 < bx1 and by0 < by1:
        fx0 = bx0 - x0
        fy0 = by0 - y0
        fx1 = fx0 + (bx1 - bx0)
        fy1 = fy0 + (by1 - by0)
        alpha_patch = foreground_rgba[fy0:fy1, fx0:fx1, 3]
        obj_mask[by0:by1, bx0:bx1] = alpha_patch

    return composited, obj_mask


def feather_mask(mask: np.ndarray, radius: int = 10) -> np.ndarray:
    """Return a float32 [0,1] feathered version of *mask* for soft compositing."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (radius * 2 + 1, radius * 2 + 1), radius / 3)
    return (blurred / 255.0).clip(0.0, 1.0)


def poisson_blend(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    position: tuple[int, int],
) -> np.ndarray:
    """Seamlessly blend *source* into *target* using Poisson blending (cv2).

    Parameters
    ----------
    source   : H_s×W_s×3 uint8 RGB — patch to insert
    target   : H×W×3 uint8 RGB — destination image
    mask     : H_s×W_s uint8 — 255 inside the source region to blend
    position : (x, y) centre of source in target coordinates

    Returns
    -------
    blended : H×W×3 uint8 RGB
    """
    source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    try:
        blended_bgr = cv2.seamlessClone(
            src=source_bgr,
            dst=target_bgr,
            mask=mask,
            p=position,
            flags=cv2.NORMAL_CLONE,
        )
    except cv2.error:
        # Fallback to direct paste if Poisson fails (e.g. mask too small)
        blended_bgr = target_bgr.copy()
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def create_checkerboard(height: int, width: int, tile: int = 32) -> np.ndarray:
    """Return H×W×3 uint8 checkerboard for transparent region visualisation."""
    board = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                board[y:y + tile, x:x + tile] = 200
            else:
                board[y:y + tile, x:x + tile] = 128
    return np.stack([board] * 3, axis=-1)


def perspective_transform(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply a perspective warp mapping src_points → dst_points.

    Parameters
    ----------
    image       : H×W×C uint8 (RGB or RGBA)
    src_points  : (4, 2) float32 — source quadrilateral corners (TL, TR, BR, BL)
    dst_points  : (4, 2) float32 — destination quadrilateral corners
    output_size : (width, height) output canvas; defaults to input image size

    Returns
    -------
    warped : same dtype/channels as input, alpha preserved for RGBA
    """
    h, w = image.shape[:2]
    out_w, out_h = output_size or (w, h)
    src = src_points.astype(np.float32)
    dst = dst_points.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    flags = cv2.INTER_LANCZOS4

    if image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.warpPerspective(image[:, :, :3], H, (out_w, out_h), flags=flags)
        alpha = cv2.warpPerspective(image[:, :, 3], H, (out_w, out_h), flags=flags)
        return np.dstack([rgb, alpha])
    return cv2.warpPerspective(image, H, (out_w, out_h), flags=flags)


def simple_color_match(
    fg: np.ndarray,
    bg: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Match *fg* colour distribution to the *bg* region sampled under *mask*.

    Uses per-channel mean+std transfer (Reinhard colour transfer).  Fast and
    dependency-free; serves as a stand-in for learned harmonisation.

    Parameters
    ----------
    fg   : H_f×W_f×3 uint8 RGB — foreground patch to colour-correct
    bg   : H×W×3 uint8 RGB — background scene used as colour reference
    mask : H×W uint8 — 255 where to sample background statistics

    Returns
    -------
    matched : H_f×W_f×3 uint8 RGB — fg with colours shifted to match bg
    """
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)

    bg_pixels = bg_f[mask > 127]  # (N, 3)
    if len(bg_pixels) < 10:
        return fg  # not enough context

    result = fg_f.copy()
    for c in range(3):
        mu_fg = fg_f[:, :, c].mean()
        std_fg = fg_f[:, :, c].std() + 1e-6
        mu_bg = float(bg_pixels[:, c].mean())
        std_bg = float(bg_pixels[:, c].std()) + 1e-6
        result[:, :, c] = (fg_f[:, :, c] - mu_fg) * (std_bg / std_fg) + mu_bg

    return result.clip(0, 255).astype(np.uint8)
