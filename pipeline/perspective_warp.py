"""
ReScene AI — Perspective warp wrapper (depth-aware, OpenCV-based).

Replaces the ST-GAN placeholder with a fully functional pure-CV pipeline:

  1. Resize the foreground RGBA to the depth-corrected target_size.
  2. Apply a subtle trapezoid keystone warp that simulates the perspective
     foreshortening that objects exhibit at their new depth in the scene.
     Objects placed further away (scale_factor < 1) get more top-edge
     compression; closer objects get none.
  3. Per-channel mean+std colour transfer from the scene patch so the
     object's lighting roughly matches its new surroundings.

This produces visually plausible results without any trained weights.  When
ST-GAN or a real homography estimator is integrated later, this file is the
only one that needs to change — the orchestrator interface is identical.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from config import Config
from utils.composite_utils import perspective_transform

logger = logging.getLogger(__name__)

# Maximum top-edge compression ratio (15 % per side at deepest scene depth)
_MAX_TOP_SQUEEZE = 0.15
# Blend strength for colour transfer into the object (0 = no transfer, 1 = full)
_COLOR_TRANSFER_STRENGTH = 0.25


class PerspectiveWarpWrapper:
    """Depth-aware perspective correction for furniture placement.

    Requires no trained weights — uses OpenCV homography + colour transfer.
    The interface matches the placeholder so the orchestrator is unchanged.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None   # sentinel; True when loaded

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading perspective warp (OpenCV depth-aware, no weights).")
        self._model = True
        logger.info("Perspective warp ready.")

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                logger.warning("Perspective warp CUDA cleanup skipped: %s", exc)
        logger.info("Perspective warp unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        foreground_rgba: np.ndarray,
        scene_patch: np.ndarray,
        target_size: tuple[int, int],
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        """Warp *foreground_rgba* to fit perspective and lighting of *scene_patch*.

        Parameters
        ----------
        foreground_rgba : H_f×W_f×4 uint8 RGBA — furniture cutout (alpha channel)
        scene_patch     : H_p×W_p×3 uint8 RGB  — scene crop at the insertion point
        target_size     : (width, height) — depth-corrected output dimensions
                          (already equals original_size * scale_factor; do NOT
                          re-apply scale_factor here)
        scale_factor    : depth ratio — used for perspective distortion strength

        Returns
        -------
        warped_rgba : target_h×target_w×4 uint8 RGBA
        """
        if not self.is_loaded():
            raise RuntimeError("PerspectiveWarpWrapper not loaded. Call load() first.")

        tw, th = max(target_size[0], 4), max(target_size[1], 4)

        # ── Step 1: resize to depth-corrected target size ────────────────
        resized = cv2.resize(foreground_rgba, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # ── Step 2: trapezoid perspective warp ───────────────────────────
        # Compress top edge when the object sits deeper in the scene.
        # scale_factor < 1  →  object is further away  →  more squeeze.
        # scale_factor >= 1 →  object is at/closer than reference  →  no squeeze.
        top_squeeze = float(np.clip((1.0 - scale_factor) * _MAX_TOP_SQUEEZE / 0.5,
                                    0.0, _MAX_TOP_SQUEEZE))
        squeeze_px = int(top_squeeze * tw / 2)

        if squeeze_px >= 2:
            src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
            dst = np.float32([
                [squeeze_px,      0],
                [tw - squeeze_px, 0],
                [tw,              th],
                [0,               th],
            ])
            resized = perspective_transform(resized, src, dst, output_size=(tw, th))

        # ── Step 3: colour transfer from scene_patch ─────────────────────
        if scene_patch is not None and scene_patch.size > 0:
            resized = _apply_color_transfer(resized, scene_patch, _COLOR_TRANSFER_STRENGTH)

        return resized

    # ------------------------------------------------------------------
    # Static utility: depth-only homography (available to orchestrator)
    # ------------------------------------------------------------------

    @staticmethod
    def depth_homography(
        depth_map: np.ndarray,
        src_point: tuple[int, int],
        dst_point: tuple[int, int],
    ) -> np.ndarray:
        """Return a 3×3 scale-only homography derived from depth at two points."""
        d_src = max(float(depth_map[src_point[1], src_point[0]]), 0.01)
        d_dst = max(float(depth_map[dst_point[1], dst_point[0]]), 0.01)
        scale = d_src / d_dst
        H = np.eye(3, dtype=np.float64)
        H[0, 0] = scale
        H[1, 1] = scale
        H[0, 2] = dst_point[0] * (1.0 - scale)
        H[1, 2] = dst_point[1] * (1.0 - scale)
        return H


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_color_transfer(
    rgba: np.ndarray,
    scene_patch: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Shift the RGB channels of *rgba* toward the colour statistics of *scene_patch*.

    Only pixels with alpha > 10 contribute to the fg statistics, so
    transparent regions don't skew the colour transfer.
    """
    alpha = rgba[:, :, 3]
    visible = alpha > 10

    if not visible.any():
        return rgba

    fg_f = rgba[:, :, :3].astype(np.float32)
    bg_f = scene_patch.astype(np.float32)
    bg_flat = bg_f.reshape(-1, 3)

    result_f = fg_f.copy()
    for c in range(3):
        fg_vals = fg_f[visible, c]
        mu_fg = fg_vals.mean()
        std_fg = fg_vals.std() + 1e-6
        mu_bg = float(bg_flat[:, c].mean())
        std_bg = float(bg_flat[:, c].std()) + 1e-6

        ratio = float(np.clip(std_bg / std_fg, 0.5, 2.0))
        transferred = (fg_f[:, :, c] - mu_fg) * ratio + mu_bg
        # Blend: partial transfer controlled by strength
        result_f[:, :, c] = fg_f[:, :, c] * (1.0 - strength) + transferred * strength

    rgb_matched = result_f.clip(0, 255).astype(np.uint8)
    return np.dstack([rgb_matched, alpha])
