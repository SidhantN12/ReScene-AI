"""
ReScene AI — Image harmonization wrapper (iHarmony4 / Poisson-blending).

Real implementation uses iHarmony4 (CDTNet or DoveNet) to re-light and
re-colour a composited foreground region so it visually matches the background.

This implementation uses two complementary stages:
  1. LAB mean+std colour transfer — adaptively shifts the foreground's colour
     distribution toward the surrounding background ring, using a strength
     proportional to the statistical difference (10–55 % max).  This preserves
     the object's original appearance while nudging it toward scene lighting.
  2. Poisson seamless cloning (cv2.NORMAL_CLONE) — solves a Poisson system over
     the foreground region so gradient transitions at the mask boundary are
     smooth and seamlessly integrated with the background.  A soft Gaussian
     fallback handles edge-adjacent or very-small masks where the Poisson solver
     cannot converge.

Together these stages produce harmonization comparable to simple neural
approaches at a fraction of the compute cost and with no weights required.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from config import Config

logger = logging.getLogger(__name__)

# Radius of the background ring sampled as colour reference
_RING_RADIUS: int = 31
# Max LAB transfer strength regardless of statistical difference
_MAX_STRENGTH: float = 0.55


class HarmonizationWrapper:
    """Poisson + LAB harmonization wrapper (no trained weights)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None

    def load(self) -> None:
        logger.info("Loading harmonization (Poisson cloning + LAB transfer, no weights).")
        self._model = True
        logger.info("Harmonization ready.")

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Harmonization unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, composite: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        """Harmonize foreground region in *composite* to match background.

        Parameters
        ----------
        composite       : H×W×3 uint8 RGB — scene with foreign object pasted in
        foreground_mask : H×W uint8 — 255 where the pasted object is

        Returns
        -------
        harmonized : H×W×3 uint8 RGB
        """
        if not self.is_loaded():
            raise RuntimeError("Harmonization not loaded. Call load() first.")

        fg_mask = foreground_mask > 127
        bg_mask = ~fg_mask

        if not fg_mask.any() or not bg_mask.any():
            return composite.copy()

        # ── Stage 1: LAB colour transfer (fg → background ring) ──────────
        adjusted = _lab_transfer(composite, fg_mask, bg_mask)

        # ── Stage 2: Poisson seamless cloning for boundary smoothness ────
        return _poisson_blend(adjusted, composite, fg_mask)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _lab_transfer(
    composite: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> np.ndarray:
    """Adaptively shift fg colour statistics toward the surrounding bg ring."""
    h, w = composite.shape[:2]
    lab = cv2.cvtColor(composite, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Sample the ring of background pixels just outside the object boundary
    fg_u8 = (fg_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_RING_RADIUS * 2 + 1, _RING_RADIUS * 2 + 1)
    )
    dilated = cv2.dilate(fg_u8, kernel)
    ring_mask = (dilated > 127) & bg_mask
    if not ring_mask.any():
        ring_mask = bg_mask

    adjusted = lab.copy()
    for c in range(3):
        fg_vals = lab[:, :, c][fg_mask]
        bg_vals = lab[:, :, c][ring_mask]
        mu_fg, sd_fg = fg_vals.mean(), fg_vals.std() + 1e-6
        mu_bg, sd_bg = bg_vals.mean(), bg_vals.std() + 1e-6

        transferred = (fg_vals - mu_fg) * (sd_bg / sd_fg) + mu_bg

        # Strength proportional to mean difference, capped at _MAX_STRENGTH
        diff = abs(mu_fg - mu_bg) / 128.0
        strength = float(np.clip(diff * 0.55, 0.10, _MAX_STRENGTH))

        adjusted[:, :, c][fg_mask] = fg_vals * (1.0 - strength) + transferred * strength

    adjusted_u8 = cv2.cvtColor(
        np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
    )
    return adjusted_u8


def _poisson_blend(
    source: np.ndarray,
    dest: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Blend *source* (adjusted fg) into *dest* (original composite) using Poisson.

    Falls back to a soft Gaussian boundary blend when the Poisson solver cannot
    converge (e.g. mask touches image edge, or region is very small).
    """
    ys, xs = np.where(fg_mask)
    h, w = dest.shape[:2]

    # Clamp centroid away from borders so seamlessClone has room to work
    margin = 6
    cx = int(np.clip(float(xs.mean()), margin, w - 1 - margin))
    cy = int(np.clip(float(ys.mean()), margin, h - 1 - margin))
    clone_mask = (fg_mask * 255).astype(np.uint8)

    try:
        result = cv2.seamlessClone(source, dest, clone_mask, (cx, cy), cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        # Poisson solver failed — use soft Gaussian boundary blend
        return _soft_blend(source, dest, fg_mask)


def _soft_blend(
    source: np.ndarray,
    dest: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Feathered blend: erode + Gaussian on the mask boundary."""
    fg_u8 = (fg_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(fg_u8, kernel)
    alpha = cv2.GaussianBlur(eroded, (9, 9), 3).astype(np.float32) / 255.0
    result = (source.astype(np.float32) * alpha[:, :, np.newaxis]
              + dest.astype(np.float32) * (1.0 - alpha[:, :, np.newaxis]))
    return result.clip(0, 255).astype(np.uint8)
