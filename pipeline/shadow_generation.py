"""
ReScene AI — Contact-shadow generation wrapper (ARShadowGAN).

Real implementation will:
  - Take the composited RGB image + the binary object mask
  - ARShadowGAN encodes: (scene_RGB, object_mask) → shadow_mask
  - The shadow_mask is then used to darken the scene beneath the object
  - Additionally a soft Gaussian penumbra is added around the hard shadow
  - Light direction can be estimated from the depth map or provided explicitly

Placeholder generates a plausible soft drop-shadow by:
  1. Estimating the dominant light direction from scene brightness.
  2. Adding a tight contact shadow right at the object base.
  3. Adding a cast shadow offset in the estimated light direction, with
     a larger Gaussian penumbra for the softer outer falloff.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from config import Config

logger = logging.getLogger(__name__)


class ShadowGenerationWrapper:
    """ARShadowGAN contact-shadow generation wrapper."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None

    def load(self) -> None:
        device: str = self._config.device.device
        logger.info("Loading ARShadowGAN on %s", device)

        # --- real implementation ---
        # from arshadowgan.networks import define_G
        # self._model = define_G(input_nc=4, output_nc=1, ngf=64, netG="unet_256")
        # state = torch.load(self._config.models.arshadowgan_checkpoint, map_location=device)
        # self._model.load_state_dict(state)
        # self._model.to(device).eval()

        self._model = _DummyARShadowGAN()
        logger.info("ARShadowGAN loaded (placeholder with light estimation).")

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ARShadowGAN unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
        light_direction: tuple[float, float] = (0.3, 0.8),
        shadow_opacity: float = 0.55,
    ) -> np.ndarray:
        """Render contact shadow for *object_mask* into *image*.

        Parameters
        ----------
        image         : H×W×3 uint8 RGB — scene with object already composited
        object_mask   : H×W uint8 — 255 where the placed object is
        light_direction : (dx, dy) hint; actual direction estimated from scene
        shadow_opacity  : 0–1, how dark the combined shadow is

        Returns
        -------
        shadowed : H×W×3 uint8 RGB — scene with shadow applied under object
        """
        if not self.is_loaded():
            raise RuntimeError("ARShadowGAN not loaded. Call load() first.")

        h, w = image.shape[:2]

        # Estimate light direction from scene brightness
        est = _estimate_light_direction(image, object_mask)
        dx = int(est[0] * h * 0.10)
        dy = int(est[1] * h * 0.10)

        # ── 1. Contact shadow ─────────────────────────────────────────────
        # Tight blur right at the object boundary — stays close to the base.
        cr = max(7, h // 60) | 1   # odd kernel radius
        contact_blurred = cv2.GaussianBlur(object_mask, (cr, cr), cr // 4)
        contact_soft = np.where(object_mask > 127, 0, contact_blurred).astype(
            np.float32
        ) / 255.0

        # ── 2. Cast shadow ────────────────────────────────────────────────
        # Stretch mask slightly in cast direction then offset it.
        stretch = 1.0 + abs(dx) / max(w, 1) * 1.5
        # Centre the stretch so the base stays roughly in place
        M_str = np.float32([[stretch, 0, -w * (stretch - 1) * 0.5], [0, 1.0, 0]])
        mask_stretched = cv2.warpAffine(object_mask, M_str, (w, h))

        M_cast = np.float32([[1, 0, dx], [0, 1, dy]])
        shadow_raw = cv2.warpAffine(mask_stretched, M_cast, (w, h))
        shadow_raw = np.where(object_mask > 127, 0, shadow_raw).astype(np.uint8)

        pen = max(21, (abs(dx) + abs(dy)) * 2 + 5) | 1   # odd
        cast_soft = cv2.GaussianBlur(shadow_raw, (pen, pen), pen // 4).astype(
            np.float32
        ) / 255.0

        # ── 3. Combine and apply ──────────────────────────────────────────
        # Contact shadow is 75 % weight (dark + tight), cast is 55 % (wider).
        combined = np.clip(contact_soft * 0.75 + cast_soft * 0.55, 0.0, 1.0)

        result = image.astype(np.float32)
        result *= (1.0 - combined * shadow_opacity)[:, :, np.newaxis]
        return result.clip(0, 255).astype(np.uint8)


class _DummyARShadowGAN:
    pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _estimate_light_direction(
    image: np.ndarray,
    object_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Infer shadow-cast direction from the scene's brightness distribution.

    Finds the centroid of the brightest 20 % of pixels (the illuminated zone)
    and returns the unit vector pointing from that centroid toward the object.
    Shadows always have a downward component so they fall on the floor.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Heavily blur to find the broad illumination zone, not individual highlights
    kw = max(w // 8, 3) | 1
    kh = max(h // 8, 3) | 1
    blurred = cv2.GaussianBlur(gray, (kw, kh), 0)

    thresh = float(np.percentile(blurred, 80))
    ys, xs = np.where(blurred > thresh)
    if len(ys) == 0:
        return (0.2, 1.0)

    light_cx = float(xs.mean()) / w
    light_cy = float(ys.mean()) / h

    # Object centroid
    obj_cx = obj_cy = 0.5
    if object_mask is not None and object_mask.max() > 0:
        oys, oxs = np.where(object_mask > 127)
        if len(oxs) > 0:
            obj_cx = float(oxs.mean()) / w
            obj_cy = float(oys.mean()) / h

    # Shadow goes FROM light centroid TOWARD (and past) the object
    dx = obj_cx - light_cx
    dy = max(obj_cy - light_cy, 0.15)   # floor shadows always go downward
    norm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-6
    return (float(dx / norm) * 0.6, float(dy / norm))
