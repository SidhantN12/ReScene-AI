"""
ReScene AI — Inpainting wrapper (LaMa / OpenCV fallback).

Primary backend  : simple-lama-inpainting (pip-installable, auto-downloads
                   big-lama weights from HuggingFace ~200 MB on first use).
                     pip install simple-lama-inpainting

Fallback backend : OpenCV INPAINT_TELEA (zero-dependency; coherent for
                   small-medium masks, suitable for testing without weights).

Pipeline for every request
--------------------------
1. Dilate mask by dilation_px to hide seam artefacts.
2. Resize image+mask to a multiple of pad_modulo (LaMa requirement).
3. Run inference.
4. Crop result back to original size.
5. Poisson-blend the boundary strip so the fill merges cleanly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from config import Config

logger = logging.getLogger(__name__)

Backend = Literal["lama", "opencv"]


class InpaintingWrapper:
    """LaMa / OpenCV inpainting wrapper.

    Follows the standard ModelWrapper interface:
        load() / unload() / is_loaded() / predict()
    """

    def __init__(self, config: Config, backend: Backend = "lama") -> None:
        self._config = config
        self._backend: Backend = backend
        self._model: Any = None
        self._use_real: bool = False
        self._runtime_device: str = config.device.device
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the inpainting model.

        LaMa path  : attempts to import simple_lama_inpainting; if not
                     installed or the download fails, falls back to OpenCV.
        OpenCV path: no weights required; load() is a no-op sentinel.
        """
        device: str = self._config.device.device
        logger.info("Loading inpainting (%s) on %s", self._backend, device)

        if self._backend == "lama":
            try:
                from simple_lama_inpainting import SimpleLama   # type: ignore
                resolved_model = _resolve_big_lama_checkpoint(self._config)
                if resolved_model is not None:
                    os.environ["LAMA_MODEL"] = str(resolved_model)
                    self._model_path = str(resolved_model)
                    logger.info("Loading big-lama from %s", resolved_model)
                else:
                    logger.info("Downloading / loading big-lama weights (first run: ~200 MB)…")
                try:
                    self._model = SimpleLama(device=torch.device(device))
                    self._runtime_device = device
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc) or device != "cuda":
                        raise
                    logger.warning("big-lama CUDA load failed (%s) — retrying on CPU.", exc)
                    self._model = SimpleLama(device=torch.device("cpu"))
                    self._runtime_device = "cpu"
                self._use_real = True
                logger.info("big-lama loaded (real weights) on %s.", self._runtime_device)
                return
            except ImportError:
                logger.warning(
                    "simple-lama-inpainting not installed — falling back to OpenCV.\n"
                    "  pip install simple-lama-inpainting"
                )
            except Exception as exc:
                logger.warning("LaMa load failed (%s) — falling back to OpenCV.", exc)

        # OpenCV sentinel (also used when lama load fails)
        self._model = _OpenCVInpainter()
        self._backend = "opencv"
        self._use_real = False
        self._runtime_device = self._config.device.device
        logger.info("OpenCV inpainting active (INPAINT_TELEA).")

    def unload(self) -> None:
        self._model = None
        self._use_real = False
        self._runtime_device = self._config.device.device
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                logger.warning("Inpainting CUDA cleanup skipped: %s", exc)
        logger.info("Inpainting model unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        dilation_px: int = 8,
    ) -> np.ndarray:
        """Fill the masked region with synthesised background content.

        Parameters
        ----------
        image       : H×W×3 uint8 RGB
        mask        : H×W uint8 — 255 = region to fill, 0 = keep
        dilation_px : grow mask outward this many pixels before inpainting
                      (hides compressed-JPEG seam artefacts; use 0 to skip)

        Returns
        -------
        inpainted : H×W×3 uint8 RGB
        """
        if not self.is_loaded():
            raise RuntimeError("Inpainting not loaded. Call load() first.")

        h, w = image.shape[:2]

        # Validate mask shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1. Dilate mask
        if dilation_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
            )
            work_mask = cv2.dilate(mask, k)
        else:
            work_mask = mask.copy()

        # 2. Pad to pad_modulo
        pad = self._config.inference.lama_pad_modulo
        ph = ((h + pad - 1) // pad) * pad
        pw = ((w + pad - 1) // pad) * pad
        if ph != h or pw != w:
            padded_img = np.zeros((ph, pw, 3), dtype=np.uint8)
            padded_img[:h, :w] = image
            padded_msk = np.zeros((ph, pw), dtype=np.uint8)
            padded_msk[:h, :w] = work_mask
        else:
            padded_img = image
            padded_msk = work_mask

        # 3. Run inference
        if self._use_real:
            result_padded = self._run_lama(padded_img, padded_msk)
        else:
            result_padded = self._run_opencv(padded_img, padded_msk)

        # 4. Crop back
        result = result_padded[:h, :w]

        # 5. Poisson-blend the boundary strip so the fill merges cleanly
        result = _blend_boundary(image, result, work_mask[:h, :w])

        return result

    # ------------------------------------------------------------------
    # Internal inference paths
    # ------------------------------------------------------------------

    def _run_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run simple_lama_inpainting."""
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pil_msk = PILImage.fromarray(mask).convert("L")
        try:
            pil_result = self._model(pil_img, pil_msk)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or self._runtime_device != "cuda":
                raise
            logger.warning("big-lama CUDA inference failed (%s) — retrying on CPU.", exc)
            self._reload_lama_on_cpu()
            pil_result = self._model(pil_img, pil_msk)
        result = np.array(pil_result)
        if result.ndim == 2:
            result = np.stack([result] * 3, axis=-1)
        # simple_lama_inpainting may return RGBA
        if result.shape[2] == 4:
            result = result[:, :, :3]
        return result.astype(np.uint8)

    def _run_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """OpenCV INPAINT_TELEA — good structural coherence for most masks."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask_u8 = (mask > 127).astype(np.uint8) * 255
        radius = max(3, int(mask_u8.sum() ** 0.5 / 20))   # adaptive radius
        radius = min(radius, 21)
        result_bgr = cv2.inpaint(bgr, mask_u8, radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def _reload_lama_on_cpu(self) -> None:
        from simple_lama_inpainting import SimpleLama  # type: ignore

        if self._model_path:
            os.environ["LAMA_MODEL"] = self._model_path
        self._model = SimpleLama(device=torch.device("cpu"))
        self._runtime_device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# Boundary blending (Poisson seam fix)
# ---------------------------------------------------------------------------

def _blend_boundary(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    boundary_px: int = 6,
) -> np.ndarray:
    """Feather the boundary between the inpainted region and the original.

    A Gaussian-blurred erosion of the mask creates a soft alpha ramp so the
    edge of the fill never has a hard cut.  Falls back to alpha blend if
    cv2.seamlessClone raises an error (e.g. mask touches image border).
    """
    h, w = original.shape[:2]
    bin_mask = (mask > 127).astype(np.uint8) * 255

    # Try Poisson cloning for best quality
    ys, xs = np.where(bin_mask)
    if len(ys) == 0:
        return inpainted

    cx, cy = int(xs.mean()), int(ys.mean())
    # Poisson clone requires mask not to touch image borders
    if (ys.min() > 1 and ys.max() < h - 2 and
            xs.min() > 1 and xs.max() < w - 2):
        try:
            src_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
            dst_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, bin_mask, (cx, cy),
                                            cv2.NORMAL_CLONE)
            return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error:
            pass   # fall through to alpha blend

    # Feathered alpha blend fallback
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (boundary_px * 2 + 1, boundary_px * 2 + 1)
    )
    eroded = cv2.erode(bin_mask, kernel)
    alpha = cv2.GaussianBlur(
        eroded.astype(np.float32), (boundary_px * 4 + 1, boundary_px * 4 + 1), boundary_px
    ) / 255.0
    blended = (inpainted.astype(np.float32) * alpha[:, :, None]
               + original.astype(np.float32) * (1 - alpha[:, :, None]))
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sentinel for OpenCV path
# ---------------------------------------------------------------------------

class _OpenCVInpainter:
    """Sentinel object — actual inference done in InpaintingWrapper._run_opencv."""
    pass


def _resolve_big_lama_checkpoint(config: Config) -> Path | None:
    candidates = [
        config.models.lama_checkpoint,
        _torch_hub_big_lama_path(),
    ]
    env_path = os.environ.get("LAMA_MODEL")
    if env_path:
        candidates.insert(0, Path(env_path))

    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _torch_hub_big_lama_path() -> Path | None:
    hub_dir = Path(torch.hub.get_dir())
    candidate = hub_dir / "checkpoints" / "big-lama.pt"
    return candidate if candidate.exists() else None


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg
