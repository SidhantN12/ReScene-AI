"""
ReScene AI - Depth estimation wrapper (ZoeDepth-NK).

Uses the ZoeD_NK model (joint NYUv2 + KITTI training), which generalises
well to indoor scenes with mixed near/far depth ranges.

Outputs metric depth in metres (ZoeDepth is an absolute-depth model, not
relative), so scale factors are directly usable for furniture sizing.

VRAM: ~1.2 GB in fp32 / ~0.7 GB with autocast fp16.
Weights: stored at models/ZoeD_M12_NK.pt after running download_models.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage

from config import Config

logger = logging.getLogger(__name__)


class DepthEstimationWrapper:
    """ZoeDepth-NK monocular depth estimation wrapper.

    Public methods
    --------------
    load() / unload() / is_loaded() / predict()
    get_depth_at_point(depth_map, x, y)   → float  [metres]
    get_scale_factor(depth_a, depth_b)    → float  [ratio]
    scale_factor_at(depth_map, y, x)      → float  [ratio vs reference depth]
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None
        self._use_real: bool = False
        self._runtime_device: str = config.device.device

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ZoeDepth-NK from the local checkpoint with tolerant state loading.

        The upstream torch.hub pretrained=True path is brittle on newer
        torch/timm stacks because the checkpoint can contain non-persistent
        keys such as relative_position_index. Build the model without weights,
        then load the local checkpoint with strict=False instead.
        """
        device: str = self._config.device.device
        model_name: str = self._config.models.zoedepth_model_name
        checkpoint: Path = self._config.models.zoedepth_checkpoint
        logger.info("Loading ZoeDepth (%s) from %s on %s", model_name, checkpoint, device)

        try:
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"ZoeDepth checkpoint not found: {checkpoint}\n"
                    "  Run: python download_models.py"
                )

            with torch.no_grad():
                model = torch.hub.load(
                    "isl-org/ZoeDepth",
                    model_name,
                    pretrained=False,
                    verbose=False,
                )

            state_dict = _load_zoedepth_state_dict(checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                logger.warning(
                    "ZoeDepth loaded with ignored checkpoint keys: %s",
                    ", ".join(unexpected_keys[:8]) + (" ..." if len(unexpected_keys) > 8 else ""),
                )
            if missing_keys:
                logger.warning(
                    "ZoeDepth checkpoint missing model keys: %s",
                    ", ".join(missing_keys[:8]) + (" ..." if len(missing_keys) > 8 else ""),
                )

            _disable_drop_path_if_present(model)
            try:
                model.to(device)
                self._runtime_device = device
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or device != "cuda":
                    raise
                logger.warning("ZoeDepth CUDA load failed (%s) — retrying on CPU.", exc)
                model.to("cpu")
                self._runtime_device = "cpu"
            model.eval()
            self._model = model
            self._use_real = True
            logger.info("ZoeDepth (%s) loaded (real weights).", model_name)
        except Exception as exc:
            logger.warning(
                "ZoeDepth real load failed (%s) — falling back to placeholder.", exc
            )
            self._model = _DummyZoeDepth()
            self._use_real = False

    def unload(self) -> None:
        self._model = None
        self._use_real = False
        self._runtime_device = self._config.device.device
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                logger.warning("ZoeDepth CUDA cleanup skipped: %s", exc)
        logger.info("ZoeDepth unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate per-pixel metric depth.

        Parameters
        ----------
        image : H×W×3 uint8 RGB numpy array

        Returns
        -------
        depth : (H, W) float32 — metric depth in metres (near=small, far=large)
        """
        if not self.is_loaded():
            raise RuntimeError("ZoeDepth not loaded. Call load() first.")

        h, w = image.shape[:2]

        if self._use_real:
            pil_img = PILImage.fromarray(image)
            try:
                depth = self._infer_real(pil_img)
            except RuntimeError as exc:
                if not (_is_nvrtc_error(exc) or _is_cuda_oom(exc)) or self._runtime_device == "cpu":
                    raise
                logger.warning(
                    "ZoeDepth CUDA inference failed (%s) - retrying on CPU.",
                    exc,
                )
                self._model.to("cpu")
                self._runtime_device = "cpu"
                depth = self._infer_real(pil_img)

            # ZoeDepth returns float32 numpy H×W (metric metres)
            if isinstance(depth, PILImage.Image):
                depth = np.array(depth, dtype=np.float32)
            else:
                depth = np.array(depth, dtype=np.float32)

            # Ensure output matches input spatial resolution
            if depth.shape != (h, w):
                import cv2
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

            return depth

        # Placeholder: linear gradient (top=5 m, bottom=1 m)
        depth = np.linspace(5.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        return np.broadcast_to(depth, (h, w)).copy()

    # ------------------------------------------------------------------
    # Utility methods (all operate on already-computed depth maps —
    # no GPU required after predict() returns)
    # ------------------------------------------------------------------

    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Return the metric depth (metres) at pixel (x, y).

        Clamps (x, y) to valid image bounds.
        """
        h, w = depth_map.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        return float(depth_map[y, x])

    def get_scale_factor(self, depth_a: float, depth_b: float) -> float:
        """Compute the size scale factor when moving an object from depth_a to depth_b.

        An object at depth_a re-placed at depth_b should be scaled by depth_a/depth_b.
        For example: moving from 4 m → 2 m doubles apparent size (scale = 2.0).

        Parameters
        ----------
        depth_a : source depth in metres
        depth_b : target depth in metres

        Returns
        -------
        scale : float — multiply object pixel dimensions by this value
        """
        depth_b = max(depth_b, 1e-3)
        depth_a = max(depth_a, 1e-3)
        return depth_a / depth_b

    def scale_factor_at(self, depth_map: np.ndarray, y: int, x: int) -> float:
        """Return the scale factor for placing an object at pixel (x, y).

        Scale is relative to config.inference.depth_reference_m.
        Closer pixels (smaller depth) → scale > 1 (object appears larger).
        """
        ref = self._config.inference.depth_reference_m
        d = self.get_depth_at_point(depth_map, x, y)
        return ref / max(d, 0.01)

    def depth_stats(self, depth_map: np.ndarray) -> dict[str, float]:
        """Return basic statistics of a depth map (for logging / UI display)."""
        return {
            "min_m": float(depth_map.min()),
            "max_m": float(depth_map.max()),
            "mean_m": float(depth_map.mean()),
            "median_m": float(np.median(depth_map)),
        }

    def _infer_real(self, pil_img: PILImage.Image) -> np.ndarray:
        device = self._runtime_device
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device == "cuda"
            else torch.no_grad()
        )
        with torch.no_grad(), autocast_ctx:
            depth = self._model.infer_pil(pil_img)
        return np.array(depth, dtype=np.float32)


class _DummyZoeDepth:
    pass


def _is_nvrtc_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return "nvrtc" in msg.lower() or "nvrtc-builtins64_121.dll" in msg.lower()


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _load_zoedepth_state_dict(checkpoint: Path) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model", state)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected ZoeDepth checkpoint format in {checkpoint}")

    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        if key.endswith("relative_position_index"):
            continue
        cleaned[key] = value
    return cleaned


def _disable_drop_path_if_present(model: Any) -> None:
    try:
        blocks = model.core.core.pretrained.model.blocks
    except AttributeError:
        return

    for block in blocks:
        if hasattr(block, "drop_path"):
            block.drop_path = torch.nn.Identity()
            continue

        # Newer timm builds expose split drop paths while MiDaS' patched BEiT
        # forward still expects a single `drop_path` attribute.
        if hasattr(block, "drop_path1"):
            block.drop_path = block.drop_path1
        else:
            block.drop_path = torch.nn.Identity()
