"""
ReScene AI — VRAM-aware sequential model manager.

Only one heavyweight model occupies GPU memory at a time.  Before loading a
model the manager checks free VRAM; if headroom is insufficient it unloads the
current resident first, then garbage-collects and empties the CUDA cache.

Usage
-----
    mm = ModelManager(config)
    mm.load("sam")
    result = mm.get("sam").predict(...)
    mm.unload_current()          # or mm.load("lama") — implicitly unloads sam

Thread safety: not re-entrant.  All pipeline calls should be sequential.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Optional

import torch

from config import Config, MODEL_VRAM_GB

logger = logging.getLogger(__name__)


def _safe_cuda_cleanup() -> None:
    if not torch.cuda.is_available():
        return
    for fn in (torch.cuda.empty_cache, torch.cuda.synchronize):
        try:
            fn()
        except RuntimeError as exc:
            logger.warning("CUDA cleanup skipped after device error: %s", exc)
            break


class ModelManager:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._registry: dict[str, object] = {}   # name → wrapper instance
        self._current: Optional[str] = None       # name of model on GPU right now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, wrapper: object) -> None:
        """Register a ModelWrapper instance under *name*."""
        self._registry[name] = wrapper
        logger.debug("Registered model wrapper: %s", name)

    def load(self, name: str) -> None:
        """Ensure *name* is loaded on GPU, unloading current resident first."""
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not registered with ModelManager.")

        if self._current == name and self._registry[name].is_loaded():
            logger.debug("'%s' already loaded — skipping.", name)
            return

        # Unload whatever is on GPU now
        if self._current is not None:
            self._unload(self._current)

        # VRAM headroom check
        self._assert_vram_headroom(name)

        logger.info("Loading model: %s", name)
        t0 = time.perf_counter()
        self._registry[name].load()
        elapsed = time.perf_counter() - t0
        logger.info("'%s' loaded in %.2fs  |  VRAM used: %.2f GB", name, elapsed, self._vram_used_gb())
        self._current = name

    def get(self, name: str) -> object:
        """Return the wrapper for *name*, loading it if necessary."""
        if self._current != name or not self._registry[name].is_loaded():
            self.load(name)
        return self._registry[name]

    def unload_current(self) -> None:
        if self._current is not None:
            self._unload(self._current)
            self._current = None

    def unload_all(self) -> None:
        for name, wrapper in self._registry.items():
            if wrapper.is_loaded():
                self._unload(name)
        self._current = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unload(self, name: str) -> None:
        wrapper = self._registry.get(name)
        if wrapper is None or not wrapper.is_loaded():
            return
        logger.info("Unloading model: %s", name)
        try:
            wrapper.unload()
        except RuntimeError as exc:
            logger.warning("Model '%s' unload raised after runtime failure: %s", name, exc)
        gc.collect()
        _safe_cuda_cleanup()
        logger.debug("VRAM after unload: %.2f GB used", self._vram_used_gb())
        if self._current == name:
            self._current = None

    def _vram_used_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 ** 3

    def _vram_free_gb(self) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated()
        return (total - used) / 1024 ** 3

    def _assert_vram_headroom(self, name: str) -> None:
        needed = MODEL_VRAM_GB.get(name, 1.0)
        free = self._vram_free_gb()
        limit = self._config.device.vram_limit_gb
        if needed > limit:
            raise RuntimeError(
                f"Model '{name}' needs ~{needed:.1f} GB but VRAM limit is {limit:.1f} GB."
            )
        if needed > free:
            logger.warning(
                "'%s' needs %.1f GB but only %.1f GB free — attempting to free memory.",
                name, needed, free,
            )
            gc.collect()
            _safe_cuda_cleanup()
            free = self._vram_free_gb()
            if needed > free:
                raise RuntimeError(
                    f"Insufficient VRAM for '{name}': need {needed:.1f} GB, have {free:.1f} GB."
                )

    # ------------------------------------------------------------------
    # Context manager — auto-unload on exit
    # ------------------------------------------------------------------

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *_) -> None:
        self.unload_all()
