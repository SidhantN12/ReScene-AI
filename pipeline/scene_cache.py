"""
ReScene AI — Scene-context LRU cache (Phase 8).

Caches up to MAX_ENTRIES SceneContext objects keyed by the image's sha256 hash.
Avoids re-running SAM + ZoeDepth + CLIP when the same image is uploaded again
(e.g. switching tabs, or re-running the compound pipeline on the same photo).

Usage
-----
    cache = SceneCache()
    ctx = cache.get_or_build(image, builder)

Thread safety: not guaranteed — Gradio runs single-threaded by default.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.scene_context import SceneContext, SceneContextBuilder

logger = logging.getLogger(__name__)

MAX_ENTRIES: int = 4


class SceneCache:
    """LRU cache for SceneContext objects."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, "SceneContext"] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_build(
        self,
        image: np.ndarray,
        builder: "SceneContextBuilder",
    ) -> "SceneContext":
        """Return a cached SceneContext, or build (and cache) a new one."""
        key = _sha256(image)

        if key in self._store:
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            logger.debug("[SCENE CACHE] hit  key=%s…", key[:8])
            return self._store[key]

        logger.debug("[SCENE CACHE] miss key=%s… — building context.", key[:8])
        ctx = builder.build(image)
        self._put(key, ctx)
        return ctx

    def get(self, image: np.ndarray) -> "SceneContext | None":
        """Return a cached context without building, or None on miss."""
        key = _sha256(image)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def invalidate(self, image: np.ndarray) -> None:
        """Remove the cached entry for *image*, if present."""
        key = _sha256(image)
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _put(self, key: str, ctx: "SceneContext") -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = ctx
        while len(self._store) > self._max:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("[SCENE CACHE] evicted key=%s…", evicted_key[:8])


# ---------------------------------------------------------------------------
# Module-level singleton (imported by app.py and orchestrator)
# ---------------------------------------------------------------------------

_scene_cache: SceneCache | None = None


def get_scene_cache() -> SceneCache:
    """Return the module-level SceneCache singleton, creating it if needed."""
    global _scene_cache
    if _scene_cache is None:
        _scene_cache = SceneCache()
    return _scene_cache


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sha256(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()
