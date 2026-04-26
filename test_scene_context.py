"""
ReScene AI — Phase 8: SceneContext unit tests.

Runs without any ML model weights: all heavy models (SAM, ZoeDepth, CLIP) are
absent so the builder uses heuristic fallbacks.  Tests verify that the returned
SceneContext always has the mandatory structural fields populated.

Usage
-----
    conda activate torch-cu121
    python test_scene_context.py [path/to/room1.jpg [room2.jpg ...]]

If no paths are given the script generates three synthetic room images.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_scene_context")

from pipeline.scene_context import SceneContext, SceneContextBuilder
from pipeline.scene_cache import SceneCache, get_scene_cache


# ---------------------------------------------------------------------------
# Synthetic test images (no external files required)
# ---------------------------------------------------------------------------

def _make_synthetic_room(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Simple geometric room: floor, wall, window, sofa, plant."""
    import cv2

    rng = np.random.default_rng(seed)
    canvas = np.full((h, w, 3), [240, 235, 228], dtype=np.uint8)

    # Floor
    floor_pts = np.array([
        [0, h], [w, h],
        [int(w * 0.85), int(h * 0.55)],
        [int(w * 0.15), int(h * 0.55)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [floor_pts], [200, 185, 165])

    # Wall
    wall_pts = np.array([
        [int(w * 0.15), int(h * 0.55)], [int(w * 0.85), int(h * 0.55)],
        [int(w * 0.85), 0],             [int(w * 0.15), 0],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [wall_pts], [230, 225, 218])

    # Window (strong edge lines — good for VP detection)
    wx0, wy0 = int(w * 0.38), int(h * 0.08)
    wx1, wy1 = int(w * 0.62), int(h * 0.42)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), [180, 210, 235], -1)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), [140, 120, 100], 5)

    # Sofa
    cv2.rectangle(canvas, (int(w * 0.05), int(h * 0.62)),
                  (int(w * 0.55), int(h * 0.88)), [110, 85, 75], -1)

    return canvas


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert(condition: bool, msg: str) -> None:
    if not condition:
        logger.error("FAIL: %s", msg)
        raise AssertionError(msg)
    logger.info("PASS: %s", msg)


def _check_context(ctx: SceneContext, label: str) -> None:
    logger.info("── Checking context for: %s ──", label)

    _assert(ctx.image is not None and ctx.image.ndim == 3, "image is 3-D array")
    _assert(len(ctx.image_hash) == 64, "image_hash is 64-char hex")

    # floor_mask must be non-empty (either from SAM or heuristic)
    _assert(ctx.floor_mask is not None, "floor_mask is not None")
    _assert(ctx.floor_mask.dtype == bool, "floor_mask is bool array")
    floor_px = int(ctx.floor_mask.sum())
    _assert(floor_px > 0, f"floor_mask has {floor_px} > 0 True pixels")

    # At least one wall mask
    _assert(len(ctx.wall_masks) >= 1, f"wall_masks has {len(ctx.wall_masks)} ≥ 1 entry")
    wall_total = sum(m.sum() for m in ctx.wall_masks)
    _assert(wall_total > 0, f"wall_masks total pixels {wall_total} > 0")

    # anchor_candidates non-empty
    _assert(len(ctx.anchor_candidates) > 0,
            f"anchor_candidates has {len(ctx.anchor_candidates)} > 0 entries")
    for a in ctx.anchor_candidates:
        _assert("x" in a and "y" in a, "anchor has x, y keys")

    # free_floor_mask shape matches image
    h, w = ctx.image.shape[:2]
    if ctx.free_floor_mask is not None:
        _assert(ctx.free_floor_mask.shape == (h, w),
                f"free_floor_mask shape {ctx.free_floor_mask.shape} == ({h},{w})")

    # metadata populated
    _assert(isinstance(ctx.metadata, dict), "metadata is dict")

    logger.info("   floor_px=%d  walls=%d  anchors=%d  vps=%d",
                floor_px, len(ctx.wall_masks),
                len(ctx.anchor_candidates), len(ctx.vanishing_points))


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

def _check_cache(image: np.ndarray) -> None:
    logger.info("── Cache tests ──")
    builder = SceneContextBuilder(model_manager=None, config=None)
    cache = SceneCache(max_entries=2)

    ctx1 = cache.get_or_build(image, builder)
    ctx2 = cache.get_or_build(image, builder)
    _assert(ctx1 is ctx2, "Cache returns same instance for same image (identity check)")
    _assert(cache.size == 1, f"Cache size is 1 after two hits on same image, got {cache.size}")

    # LRU eviction: add two more different images
    img_b = np.zeros_like(image)
    img_b[:, :, 0] = 200
    img_c = np.zeros_like(image)
    img_c[:, :, 1] = 200

    cache.get_or_build(img_b, builder)
    _assert(cache.size == 2, f"Cache size 2 after 2 distinct images, got {cache.size}")

    cache.get_or_build(img_c, builder)
    _assert(cache.size == 2, f"Cache size stays ≤ 2 after LRU eviction, got {cache.size}")

    # Original image evicted; re-building should not crash
    ctx3 = cache.get_or_build(image, builder)
    _assert(ctx3 is not ctx1, "Cache rebuilt after eviction — different object")
    logger.info("PASS: cache LRU eviction and rebuild")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(image_paths: list[Path]) -> None:
    builder = SceneContextBuilder(model_manager=None, config=None)

    images: list[tuple[str, np.ndarray]] = []

    if image_paths:
        from utils.image_utils import load_image, resize_long_edge
        for p in image_paths:
            if not p.exists():
                logger.warning("Image not found: %s — skipping.", p)
                continue
            img = load_image(p)
            img = resize_long_edge(img, 1024)
            images.append((p.name, img))
    else:
        logger.info("No image paths given — using 3 synthetic rooms.")
        for seed in range(3):
            img = _make_synthetic_room(seed=seed)
            images.append((f"synthetic_room_{seed}", img))

    if not images:
        logger.error("No images to test.")
        sys.exit(1)

    failures: list[str] = []
    for label, image in images:
        try:
            ctx = builder.build(image)
            _check_context(ctx, label)
        except AssertionError as e:
            failures.append(f"{label}: {e}")

    # Cache test on first image
    _, first_image = images[0]
    try:
        _check_cache(first_image)
    except AssertionError as e:
        failures.append(f"cache: {e}")

    logger.info("─" * 60)
    if failures:
        logger.error("%d / %d test(s) FAILED:", len(failures), len(images) + 1)
        for f in failures:
            logger.error("  %s", f)
        sys.exit(1)
    else:
        logger.info("All %d test(s) PASSED.", len(images) + 1)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    run(paths)
