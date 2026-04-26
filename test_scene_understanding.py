"""
ReScene AI — Phase 2 test script.

Takes a room photograph and outputs:
  outputs/seg_overlay.png   — all SAM masks drawn over the image
  outputs/depth_map.png     — ZoeDepth heatmap with min/max annotations
  outputs/seg_masks/        — each individual mask as a separate PNG

Usage
-----
    # With a real image:
    python test_scene_understanding.py path/to/room.jpg

    # Auto-generates a synthetic test image if no argument given:
    python test_scene_understanding.py

Both real models and placeholder mode work — the script detects which is
available and logs accordingly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_scene_understanding")

from config import config, OUTPUTS_DIR
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.segmentation import draw_segmentation_overlay
from utils.depth_utils import depth_colourmap, normalise_depth
from utils.image_utils import load_image, save_image, resize_long_edge


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run(image_path: Path | None) -> None:
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "seg_masks"
    masks_dir.mkdir(exist_ok=True)

    # ── Load or generate test image ──────────────────────────────────────
    if image_path is not None and image_path.exists():
        logger.info("Loading image: %s", image_path)
        image = load_image(image_path)
    else:
        if image_path is not None:
            logger.warning("Image not found: %s — generating synthetic test image.", image_path)
        else:
            logger.info("No image path provided — generating synthetic test image.")
        image = _make_synthetic_room(720, 1280)
        synth_path = out_dir / "synthetic_room.png"
        save_image(image, synth_path)
        logger.info("Synthetic image saved: %s", synth_path)

    # Cap to 1024 px long edge (matches pipeline config)
    image = resize_long_edge(image, config.inference.max_image_size)
    logger.info("Image shape after resize: %s", image.shape)

    # ── Run scene understanding ───────────────────────────────────────────
    orch = PipelineOrchestrator(config)
    t_total = time.perf_counter()

    logger.info("─" * 60)
    logger.info("STEP 1 / 2 — Segmentation (SAM ViT-H)")
    logger.info("─" * 60)
    t0 = time.perf_counter()
    scene = orch.understand_scene(image)
    seg_elapsed = time.perf_counter() - t0

    seg_result = scene["segmentation"]
    depth_map = scene["depth_map"]
    depth_stats = scene["depth_stats"]

    n_masks = len(seg_result["masks"])
    logger.info("Segmentation done in %.2f s — %d objects detected.", seg_elapsed, n_masks)
    for i, (label, score, box) in enumerate(
        zip(seg_result["labels"], seg_result["scores"], seg_result["boxes"])
    ):
        area_pct = seg_result["masks"][i].astype(bool).sum() / image[:,:,0].size * 100
        logger.info("  [%s]  IoU=%.3f  bbox=%s  area=%.1f%%", label, score, box, area_pct)

    logger.info("─" * 60)
    logger.info("STEP 2 / 2 — Depth statistics")
    logger.info("─" * 60)
    logger.info(
        "  min=%.2f m  max=%.2f m  mean=%.2f m  median=%.2f m",
        depth_stats["min_m"], depth_stats["max_m"],
        depth_stats["mean_m"], depth_stats["median_m"],
    )

    total_elapsed = time.perf_counter() - t_total
    logger.info("Total wall time: %.2f s", total_elapsed)

    # ── Save outputs ──────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("SAVING OUTPUTS → %s", out_dir)
    logger.info("─" * 60)

    # 1. Segmentation overlay
    seg_overlay = draw_segmentation_overlay(image, seg_result, alpha=0.45)
    seg_path = out_dir / "seg_overlay.png"
    save_image(seg_overlay, seg_path)
    logger.info("Saved: %s", seg_path)

    # 2. Depth heatmap
    depth_vis = _render_depth_vis(image, depth_map, depth_stats)
    depth_path = out_dir / "depth_map.png"
    save_image(depth_vis, depth_path)
    logger.info("Saved: %s", depth_path)

    # 3. Individual masks
    for label, mask in seg_result["label_map"].items():
        mask_path = masks_dir / f"{label}.png"
        cv2.imwrite(str(mask_path), mask)
    logger.info("Saved %d individual masks → %s", n_masks, masks_dir)

    # 4. Side-by-side summary PNG
    summary = _make_summary(image, seg_overlay, depth_vis)
    summary_path = out_dir / "summary.png"
    save_image(summary, summary_path)
    logger.info("Saved summary: %s", summary_path)

    logger.info("─" * 60)
    logger.info("Done.  Check %s", out_dir)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _render_depth_vis(
    image: np.ndarray,
    depth_map: np.ndarray,
    stats: dict[str, float],
) -> np.ndarray:
    """Render depth as TURBO heatmap with annotation bar."""
    h, w = image.shape[:2]

    # Main heatmap
    heat = depth_colourmap(depth_map)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    # Annotation strip at top
    canvas = cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    bar_h = 36
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"ZoeDepth-NK  |  near {stats['min_m']:.2f} m   "
        f"mean {stats['mean_m']:.2f} m   "
        f"far {stats['max_m']:.2f} m",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    # Gradient colour bar at bottom
    bar = _colourbar(w, 20)
    canvas[-20:] = bar

    # Near / Far labels on bar
    cv2.putText(canvas, f"Near {stats['min_m']:.1f}m", (4, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Far {stats['max_m']:.1f}m", (w - 100, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _colourbar(width: int, height: int) -> np.ndarray:
    """Return a width×height BGR gradient matching TURBO colourmap."""
    gradient = np.tile(np.arange(256, dtype=np.uint8), (height, 1))
    gradient = cv2.resize(gradient, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)


def _make_summary(
    original: np.ndarray,
    seg_overlay: np.ndarray,
    depth_vis: np.ndarray,
) -> np.ndarray:
    """Three-panel side-by-side: original | segmentation | depth."""
    h = original.shape[0]

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (240, 240, 240), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    panels = [
        _label(original, "Original"),
        _label(seg_overlay, f"Segmentation ({len(np.unique(np.zeros(1)))} objects)"),
        _label(depth_vis, "Depth (ZoeDepth-NK)"),
    ]

    # Uniform height
    target_h = h
    resized = []
    for p in panels:
        ph, pw = p.shape[:2]
        new_w = int(pw * target_h / ph)
        resized.append(cv2.resize(p, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4))

    # 4-px white divider
    div = np.full((target_h, 4, 3), 240, dtype=np.uint8)
    combined = np.concatenate([resized[0], div, resized[1], div, resized[2]], axis=1)
    return combined


# ---------------------------------------------------------------------------
# Synthetic room image for testing without a real photo
# ---------------------------------------------------------------------------

def _make_synthetic_room(h: int = 720, w: int = 1280) -> np.ndarray:
    """Generate a simple geometric room scene with distinct depth layers."""
    canvas = np.ones((h, w, 3), dtype=np.uint8)

    # Sky / upper wall — warm white
    canvas[:] = [240, 235, 228]

    # Floor — perspective trapezoid
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.85), int(h * 0.55)],
                           [int(w * 0.15), int(h * 0.55)]], dtype=np.int32)
    cv2.fillPoly(canvas, [floor_pts], [200, 185, 165])

    # Back wall
    wall_pts = np.array([[int(w * 0.15), int(h * 0.55)], [int(w * 0.85), int(h * 0.55)],
                          [int(w * 0.85), 0], [int(w * 0.15), 0]], dtype=np.int32)
    cv2.fillPoly(canvas, [wall_pts], [230, 225, 218])

    # Window
    wx0, wy0 = int(w * 0.38), int(h * 0.08)
    wx1, wy1 = int(w * 0.62), int(h * 0.42)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), [180, 210, 235], -1)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), [160, 140, 120], 8)
    cv2.line(canvas, ((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1), [160, 140, 120], 4)
    cv2.line(canvas, (wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2), [160, 140, 120], 4)

    # Sofa (large near object)
    sx0, sy0 = int(w * 0.05), int(h * 0.60)
    sx1, sy1 = int(w * 0.55), int(h * 0.88)
    cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), [120, 90, 80], -1)
    cv2.rectangle(canvas, (sx0, sy0), (sx1, int(sy0 + (sy1 - sy0) * 0.35)), [100, 70, 65], -1)
    # Sofa legs
    leg_w, leg_h = 12, 22
    for lx in [sx0 + 20, sx1 - 30]:
        cv2.rectangle(canvas, (lx, sy1), (lx + leg_w, sy1 + leg_h), [80, 60, 50], -1)

    # Side table
    tx0, ty0 = int(w * 0.58), int(h * 0.65)
    tx1, ty1 = int(w * 0.68), int(h * 0.75)
    cv2.rectangle(canvas, (tx0, ty0), (tx1, ty1), [160, 130, 100], -1)
    cv2.rectangle(canvas, (tx0 + 4, ty1), (tx0 + 12, ty1 + 25), [140, 110, 80], -1)
    cv2.rectangle(canvas, (tx1 - 12, ty1), (tx1 - 4, ty1 + 25), [140, 110, 80], -1)

    # Plant
    px, py = int(w * 0.80), int(h * 0.58)
    cv2.ellipse(canvas, (px, py - 40), (35, 50), 0, 0, 360, [60, 120, 60], -1)
    cv2.ellipse(canvas, (px - 25, py - 20), (22, 35), -20, 0, 360, [50, 100, 50], -1)
    cv2.rectangle(canvas, (px - 18, py), (px + 18, py + 45), [120, 80, 60], -1)

    # Picture frame on wall
    ff = [int(w * 0.17), int(h * 0.12), int(w * 0.30), int(h * 0.38)]
    cv2.rectangle(canvas, (ff[0], ff[1]), (ff[2], ff[3]), [80, 70, 60], 6)
    cv2.rectangle(canvas, (ff[0] + 6, ff[1] + 6), (ff[2] - 6, ff[3] - 6),
                  [140, 160, 180], -1)

    # Floor shadow under sofa
    shadow = canvas[sy1:sy1 + 18, sx0:sx1].copy().astype(np.float32)
    canvas[sy1:sy1 + 18, sx0:sx1] = (shadow * 0.75).clip(0, 255).astype(np.uint8)

    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReScene AI — Phase 2 scene understanding test"
    )
    p.add_argument(
        "image",
        nargs="?",
        default=None,
        type=Path,
        help="Path to a room JPEG/PNG (omit to use a synthetic test image)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.image)
