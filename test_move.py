"""
ReScene AI — Phase 4 test script: Move operation.

Generates 4 distinct synthetic room images, provides pre-computed object masks
and target positions, runs the full move pipeline (SAM skipped — masks given
directly), and saves:

    outputs/move_test_N_before.png
    outputs/move_test_N_after.png
    outputs/move_test_N_comparison.png   ← 3-panel: Before | Source mask | After
    outputs/move_summary.png             ← all 4 tests tiled vertically

Pipeline steps exercised
------------------------
  depth estimation (ZoeDepth placeholder → linear gradient)
  → inpainting (LaMa or OpenCV fallback)
  → perspective warp (depth-aware trapezoid + colour transfer)
  → colour-match composite (Reinhard per-channel transfer)
  → shadow placeholder
  → harmonisation placeholder

Usage
-----
    conda activate torch-cu121
    python test_move.py
    python test_move.py --backend opencv   # force OpenCV inpainter
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_move")

from config import config, OUTPUTS_DIR
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.inpainting import InpaintingWrapper
from utils.image_utils import save_image


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class MoveTestCase:
    name: str
    image: np.ndarray           # H×W×3 uint8 RGB
    source_mask: np.ndarray     # H×W uint8 (255 = object to move)
    target_position: tuple[int, int]   # (x, y) destination in image coords
    description: str


# ---------------------------------------------------------------------------
# Synthetic room generators
# (each returns: image, source_mask, target_position)
# ---------------------------------------------------------------------------

def _make_living_room_sofa(h: int = 600, w: int = 800,
                            ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Living room — move sofa from left side to right side."""
    img = np.full((h, w, 3), [235, 228, 218], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.85), int(h * 0.55)],
                           [int(w * 0.15), int(h * 0.55)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [195, 178, 155])

    # Back wall
    wall_pts = np.array([[int(w * 0.15), int(h * 0.55)], [int(w * 0.85), int(h * 0.55)],
                          [w, 0], [0, 0]], np.int32)
    cv2.fillPoly(img, [wall_pts], [228, 222, 212])

    # Window on wall
    wx, wy = int(w * 0.60), int(h * 0.15)
    cv2.rectangle(img, (wx - 55, wy), (wx + 55, wy + 80), [180, 205, 230], -1)
    cv2.rectangle(img, (wx - 55, wy), (wx + 55, wy + 80), [160, 148, 132], 3)
    cv2.line(img, (wx, wy), (wx, wy + 80), [160, 148, 132], 2)

    # ── Sofa on left (object to move) ──────────────────────────────────
    sx0, sy0 = 20, int(h * 0.53)
    sx1, sy1 = int(w * 0.44), int(h * 0.84)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), [110, 82, 72], -1)
    # Back cushion
    cv2.rectangle(img, (sx0, sy0), (sx1, sy0 + int(h * 0.08)), [90, 65, 58], -1)
    # Seat cushions
    mid_x = (sx0 + sx1) // 2
    cv2.line(img, (mid_x, sy0 + int(h * 0.08)), (mid_x, sy1), [90, 65, 58], 2)
    # Legs
    for lx in [sx0 + 8, sx1 - 16]:
        cv2.rectangle(img, (lx, sy1), (lx + 10, sy1 + 14), [70, 55, 45], -1)

    # Rug in center
    cv2.ellipse(img, (int(w * 0.52), int(h * 0.77)), (185, 48), 0, 0, 360,
                [175, 115, 95], -1)

    # Coffee table
    cv2.rectangle(img, (int(w * 0.30), int(h * 0.68)),
                  (int(w * 0.65), int(h * 0.78)), [155, 125, 95], -1)

    # Build source mask for sofa
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[sy0:sy1 + 14, sx0:sx1] = 255

    # Target: move sofa to right side
    target = (int(w * 0.73), int(h * 0.69))
    return img, mask, target


def _make_bedroom_lamp(h: int = 600, w: int = 800,
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Bedroom — move bedside lamp from right side to the left of the bed."""
    img = np.full((h, w, 3), [230, 225, 218], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.88), int(h * 0.52)],
                           [int(w * 0.12), int(h * 0.52)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [185, 168, 148])

    # Bed
    bx0, bx1 = int(w * 0.12), int(w * 0.78)
    by0, by1 = int(h * 0.40), int(h * 0.90)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), [200, 195, 190], -1)
    # Headboard
    cv2.rectangle(img, (bx0, int(h * 0.20)), (bx1, by0 + 5), [140, 110, 88], -1)
    # Pillows
    cv2.rectangle(img, (bx0 + 18, by0 + 12), (bx0 + 130, by0 + 58), [250, 248, 245], -1)
    cv2.rectangle(img, (bx0 + 145, by0 + 12), (bx0 + 258, by0 + 58), [250, 248, 245], -1)
    # Duvet fold
    cv2.rectangle(img, (bx0 + 4, by0 + 60), (bx1 - 4, by0 + 95), [225, 220, 215], -1)

    # Bedside table (right side)
    tx0, ty0 = int(w * 0.80), int(h * 0.46)
    tx1, ty1 = int(w * 0.94), int(h * 0.68)
    cv2.rectangle(img, (tx0, ty0), (tx1, ty1), [158, 125, 95], -1)
    cv2.line(img, (tx0, (ty0 + ty1) // 2), (tx1, (ty0 + ty1) // 2), [138, 108, 80], 2)

    # ── Bedside lamp on right table (object to move) ──────────────────
    lx = (tx0 + tx1) // 2
    ly_top = ty0 - 72
    # Shade (truncated ellipse)
    shade_pts = np.array([[lx - 32, ty0 - 5],
                           [lx + 32, ty0 - 5],
                           [lx + 18, ly_top],
                           [lx - 18, ly_top]], np.int32)
    cv2.fillPoly(img, [shade_pts], [245, 230, 195])
    cv2.polylines(img, [shade_pts], True, [170, 150, 115], 2)
    # Pole
    cv2.rectangle(img, (lx - 3, ty0 - 5), (lx + 3, ty0), [80, 70, 60], -1)
    # Base
    cv2.ellipse(img, (lx, ty0 - 1), (18, 6), 0, 0, 360, [95, 80, 65], -1)

    # Source mask for lamp
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shade_pts], 255)
    mask[max(0, ly_top - 4):ty0, lx - 22:lx + 22] = 255
    cv2.ellipse(mask, (lx, ty0 - 1), (20, 8), 0, 0, 360, 255, -1)

    # Target: place lamp on the left side of the bed (small bedside table area)
    target = (int(w * 0.09), int(h * 0.50))
    return img, mask, target


def _make_office_chair(h: int = 600, w: int = 800,
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Home office — move desk chair from foreground closer to the desk."""
    img = np.full((h, w, 3), [218, 215, 210], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.82), int(h * 0.50)],
                           [int(w * 0.18), int(h * 0.50)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [180, 165, 148])

    # Desk
    dx0, dy0 = int(w * 0.08), int(h * 0.42)
    dx1, dy1 = int(w * 0.80), int(h * 0.56)
    cv2.rectangle(img, (dx0, dy0), (dx1, dy1), [168, 140, 108], -1)
    for lx in [dx0 + 14, dx1 - 28]:
        cv2.rectangle(img, (lx, dy1), (lx + 16, dy1 + int(h * 0.22)), [145, 118, 90], -1)

    # Monitor
    mx = int(w * 0.50)
    cv2.rectangle(img, (mx - 52, int(h * 0.18)), (mx + 52, int(h * 0.42)), [38, 38, 38], -1)
    cv2.rectangle(img, (mx - 38, int(h * 0.20)), (mx + 38, int(h * 0.40)), [55, 95, 138], -1)
    cv2.rectangle(img, (mx - 8, int(h * 0.42)), (mx + 8, int(h * 0.47)), [48, 48, 48], -1)
    cv2.rectangle(img, (mx - 28, int(h * 0.47)), (mx + 28, int(h * 0.50)), [48, 48, 48], -1)

    # ── Desk chair at bottom-left foreground (object to move) ──────────
    px_c = int(w * 0.30)
    py_seat = int(h * 0.65)
    # Seat
    cv2.rectangle(img, (px_c - 55, py_seat), (px_c + 55, py_seat + 35), [48, 48, 52], -1)
    # Backrest
    cv2.rectangle(img, (px_c - 50, py_seat - 95), (px_c + 50, py_seat + 2), [52, 52, 56], -1)
    # Centre pole
    cv2.rectangle(img, (px_c - 5, py_seat + 35), (px_c + 5, py_seat + 90), [78, 78, 82], -1)
    # Star base
    base_y = py_seat + 90
    for angle_deg in range(0, 360, 72):
        ang = np.radians(angle_deg)
        ex = int(px_c + 44 * np.cos(ang))
        ey = int(base_y + 15 * np.sin(ang))
        cv2.line(img, (px_c, base_y), (ex, ey), [68, 68, 72], 3)
        cv2.circle(img, (ex, ey), 5, [58, 58, 62], -1)

    # Source mask for chair
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[py_seat - 95:py_seat + 36, px_c - 58:px_c + 58] = 255   # body
    mask[py_seat + 35:base_y + 2, px_c - 8:px_c + 8] = 255        # pole
    cv2.circle(mask, (px_c, base_y), 52, 255, -1)                  # base

    # Target: push chair closer to desk (higher in scene = deeper depth)
    target = (int(w * 0.38), int(h * 0.58))
    return img, mask, target


def _make_dining_plant(h: int = 600, w: int = 800,
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Dining room — move potted plant from right corner to left corner."""
    img = np.full((h, w, 3), [232, 226, 216], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.86), int(h * 0.54)],
                           [int(w * 0.14), int(h * 0.54)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [192, 175, 155])

    # Dining table
    tx_t, ty_t = int(w * 0.44), int(h * 0.58)
    cv2.ellipse(img, (tx_t, ty_t), (195, 68), 0, 0, 360, [158, 128, 98], -1)
    for lx in [tx_t - 135, tx_t + 135]:
        cv2.rectangle(img, (lx - 8, ty_t + 48), (lx + 8, ty_t + 128), [138, 108, 82], -1)

    # Chairs
    for cx, cy_ch, below in [
        (tx_t - 215, ty_t - 10, False), (tx_t + 215, ty_t - 10, False),
        (tx_t - 75, ty_t + 100, True),  (tx_t + 75, ty_t + 100, True),
    ]:
        back_y = cy_ch - 48 if not below else cy_ch + 8
        cv2.rectangle(img, (cx - 28, cy_ch), (cx + 28, cy_ch + 18), [178, 148, 118], -1)
        cv2.rectangle(img, (cx - 26, back_y), (cx + 26, back_y + 38), [178, 148, 118], -1)

    # ── Potted plant in right corner (object to move) ──────────────────
    plx, ply = int(w * 0.85), int(h * 0.60)
    # Pot
    pot_pts = np.array([[plx - 28, ply + 58], [plx + 28, ply + 58],
                         [plx + 20, ply],      [plx - 20, ply]], np.int32)
    cv2.fillPoly(img, [pot_pts], [162, 93, 62])
    cv2.rectangle(img, (plx - 30, ply - 5), (plx + 30, ply + 4), [138, 78, 52], -1)
    cv2.ellipse(img, (plx, ply - 3), (20, 7), 0, 0, 360, [78, 56, 38], -1)
    # Foliage
    for fx, fy, rx, ry in [
        (plx,      ply - 52, 30, 40),
        (plx - 28, ply - 35, 22, 28),
        (plx + 26, ply - 35, 22, 28),
        (plx - 16, ply - 76, 18, 25),
        (plx + 14, ply - 70, 18, 25),
    ]:
        cv2.ellipse(img, (fx, fy), (rx, ry), 0, 0, 360, [56, 115, 65], -1)

    # Source mask for plant
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pot_pts], 255)
    mask[max(0, ply - 5):ply + 5, plx - 32:plx + 32] = 255
    mask[max(0, ply - 110):ply, plx - 55:plx + 55] = 255

    # Target: move plant to left corner
    target = (int(w * 0.08), int(h * 0.62))
    return img, mask, target


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _make_comparison(
    before: np.ndarray,
    source_mask: np.ndarray,
    target_pos: tuple[int, int],
    after: np.ndarray,
    title: str,
) -> np.ndarray:
    """3-panel: Before (source mask overlay) | Arrow + target marker | After."""
    h, w = before.shape[:2]

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Before panel: red source-mask overlay
    before_vis = before.copy().astype(np.float32)
    bin_mask = source_mask > 127
    before_vis[bin_mask] = before_vis[bin_mask] * 0.4 + np.array([220, 50, 50]) * 0.6
    before_vis = before_vis.clip(0, 255).astype(np.uint8)
    # Draw target crosshair
    bgr = cv2.cvtColor(before_vis, cv2.COLOR_RGB2BGR)
    tx, ty = target_pos
    r = 16
    cv2.circle(bgr, (tx, ty), r, (50, 50, 220), 3)
    cv2.line(bgr, (tx - r, ty), (tx + r, ty), (50, 50, 220), 2)
    cv2.line(bgr, (tx, ty - r), (tx, ty + r), (50, 50, 220), 2)
    cv2.putText(bgr, "Target", (tx + 20, ty - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (50, 50, 220), 2, cv2.LINE_AA)
    before_vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    p_before = _label(before_vis, "Before  (mask=red, target=blue)")
    p_after  = _label(after,      "After   (moved + colour-matched)")

    div = np.full((h, 4, 3), 200, dtype=np.uint8)
    comparison = np.concatenate([p_before, div, p_after], axis=1)

    title_bar = np.full((38, comparison.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (220, 200, 180), 2, cv2.LINE_AA)
    return np.concatenate([title_bar, comparison], axis=0)


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run(backend: str = "lama") -> None:
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build test cases
    lr_img, lr_mask, lr_tgt = _make_living_room_sofa()
    bd_img, bd_mask, bd_tgt = _make_bedroom_lamp()
    of_img, of_mask, of_tgt = _make_office_chair()
    dr_img, dr_mask, dr_tgt = _make_dining_plant()

    test_cases: list[MoveTestCase] = [
        MoveTestCase("01_living_room_sofa",  lr_img, lr_mask, lr_tgt,
                     "Living room — move sofa left→right"),
        MoveTestCase("02_bedroom_lamp",      bd_img, bd_mask, bd_tgt,
                     "Bedroom — move bedside lamp right→left"),
        MoveTestCase("03_office_chair",      of_img, of_mask, of_tgt,
                     "Home office — push chair closer to desk"),
        MoveTestCase("04_dining_plant",      dr_img, dr_mask, dr_tgt,
                     "Dining room — move plant right→left corner"),
    ]

    # Override inpainting backend if requested
    cfg = config
    if backend == "opencv":
        from pipeline.inpainting import InpaintingWrapper
        cfg._inpaint_backend_override = backend  # orchestrator ignores; handled below

    orch = PipelineOrchestrator(cfg)

    # If OpenCV backend requested, swap out the lama wrapper so it uses the
    # fallback path without trying to download LaMa weights.
    if backend == "opencv":
        from pipeline.inpainting import InpaintingWrapper
        orch._mm._registry["lama"] = InpaintingWrapper(cfg, backend="opencv")

    comparison_panels: list[np.ndarray] = []

    logger.info("─" * 60)
    logger.info("Running %d move tests (inpaint backend=%s)", len(test_cases), backend)
    logger.info("─" * 60)

    for tc in test_cases:
        logger.info("Test: %s", tc.description)
        t0 = time.perf_counter()

        try:
            result = orch.move(
                image=tc.image,
                object_name="",                   # mask provided — skip SAM
                target_position=tc.target_position,
                source_mask=tc.source_mask,
            )
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            result = tc.image.copy()   # fallback: keep original so summary still renders

        elapsed = time.perf_counter() - t0
        logger.info("  Done in %.2f s", elapsed)

        prefix = out_dir / f"move_{tc.name}"
        save_image(tc.image,  f"{prefix}_before.png")
        save_image(result,    f"{prefix}_after.png")
        cv2.imwrite(str(f"{prefix}_mask.png"), tc.source_mask)

        panel = _make_comparison(
            tc.image, tc.source_mask, tc.target_position, result, tc.description
        )
        save_image(panel, f"{prefix}_comparison.png")
        comparison_panels.append(panel)
        logger.info("  Saved: %s_*", prefix.name)

    # ── Tiled summary ──────────────────────────────────────────────────
    target_w = max(p.shape[1] for p in comparison_panels)
    rows: list[np.ndarray] = []
    for p in comparison_panels:
        if p.shape[1] < target_w:
            pad = np.full((p.shape[0], target_w - p.shape[1], 3), 240, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        rows.append(p)
        rows.append(np.full((6, target_w, 3), 200, dtype=np.uint8))

    summary = np.concatenate(rows[:-1], axis=0)
    summary_path = out_dir / "move_summary.png"
    save_image(summary, summary_path)

    logger.info("─" * 60)
    logger.info("Summary saved: %s", summary_path)
    logger.info("All outputs in: %s", out_dir)

    orch.teardown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReScene AI — Phase 4 move test")
    p.add_argument(
        "--backend",
        choices=["lama", "opencv"],
        default="lama",
        help="Inpainting backend (default: lama — falls back to opencv if weights absent)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(backend=args.backend)
