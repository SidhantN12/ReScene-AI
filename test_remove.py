"""
ReScene AI — Phase 3 test script: Remove operation.

Generates 4 distinct synthetic room images, applies a known object mask to
each, runs the full remove pipeline (inpainting + harmonization), and saves:

    outputs/remove_test_N_before.png
    outputs/remove_test_N_mask.png
    outputs/remove_test_N_after.png
    outputs/remove_test_N_comparison.png   ← 3-panel: before | mask | after
    outputs/remove_summary.png             ← all 4 tests tiled

Usage
-----
    # Full pipeline (uses LaMa if installed, else OpenCV):
    conda activate torch-cu121
    python test_remove.py

    # Force OpenCV fallback (no model weights needed):
    python test_remove.py --backend opencv
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
logger = logging.getLogger("test_remove")

from config import config, OUTPUTS_DIR
from pipeline.inpainting import InpaintingWrapper
from pipeline.harmonization import HarmonizationWrapper
from pipeline.model_manager import ModelManager
from utils.image_utils import save_image


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    image: np.ndarray          # H×W×3 uint8 RGB
    mask: np.ndarray           # H×W uint8  (255 = object to remove)
    description: str


# ---------------------------------------------------------------------------
# Synthetic room generators
# ---------------------------------------------------------------------------

def _make_living_room(h: int = 600, w: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Living room with a floor lamp — remove the lamp."""
    img = np.full((h, w, 3), [235, 228, 218], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.85), int(h * 0.55)],
                           [int(w * 0.15), int(h * 0.55)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [195, 178, 155])

    # Back wall
    wall_pts = np.array([[int(w * 0.15), int(h * 0.55)], [int(w * 0.85), int(h * 0.55)],
                          [w, 0], [0, 0]], np.int32)
    cv2.fillPoly(img, [wall_pts], [228, 222, 212])

    # Sofa
    cv2.rectangle(img, (30, int(h * 0.55)), (int(w * 0.52), int(h * 0.85)),
                  [115, 85, 75], -1)
    cv2.rectangle(img, (30, int(h * 0.55)), (int(w * 0.52), int(h * 0.63)),
                  [95, 65, 58], -1)

    # Rug
    cv2.ellipse(img, (int(w * 0.42), int(h * 0.78)), (220, 55), 0, 0, 360,
                [175, 120, 100], -1)

    # ── Floor lamp (object to remove) ──
    lx, ly_base = int(w * 0.70), int(h * 0.88)
    # Pole
    cv2.rectangle(img, (lx - 4, int(h * 0.32)), (lx + 4, ly_base), [60, 55, 50], -1)
    # Shade
    shade_pts = np.array([[lx - 38, int(h * 0.32)],
                           [lx + 38, int(h * 0.32)],
                           [lx + 22, int(h * 0.17)],
                           [lx - 22, int(h * 0.17)]], np.int32)
    cv2.fillPoly(img, [shade_pts], [240, 225, 190])
    cv2.polylines(img, [shade_pts], True, [160, 140, 110], 3)
    # Base
    cv2.ellipse(img, (lx, ly_base), (28, 10), 0, 0, 360, [70, 65, 60], -1)

    # Build mask for the lamp
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (lx - 6, int(h * 0.32)), (lx + 6, ly_base + 2), 255, -1)
    cv2.fillPoly(mask, [shade_pts], 255)
    mask[int(h * 0.17):int(h * 0.32), lx - 40:lx + 40] = 255  # full shade zone
    cv2.ellipse(mask, (lx, ly_base), (30, 12), 0, 0, 360, 255, -1)

    return img, mask


def _make_bedroom(h: int = 600, w: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Bedroom with a bedside cabinet — remove the cabinet."""
    img = np.full((h, w, 3), [230, 225, 218], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.88), int(h * 0.52)],
                           [int(w * 0.12), int(h * 0.52)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [185, 168, 148])

    # Bed
    bx0, by0 = 20, int(h * 0.42)
    bx1, by1 = int(w * 0.68), int(h * 0.90)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), [200, 195, 190], -1)
    # Pillow
    cv2.rectangle(img, (bx0 + 20, by0 + 12), (bx0 + 150, by0 + 60), [250, 248, 245], -1)
    cv2.rectangle(img, (bx0 + 165, by0 + 12), (bx0 + 300, by0 + 60), [250, 248, 245], -1)
    # Headboard
    cv2.rectangle(img, (bx0, int(h * 0.20)), (bx1, by0 + 5), [140, 110, 90], -1)

    # ── Bedside cabinet (object to remove) ──
    cx0, cy0 = int(w * 0.70), int(h * 0.48)
    cx1, cy1 = int(w * 0.86), int(h * 0.75)
    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), [155, 120, 90], -1)
    # Drawer lines
    mid_y = (cy0 + cy1) // 2
    cv2.line(img, (cx0, mid_y), (cx1, mid_y), [130, 100, 75], 2)
    # Knobs
    for ky in [(cy0 + mid_y) // 2, (mid_y + cy1) // 2]:
        cv2.circle(img, ((cx0 + cx1) // 2, ky), 5, [90, 70, 55], -1)
    # Lamp on cabinet
    cv2.ellipse(img, ((cx0 + cx1) // 2, cy0 - 18), (22, 28), 0, 0, 360, [240, 230, 200], -1)
    cv2.line(img, ((cx0 + cx1) // 2, cy0 - 5), ((cx0 + cx1) // 2, cy0), [100, 90, 80], 3)

    # Build mask for cabinet + lamp on it
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[cy0:cy1, cx0:cx1] = 255
    mask[cy0 - 50:cy0, cx0 + 5:cx1 - 5] = 255  # lamp above cabinet

    return img, mask


def _make_home_office(h: int = 600, w: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Home office — remove the desk chair."""
    img = np.full((h, w, 3), [218, 215, 210], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.82), int(h * 0.50)],
                           [int(w * 0.18), int(h * 0.50)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [180, 165, 148])

    # Desk
    dx0, dy0 = int(w * 0.05), int(h * 0.40)
    dx1, dy1 = int(w * 0.78), int(h * 0.55)
    cv2.rectangle(img, (dx0, dy0), (dx1, dy1), [170, 140, 110], -1)
    # Desk legs
    for lx in [dx0 + 15, dx1 - 30]:
        cv2.rectangle(img, (lx, dy1), (lx + 16, dy1 + int(h * 0.22)),
                      [145, 118, 92], -1)
    # Monitor on desk
    mx = int(w * 0.50)
    cv2.rectangle(img, (mx - 55, int(h * 0.18)), (mx + 55, int(h * 0.40)),
                  [40, 40, 40], -1)
    cv2.rectangle(img, (mx - 40, int(h * 0.20)), (mx + 40, int(h * 0.38)),
                  [60, 100, 140], -1)
    cv2.rectangle(img, (mx - 8, int(h * 0.40)), (mx + 8, int(h * 0.46)),
                  [50, 50, 50], -1)

    # ── Desk chair (object to remove) ──
    # Seat
    sx0, sy0 = int(w * 0.28), int(h * 0.52)
    sx1, sy1 = int(w * 0.56), int(h * 0.62)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), [50, 50, 55], -1)
    # Backrest
    bkx0, bkx1 = int(w * 0.30), int(w * 0.54)
    bky0, bky1 = int(h * 0.32), int(h * 0.52)
    cv2.rectangle(img, (bkx0, bky0), (bkx1, bky1), [55, 55, 60], -1)
    # Centre pole
    px = (sx0 + sx1) // 2
    cv2.rectangle(img, (px - 5, sy1), (px + 5, int(h * 0.78)), [80, 80, 85], -1)
    # Wheels (5-point star base)
    base_y = int(h * 0.78)
    for angle_deg in range(0, 360, 72):
        angle = np.radians(angle_deg)
        ex = int(px + 45 * np.cos(angle))
        ey = int(base_y + 15 * np.sin(angle))
        cv2.line(img, (px, base_y), (ex, ey), [70, 70, 75], 3)
        cv2.circle(img, (ex, ey), 5, [60, 60, 65], -1)

    # Build mask for chair
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[bky0:sy1, bkx0:bkx1] = 255   # backrest + seat
    mask[sy1:int(h * 0.80), px - 10:px + 10] = 255   # pole
    cv2.circle(mask, (px, base_y), 55, 255, -1)   # star base area

    return img, mask


def _make_dining_room(h: int = 600, w: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Dining room — remove the potted plant in the corner."""
    img = np.full((h, w, 3), [232, 226, 216], dtype=np.uint8)

    # Floor
    floor_pts = np.array([[0, h], [w, h], [int(w * 0.86), int(h * 0.54)],
                           [int(w * 0.14), int(h * 0.54)]], np.int32)
    cv2.fillPoly(img, [floor_pts], [192, 175, 155])

    # Dining table
    tx, ty = int(w * 0.44), int(h * 0.58)
    cv2.ellipse(img, (tx, ty), (200, 70), 0, 0, 360, [160, 130, 100], -1)
    # Table legs
    for lx in [tx - 140, tx + 140]:
        cv2.rectangle(img, (lx - 8, ty + 50), (lx + 8, ty + 130), [140, 110, 85], -1)

    # Chairs around table
    chair_positions = [
        (tx - 220, ty - 10, False),
        (tx + 220, ty - 10, False),
        (tx - 80,  ty + 105, True),
        (tx + 80,  ty + 105, True),
    ]
    for cx, cy_ch, below in chair_positions:
        back_y = cy_ch - 50 if not below else cy_ch + 10
        seat_y = cy_ch if not below else cy_ch
        cv2.rectangle(img, (cx - 30, seat_y), (cx + 30, seat_y + 20), [180, 150, 120], -1)
        cv2.rectangle(img, (cx - 28, back_y), (cx + 28, back_y + 40), [180, 150, 120], -1)

    # ── Potted plant (object to remove) ──
    plx, ply = int(w * 0.84), int(h * 0.60)
    # Pot
    pot_pts = np.array([[plx - 30, ply + 60],
                         [plx + 30, ply + 60],
                         [plx + 22, ply],
                         [plx - 22, ply]], np.int32)
    cv2.fillPoly(img, [pot_pts], [165, 95, 65])
    cv2.rectangle(img, (plx - 32, ply - 6), (plx + 32, ply + 4), [140, 80, 55], -1)
    # Soil
    cv2.ellipse(img, (plx, ply - 3), (22, 8), 0, 0, 360, [80, 58, 40], -1)
    # Foliage
    for fx, fy, rx, ry in [
        (plx,      ply - 55, 32, 42),
        (plx - 30, ply - 38, 24, 30),
        (plx + 28, ply - 38, 24, 30),
        (plx - 18, ply - 80, 20, 28),
        (plx + 15, ply - 72, 20, 28),
    ]:
        cv2.ellipse(img, (fx, fy), (rx, ry), 0, 0, 360, [58, 118, 68], -1)

    # Build mask for the plant
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pot_pts], 255)
    mask[max(0, ply - 6):ply + 4, plx - 34:plx + 34] = 255
    mask[max(0, ply - 115):ply, plx - 58:plx + 58] = 255  # foliage zone

    return img, mask


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _make_comparison(before: np.ndarray, mask: np.ndarray, after: np.ndarray,
                     title: str) -> np.ndarray:
    """3-panel side-by-side: Before | Mask (red overlay) | After."""
    h, w = before.shape[:2]

    def _label(img: np.ndarray, text: str, colour=(240, 240, 240)) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    colour, 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Mask panel: red overlay
    mask_vis = before.copy().astype(np.float32)
    bin_mask = mask > 127
    mask_vis[bin_mask] = mask_vis[bin_mask] * 0.4 + np.array([220, 50, 50]) * 0.6
    mask_vis = mask_vis.clip(0, 255).astype(np.uint8)

    p_before = _label(before, "Before")
    p_mask   = _label(mask_vis, "Mask (red)")
    p_after  = _label(after, "After (inpainted)")

    div = np.full((h, 4, 3), 200, dtype=np.uint8)
    comparison = np.concatenate([p_before, div, p_mask, div, p_after], axis=1)

    # Title bar
    title_bar = np.full((38, comparison.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (220, 200, 180), 2, cv2.LINE_AA)
    return np.concatenate([title_bar, comparison], axis=0)


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run(backend: str = "lama") -> None:
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build test cases
    living_img, living_mask = _make_living_room()
    bedroom_img, bedroom_mask = _make_bedroom()
    office_img, office_mask = _make_home_office()
    dining_img, dining_mask = _make_dining_room()

    test_cases: list[TestCase] = [
        TestCase("01_living_room",  living_img,  living_mask,  "Living room — floor lamp removal"),
        TestCase("02_bedroom",      bedroom_img,  bedroom_mask, "Bedroom — bedside cabinet removal"),
        TestCase("03_home_office",  office_img,   office_mask,  "Home office — desk chair removal"),
        TestCase("04_dining_room",  dining_img,   dining_mask,  "Dining room — potted plant removal"),
    ]

    # Build minimal model manager for inpainting + harmonization
    mm = ModelManager(config)
    inpainter = InpaintingWrapper(config, backend=backend)   # type: ignore[arg-type]
    harmonizer = HarmonizationWrapper(config)
    mm.register("lama", inpainter)
    mm.register("iharmony4", harmonizer)

    comparison_panels = []

    logger.info("─" * 60)
    logger.info("Running %d remove tests (backend=%s)", len(test_cases), backend)
    logger.info("─" * 60)

    for tc in test_cases:
        logger.info("Test: %s", tc.description)
        t0 = time.perf_counter()

        # ── Inpaint ──
        inp = mm.get("lama")
        inpainted = inp.predict(image=tc.image, mask=tc.mask, dilation_px=10)
        mm.unload_current()

        # ── Harmonize ──
        harm = mm.get("iharmony4")
        result = harm.predict(composite=inpainted, foreground_mask=tc.mask)
        mm.unload_current()

        elapsed = time.perf_counter() - t0
        logger.info("  Done in %.2f s", elapsed)

        # Save individual outputs
        prefix = out_dir / f"remove_{tc.name}"
        save_image(tc.image, f"{prefix}_before.png")
        cv2.imwrite(str(f"{prefix}_mask.png"), tc.mask)
        save_image(result, f"{prefix}_after.png")

        # Comparison panel
        panel = _make_comparison(tc.image, tc.mask, result, tc.description)
        save_image(panel, f"{prefix}_comparison.png")
        comparison_panels.append(panel)
        logger.info("  Saved: %s_*", prefix.name)

    # ── Tiled summary ──────────────────────────────────────────────────
    # Stack all panels vertically (already same width)
    target_w = max(p.shape[1] for p in comparison_panels)
    rows = []
    for p in comparison_panels:
        if p.shape[1] < target_w:
            pad = np.full((p.shape[0], target_w - p.shape[1], 3), 240, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        rows.append(p)
        rows.append(np.full((6, target_w, 3), 200, dtype=np.uint8))   # divider

    summary = np.concatenate(rows[:-1], axis=0)   # drop last divider
    summary_path = out_dir / "remove_summary.png"
    save_image(summary, summary_path)

    logger.info("─" * 60)
    logger.info("Summary saved: %s", summary_path)
    logger.info("All outputs in: %s", out_dir)

    mm.unload_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReScene AI — Phase 3 remove test")
    p.add_argument(
        "--backend",
        choices=["lama", "opencv"],
        default="lama",
        help="Inpainting backend: 'lama' tries simple-lama-inpainting first, "
             "'opencv' forces OpenCV INPAINT_TELEA (no weights required)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(backend=args.backend)
