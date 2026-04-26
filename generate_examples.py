"""
ReScene AI — Generate pre-computed example pairs for the demo gallery.

For each image found in  data/examples/input/  (JPEG or PNG) this script:
  1. Runs the restyle pipeline on it (no ML weights required).
  2. Saves a labelled before/after side-by-side to  data/examples/output/.

If no input images are found, synthetic room images are generated so the
gallery is never empty during a live demo.

Usage:
    conda activate torch-cu121
    python generate_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import config, STYLES, DATA_DIR
from pipeline.style_transfer import StyleTransferWrapper
from pipeline.harmonization import HarmonizationWrapper
from utils.image_utils import ensure_rgb

INPUT_DIR  = DATA_DIR / "examples" / "input"
OUTPUT_DIR = DATA_DIR / "examples" / "output"


# ---------------------------------------------------------------------------
# Synthetic room generator (fallback when no real photos supplied)
# ---------------------------------------------------------------------------

def _make_synthetic_room(seed: int = 0, w: int = 768, h: int = 512) -> np.ndarray:
    """Draw a simple synthetic room (wall + floor + window + furniture outlines)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    floor_y = int(h * 0.62)

    # Wall
    wall_col = rng.integers([185, 180, 170], [230, 225, 215], 3).tolist()
    img[:floor_y, :] = wall_col

    # Floor (wood-toned gradient)
    for y in range(floor_y, h):
        t = (y - floor_y) / max(h - floor_y, 1)
        col = [int(140 + 30 * t), int(95 + 20 * t), int(50 + 15 * t)]
        img[y, :] = col

    # Window
    wx0, wy0 = int(w * 0.60), int(h * 0.08)
    wx1, wy1 = int(w * 0.88), int(h * 0.52)
    img[wy0:wy1, wx0:wx1] = [185, 215, 240]   # sky blue
    cv2.rectangle(img, (wx0, wy0), (wx1, wy1), [80, 70, 60], 3)
    mid_x, mid_y = (wx0 + wx1) // 2, (wy0 + wy1) // 2
    cv2.line(img, (mid_x, wy0), (mid_x, wy1), [80, 70, 60], 2)
    cv2.line(img, (wx0, mid_y), (wx1, mid_y), [80, 70, 60], 2)

    # Sofa outline
    sx0, sy0 = int(w * 0.04), floor_y - int(h * 0.20)
    sx1, sy1 = int(w * 0.48), floor_y
    sofa_col = rng.integers([80, 70, 60], [160, 140, 120], 3).tolist()
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), sofa_col, -1)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), [50, 45, 40], 2)
    # Back cushion
    cv2.rectangle(img, (sx0, sy0), (sx1, sy0 + int((sy1 - sy0) * 0.35)),
                  [c + 15 for c in sofa_col], -1)

    # Lamp
    lx, ly0 = int(w * 0.52), floor_y - int(h * 0.30)
    cv2.line(img, (lx, ly0 + int(h * 0.25)), (lx, floor_y), [90, 80, 70], 3)
    pts = np.array([[lx - 25, ly0 + 30], [lx + 25, ly0 + 30], [lx, ly0]], np.int32)
    cv2.fillPoly(img, [pts], [220, 200, 150])

    return img


# ---------------------------------------------------------------------------
# Before / after side-by-side helper
# ---------------------------------------------------------------------------

def _side_by_side(before: np.ndarray, after: np.ndarray, label: str) -> np.ndarray:
    target_h = max(before.shape[0], after.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        sc = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * sc), target_h))

    def _add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(out, text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
        return out

    b = _add_label(_fit(before), "Before")
    a = _add_label(_fit(after), "After")
    div = np.full((target_h, 4, 3), 180, dtype=np.uint8)
    composite = np.concatenate([b, div, a], axis=1)

    # Style label banner at bottom
    banner_h = 32
    banner = np.zeros((banner_h, composite.shape[1], 3), dtype=np.uint8)
    banner[:] = [30, 30, 30]
    cv2.putText(banner, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([composite, banner])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect input images
    inputs: list[tuple[str, np.ndarray]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in sorted(INPUT_DIR.glob(ext)):
            img = cv2.imread(str(p))
            if img is not None:
                inputs.append((p.stem, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    if not inputs:
        print("[generate_examples] No images in data/examples/input/ — generating synthetic rooms.")
        for i in range(3):
            inputs.append((f"synthetic_room_{i + 1}", _make_synthetic_room(seed=i)))

    # Load only the weight-free wrappers (no SAM/ZoeDepth/LaMa required)
    styler = StyleTransferWrapper(config)
    styler.load()
    harmonizer = HarmonizationWrapper(config)
    harmonizer.load()

    style_names = list(STYLES.keys())

    total = 0
    for stem, rgb in inputs:
        # Apply two styles per image to get variety
        for style_name in style_names[:2]:
            print(f"  {stem} → {style_name} …", end=" ", flush=True)
            try:
                rgb_clean = ensure_rgb(rgb)
                styled = styler.predict(image=rgb_clean, style_name=style_name)
                full_mask = np.full(rgb_clean.shape[:2], 255, dtype=np.uint8)
                result = harmonizer.predict(composite=styled, foreground_mask=full_mask)
                composite = _side_by_side(rgb, result, f"{stem} · {style_name}")
                # Save as BGR for cv2
                out_path = OUTPUT_DIR / f"{stem}_{style_name.lower().replace(' ', '_')}.jpg"
                cv2.imwrite(str(out_path),
                            cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"saved → {out_path.name}")
                total += 1
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"\n[generate_examples] Done — {total} examples saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
