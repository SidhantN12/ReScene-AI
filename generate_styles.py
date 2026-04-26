"""
ReScene AI — Generate style reference images for data/styles/.

Each 512×512 JPEG encodes the characteristic colour distribution of a named
interior style.  The Reinhard LAB transfer in pipeline/style_transfer.py
extracts mean+std statistics from these images, so getting the colour
*proportions* right is the key quality factor.

Generation strategy: weighted colour tiling — the 512×512 canvas is divided
into small tiles, each filled with a style colour sampled from a weighted
distribution.  The result is a mosaic that has exactly the global LAB statistics
we want the transfer to aim for.  Gaussian blur smooths tile boundaries.

Run once before starting the app (or after editing palettes):
    conda activate torch-cu121
    python generate_styles.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "styles"

_STYLE_FILENAMES: dict[str, str] = {
    "Scandinavian":       "scandinavian.jpg",
    "Industrial":         "industrial.jpg",
    "Bohemian":           "bohemian.jpg",
    "Mid-Century Modern": "midcentury.jpg",
    "Minimalist":         "minimalist.jpg",
    "Coastal":            "coastal.jpg",
}

# ---------------------------------------------------------------------------
# Colour palettes
#
# Each entry: ([R, G, B], weight)
# Weights are proportional to how much of a real room photo in this style
# would be covered by that colour family.  They don't need to sum to 1 —
# we normalise them internally.
# ---------------------------------------------------------------------------

_PALETTES: dict[str, list[tuple[list[int], float]]] = {

    # ── Scandinavian ────────────────────────────────────────────────────────
    # Clean, airy, Nordic.  Dominated by warm whites and very light grays;
    # birch-blonde wood as mid-tone; cool steel-blue as occasional accent.
    # Very low overall saturation, high luminance.
    "Scandinavian": [
        ([252, 249, 245], 0.30),  # warm white — walls
        ([240, 236, 230], 0.20),  # off-white — ceiling / large surfaces
        ([224, 219, 210], 0.15),  # very light warm gray
        ([208, 196, 175], 0.12),  # birch-blonde wood
        ([188, 178, 160], 0.08),  # medium birch
        ([168, 182, 192], 0.08),  # steel-blue/slate accent
        ([186, 196, 188], 0.04),  # soft sage (plant)
        ([155, 148, 142], 0.03),  # warm gray (cushion/rug)
    ],

    # ── Industrial ──────────────────────────────────────────────────────────
    # Urban, moody.  Dark charcoals and raw concrete make up most of the image;
    # rust/oxidised iron as warm accent; minimal saturation throughout.
    "Industrial": [
        ([38,  36,  34], 0.25),   # dark charcoal — walls
        ([62,  60,  56], 0.22),   # medium charcoal
        ([98,  94,  88], 0.20),   # concrete gray — floors/ceiling
        ([72,  68,  65], 0.14),   # mid-dark gray
        ([125, 62,  35], 0.09),   # rust / oxidised steel
        ([85,  82,  90], 0.06),   # blue-steel accent
        ([148, 108, 58], 0.04),   # aged brass/copper
    ],

    # ── Bohemian ────────────────────────────────────────────────────────────
    # Rich, warm, maximalist.  Terracotta and warm earth tones dominate;
    # jewel-tone accents (teal, burgundy, mustard) sprinkled throughout.
    # Medium-high saturation, medium luminance.
    "Bohemian": [
        ([198, 112, 62], 0.22),   # terracotta — main tone
        ([218, 188, 138], 0.18),  # warm sand / cream
        ([150, 70,  45], 0.14),   # deep terracotta / clay
        ([38,  105, 98], 0.11),   # deep teal
        ([85,  35,  40], 0.09),   # burgundy / wine
        ([205, 152, 28], 0.09),   # mustard gold
        ([162, 108, 68], 0.08),   # warm walnut brown
        ([88,  98,  52], 0.05),   # olive green (plants)
        ([228, 168, 118], 0.04),  # light terracotta highlight
    ],

    # ── Mid-Century Modern ──────────────────────────────────────────────────
    # Retro warmth.  Walnut/teak furniture with avocado green, burnt orange,
    # and cream walls.  Medium saturation, warm-balanced luminance.
    "Mid-Century Modern": [
        ([232, 218, 188], 0.25),  # warm cream / ivory — walls
        ([92,  58,  30], 0.22),   # walnut brown — furniture
        ([108, 122, 55], 0.14),   # avocado / olive green
        ([188, 85,  30], 0.14),   # burnt orange — accent
        ([162, 140, 94], 0.10),   # tan / khaki
        ([188, 148, 38], 0.08),   # gold / mustard
        ([68,  42,  24], 0.04),   # dark walnut shadow
        ([235, 195, 145], 0.03),  # warm highlight
    ],

    # ── Minimalist ──────────────────────────────────────────────────────────
    # Reduction to essentials.  Near-pure whites and very light grays make up
    # the vast majority; a trace of near-black for structural lines.
    # Near-zero saturation, very high luminance.
    "Minimalist": [
        ([253, 253, 251], 0.38),  # pure white
        ([244, 242, 240], 0.25),  # near white
        ([230, 228, 226], 0.18),  # very light warm gray
        ([205, 202, 200], 0.10),  # light gray — shadow areas
        ([162, 160, 158], 0.06),  # medium gray — furniture
        ([32,  30,  30],  0.03),  # near-black — frames / hardware
    ],

    # ── Coastal ─────────────────────────────────────────────────────────────
    # Breezy, fresh.  Ocean blues and seafoam teal balance warm sandy beiges;
    # white acts as high-key contrast.  Medium saturation, medium-high luminance.
    "Coastal": [
        ([205, 226, 240], 0.22),  # light sky blue — walls
        ([212, 202, 178], 0.20),  # sandy beige — floors / fabrics
        ([68,  148, 188], 0.20),  # ocean blue — accent / cushions
        ([108, 188, 175], 0.16),  # seafoam teal
        ([158, 142, 122], 0.12),  # driftwood warm gray
        ([248, 242, 228], 0.10),  # white sand — bright surfaces
    ],
}


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

_TILE: int = 16     # tile size in pixels (512 / 16 = 32 tiles per row)
_BLUR: int = 21     # Gaussian blur kernel (must be odd) — smooths tile edges
_VARIATION: float = 10.0   # per-tile colour jitter std (0-255)


def _make_palette_image(
    color_weights: list[tuple[list[int], float]],
    seed: int = 42,
) -> np.ndarray:
    """Generate a 512×512 uint8 RGB image from a weighted colour palette.

    Each tile in the canvas is filled with a colour sampled from the palette.
    A slight random jitter is added per tile so the image doesn't look like a
    flat poster.  The result is blurred to smooth tile boundaries.
    """
    rng = np.random.default_rng(seed)
    n = 512 // _TILE

    colors  = np.array([c for c, _ in color_weights], dtype=np.float32)
    weights = np.array([w for _, w in color_weights], dtype=np.float64)
    weights /= weights.sum()

    canvas = np.zeros((512, 512, 3), dtype=np.float32)

    for ti in range(n):
        for tj in range(n):
            idx = int(rng.choice(len(colors), p=weights))
            base = colors[idx]
            jitter = rng.normal(0.0, _VARIATION, 3).astype(np.float32)
            fill = np.clip(base + jitter, 0.0, 255.0)
            r0, r1 = ti * _TILE, (ti + 1) * _TILE
            c0, c1 = tj * _TILE, (tj + 1) * _TILE
            canvas[r0:r1, c0:c1] = fill

    u8 = canvas.clip(0, 255).astype(np.uint8)
    # Blur smooths tile boundaries; kernel must be odd
    return cv2.GaussianBlur(u8, (_BLUR, _BLUR), _BLUR // 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, palette in _PALETTES.items():
        img_rgb = _make_palette_image(palette)
        fname = _STYLE_FILENAMES[name]
        out_path = DATA_DIR / fname
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Print the mean LAB values so they can be compared / tuned
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        print(f"  {name:<22}  L={lab[:,:,0].mean():.1f}  "
              f"a={lab[:,:,1].mean():.1f}  b={lab[:,:,2].mean():.1f}  → {fname}")
    print(f"\nDone — {len(_PALETTES)} style reference images written to {DATA_DIR}")


if __name__ == "__main__":
    main()
