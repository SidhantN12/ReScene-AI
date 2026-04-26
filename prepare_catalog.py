"""
Prepare a real furniture catalog for ReScene AI.

Copies product images from a source folder into data/catalog/ and normalizes
them to transparent PNGs so the Add tab can use real furniture assets instead
of the synthetic samples from generate_catalog.py.

Usage:
    python prepare_catalog.py C:\\path\\to\\furniture_images
    python prepare_catalog.py C:\\path\\to\\furniture_images --clear
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

CATALOG_DIR = Path(__file__).parent / "data" / "catalog"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return cleaned.strip("_") or "item"


def _to_rgba(src: np.ndarray, white_thresh: int = 240) -> np.ndarray:
    if src.ndim == 2:
        rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        alpha = np.full(src.shape[:2], 255, dtype=np.uint8)
        return np.dstack([rgb, alpha])

    if src.shape[2] == 4:
        return cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA)

    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    is_bg = (r > white_thresh) & (g > white_thresh) & (b > white_thresh)
    alpha = np.where(is_bg, np.uint8(0), np.uint8(255))
    return np.dstack([rgb, alpha])


def _crop_to_alpha(rgba: np.ndarray, pad: int = 8) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(rgba.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(rgba.shape[0], int(ys.max()) + 1 + pad)
    return rgba[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a real furniture catalog")
    parser.add_argument("source", type=Path, help="Folder containing furniture images")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing catalog images before import")
    parser.add_argument("--white-thresh", type=int, default=240,
                        help="RGB threshold used to treat near-white background as transparent")
    args = parser.parse_args()

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}")

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for path in CATALOG_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                path.unlink()

    imported = 0
    for path in sorted(args.source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"SKIP {path.name} - could not read")
            continue

        rgba = _to_rgba(raw, white_thresh=args.white_thresh)
        rgba = _crop_to_alpha(rgba)

        stem = _slugify(path.stem)
        dest = CATALOG_DIR / f"{stem}.png"
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(dest), bgra)
        imported += 1
        print(f"OK   {path.name} -> {dest.name}")

    print(f"\nImported {imported} furniture assets into {CATALOG_DIR}")


if __name__ == "__main__":
    main()
