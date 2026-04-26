"""
Furniture image preprocessing pipeline for ReScene AI.

Reads raw product photos from Furniture_images/ and writes clean
alpha-cutout PNGs to Furniture_images_processed/.  The app reads
from the processed folder when it exists.

Background removal uses flood-fill seeded from all border pixels,
followed by morphological cleanup and edge feathering — no ML
dependencies required.

Usage
-----
    python process_furniture.py              # incremental (skip unchanged)
    python process_furniture.py --force      # reprocess everything
    python process_furniture.py --prune      # also remove orphaned outputs
    python process_furniture.py --tolerance 40  # looser BG detection

Add new images to Furniture_images/ and re-run at any time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT       = Path(__file__).parent.resolve()
SOURCE_DIR      = REPO_ROOT / "Furniture_images"
OUTPUT_DIR      = REPO_ROOT / "Furniture_images_processed"
MANIFEST_PATH   = OUTPUT_DIR / ".process_manifest.json"

SUPPORTED_EXTS  = {".png", ".jpg", ".jpeg", ".webp"}

# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------

def _flood_fill_fg_mask(rgb: np.ndarray, tolerance: int) -> np.ndarray:
    """Return uint8 mask (255 = foreground).

    Two-stage strategy that avoids eating into light-coloured furniture:

    1. Build a binary "near-background" map: pixels within `tolerance` of the
       median corner colour (works for white AND near-white / light-grey BG).
    2. Flood-fill that binary map from the 4 corners — only the *connected*
       background region is removed, not interior white pixels on the object.
    """
    h, w = rgb.shape[:2]

    # ── Stage 1: estimate BG colour from the 4 corners ──────────────────────
    corner_pixels = np.array([
        rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1]
    ], dtype=np.float32)
    bg_color = corner_pixels.mean(axis=0)          # (R, G, B)

    # Per-pixel L2 distance to the background colour
    diff = rgb.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))        # (H, W)

    # Binary mask: True = "looks like background colour"
    near_bg = (dist <= tolerance).astype(np.uint8) * 255

    # ── Stage 2: flood-fill from corners on the binary mask ─────────────────
    # This keeps only the *connected* background region.
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for (seed_y, seed_x) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if near_bg[seed_y, seed_x] > 0:
            cv2.floodFill(near_bg, fill_mask, (seed_x, seed_y),
                          (0,), (0,), (0,), flags)

    bg_hit = fill_mask[1:-1, 1:-1]                # 255 = connected background
    fg_mask = np.where(bg_hit > 0, np.uint8(0), np.uint8(255))
    return fg_mask


def _clean_mask(fg_mask: np.ndarray) -> np.ndarray:
    """Fill holes, remove noise, erode 1 px to kill the white fringe."""
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Close small holes inside the object (e.g. gaps in chair legs)
    out = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    # Remove speckles outside
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k3, iterations=1)
    # Pull the boundary inward by 1 px to eliminate anti-alias white fringe
    out = cv2.erode(out, k3, iterations=1)
    return out


def _feather(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-blur the mask to create a smooth transparent edge."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _crop_rgba(rgba: np.ndarray, pad: int = 8) -> np.ndarray:
    """Crop to the bounding box of non-transparent pixels, plus padding."""
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return rgba
    h, w = rgba.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return rgba[y0:y1, x0:x1]


def remove_background(raw: np.ndarray, tolerance: int = 30) -> np.ndarray:
    """Return RGBA uint8 with background removed and edges feathered.

    Handles:
    - BGRA input  → feather existing alpha
    - BGR input   → flood-fill BG removal → feather
    - Grayscale   → convert to RGB then same flow
    """
    # ── Already has alpha ───────────────────────────────────────────────────
    if raw.ndim == 3 and raw.shape[2] == 4:
        rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
        k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.erode(rgba[:, :, 3], k3, iterations=1)
        rgba[:, :, 3] = _feather(alpha, sigma=1.5)
        return rgba

    # ── Grayscale → RGB ─────────────────────────────────────────────────────
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    fg_mask = _flood_fill_fg_mask(rgb, tolerance)
    fg_mask = _clean_mask(fg_mask)
    alpha   = _feather(fg_mask, sigma=2.0)

    return np.dstack([rgb, alpha])


# ---------------------------------------------------------------------------
# Manifest helpers (incremental tracking)
# ---------------------------------------------------------------------------

def _load_manifest() -> dict[str, float]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(manifest: dict[str, float]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _source_changed(src: Path, manifest: dict[str, float]) -> bool:
    """Return True if the source has changed since last processing."""
    recorded_mtime = manifest.get(src.name)
    if recorded_mtime is None:
        return True
    return abs(src.stat().st_mtime - recorded_mtime) > 0.5


def _output_path(src: Path) -> Path:
    """Destination PNG path for a given source file."""
    return OUTPUT_DIR / (src.stem + ".png")


# ---------------------------------------------------------------------------
# Catalog manifest helpers (clean labels for the app dropdown)
# ---------------------------------------------------------------------------

_COMPOUND_SPLITS = {
    "tvstand":   "TV Stand",
    "wallshelf": "Wall Shelf",
    "floorlamp": "Floor Lamp",
    "armchair":  "Arm Chair",
    "sideboard": "Sideboard",
    "coffeetable": "Coffee Table",
    "nightstand": "Nightstand",
    "bookshelf": "Bookshelf",
    "bookcase":  "Bookcase",
    "cupboard":  "Cupboard",
}


def _clean_label(stem: str) -> str:
    """Turn a filename stem into a human-readable label."""
    # Strip common noise suffixes
    for suffix in ("_img", "_image", "-img", "-image", "_p", "-p"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Replace separators with spaces
    cleaned = re.sub(r"[_\-]+", " ", stem).strip().lower()
    # Single-word compound lookup (e.g. "tvstand" → "TV Stand")
    single = cleaned.replace(" ", "")
    if single in _COMPOUND_SPLITS:
        return _COMPOUND_SPLITS[single]
    return cleaned.title()


def _aliases(label: str) -> list[str]:
    base = label.lower()
    out  = {base, base.replace(" ", "")}
    rules = {
        "tv stand":   {"tvstand", "media console", "console"},
        "floor lamp": {"lamp", "standing lamp", "floorlamp"},
        "wall shelf": {"shelf", "floating shelf", "wallshelf"},
        "cupboard":   {"cabinet", "storage cabinet"},
        "sofa":       {"couch", "loveseat"},
        "chair":      {"accent chair"},
        "bookshelf":  {"bookcase", "shelf"},
        "table":      {"side table", "coffee table"},
    }
    for key, extras in rules.items():
        if key in base:
            out |= extras
    return sorted(out)


def _write_catalog_manifest(out_dir: Path) -> None:
    """Write a catalog_manifest.json with clean labels into out_dir."""
    items = []
    seen_labels: dict[str, int] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        if p.name.startswith("."):
            continue
        base_label = _clean_label(p.stem)
        # Deduplicate: "Wall Shelf", "Wall Shelf 2", etc.
        if base_label in seen_labels:
            seen_labels[base_label] += 1
            label = f"{base_label} {seen_labels[base_label]}"
        else:
            seen_labels[base_label] = 1
            label = base_label
        items.append({
            "filename": p.name,
            "label":    label,
            "aliases":  _aliases(base_label),
            "source":   "Furniture_images_processed",
        })

    manifest_path = out_dir / "catalog_manifest.json"
    manifest_path.write_text(
        json.dumps({"items": sorted(items, key=lambda x: x["label"].lower())}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote catalog manifest: {manifest_path}  ({len(items)} items)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Process furniture images: BG removal + feathering")
    parser.add_argument("--source",    type=Path, default=SOURCE_DIR,
                        help=f"Folder with raw furniture images (default: {SOURCE_DIR})")
    parser.add_argument("--output",    type=Path, default=OUTPUT_DIR,
                        help=f"Folder for processed PNGs (default: {OUTPUT_DIR})")
    parser.add_argument("--force",     action="store_true",
                        help="Reprocess all images even if unchanged")
    parser.add_argument("--prune",     action="store_true",
                        help="Remove processed PNGs whose source no longer exists")
    parser.add_argument("--tolerance", type=int, default=30,
                        help="BG flood-fill colour tolerance 0-255 (default: 30)")
    args = parser.parse_args()

    src_dir: Path = args.source
    out_dir: Path = args.output

    if not src_dir.exists():
        sys.exit(f"Source folder not found: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {} if args.force else _load_manifest()

    sources = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        and p.name != MANIFEST_PATH.name
    )

    # ── Prune orphaned outputs ───────────────────────────────────────────────
    if args.prune:
        source_stems = {p.stem for p in sources}
        removed = 0
        for out_file in list(out_dir.glob("*.png")):
            if out_file.stem not in source_stems:
                out_file.unlink()
                manifest.pop(out_file.stem + ".png", None)
                print(f"PRUNE {out_file.name}")
                removed += 1
        if removed:
            print(f"Pruned {removed} orphaned file(s).")

    # ── Process ─────────────────────────────────────────────────────────────
    processed = skipped = failed = 0

    for src in sources:
        out_path = _output_path(src)

        # Skip if unchanged
        if not args.force and not _source_changed(src, manifest) and out_path.exists():
            skipped += 1
            continue

        raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"FAIL  {src.name} — could not read")
            failed += 1
            continue

        try:
            rgba = remove_background(raw, tolerance=args.tolerance)
            rgba = _crop_rgba(rgba)

            # Sanity check: if less than 5% of pixels are foreground, warn
            fg_ratio = (rgba[:, :, 3] > 10).sum() / max(rgba[:, :, 3].size, 1)
            if fg_ratio < 0.05:
                print(f"WARN  {src.name} — only {fg_ratio:.1%} foreground pixels; "
                      "try --tolerance with a higher value")

            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(out_path), bgra)
            manifest[src.name] = src.stat().st_mtime
            print(f"OK    {src.name:40s}  →  {out_path.name}  "
                  f"({rgba.shape[1]}×{rgba.shape[0]}, fg={fg_ratio:.0%})")
            processed += 1

        except Exception as exc:
            print(f"FAIL  {src.name} — {exc}")
            failed += 1

    _save_manifest(manifest)

    # ── Write catalog_manifest.json with clean labels ────────────────────────
    _write_catalog_manifest(out_dir)

    print()
    print(f"Done.  processed={processed}  skipped={skipped}  failed={failed}")
    print(f"Output: {out_dir}")
    if processed or skipped:
        print("Restart the app or click '↺ Refresh' in the Add tab to load updated images.")


if __name__ == "__main__":
    main()
