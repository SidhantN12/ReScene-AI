"""
Sync Furniture_images/catalog_manifest.json with the files on disk.

This keeps the app's external furniture catalog easy to maintain:
    - adds new image files to the manifest
    - keeps existing custom labels/aliases when present
    - optionally removes manifest entries for missing files

Usage:
    python sync_furniture_manifest.py
    python sync_furniture_manifest.py --prune
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog_utils import (
    CatalogItem,
    SUPPORTED_CATALOG_EXTS,
    load_catalog_items,
    save_catalog_manifest,
)

CATALOG_DIR = Path(__file__).parent / "Furniture_images"


def _clean_stem_label(stem: str) -> str:
    stem = stem.strip()
    for suffix in ("_img", "_image", "-img", "-image"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ").replace("-", " ").strip()


def _default_aliases(label: str) -> list[str]:
    base = label.lower()
    aliases = {base}

    compact = base.replace(" ", "")
    if compact != base:
        aliases.add(compact)

    if "tv stand" in base:
        aliases.update({"tvstand", "media console", "console"})
    if "floor lamp" in base:
        aliases.update({"lamp", "standing lamp", "floorlamp"})
    if "wall shelf" in base:
        aliases.update({"shelf", "floating shelf", "wallshelf"})
    if "cupboard" in base:
        aliases.update({"cabinet", "storage cabinet"})
    if "sofa" in base:
        aliases.update({"couch", "loveseat"})
    if "chair" in base:
        aliases.update({"accent chair"})
    if "bookshelf" in base:
        aliases.update({"bookcase", "shelf"})
    if "table" in base:
        aliases.update({"side table", "coffee table"})

    return sorted(aliases)


def _infer_item(path: Path) -> CatalogItem:
    label = _clean_stem_label(path.stem).title()
    return CatalogItem(
        filename=path.name,
        label=label,
        aliases=_default_aliases(label),
        source="Furniture_images",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Furniture_images manifest")
    parser.add_argument("--prune", action="store_true",
                        help="Remove manifest entries whose files no longer exist")
    args = parser.parse_args()

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    existing = {item.filename: item for item in load_catalog_items(CATALOG_DIR)}
    files = {
        path.name: path
        for path in sorted(CATALOG_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_CATALOG_EXTS
    }

    merged: dict[str, CatalogItem] = {}
    for filename, path in files.items():
        if filename in existing:
            item = existing[filename]
            if not item.source:
                item.source = "Furniture_images"
            merged[filename] = item
        else:
            merged[filename] = _infer_item(path)

    if not args.prune:
        for filename, item in existing.items():
            if filename not in merged:
                merged[filename] = item

    manifest_path = save_catalog_manifest(CATALOG_DIR, list(merged.values()))
    print(f"Synced {len(merged)} manifest entries.")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
