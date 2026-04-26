"""
Catalog helpers for ReScene AI.

The Add tab and NL parser work best when furniture assets carry stable labels
and aliases instead of relying only on PNG filenames. This module provides a
small manifest format around data/catalog/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

MANIFEST_NAME = "catalog_manifest.json"
SUPPORTED_CATALOG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CatalogItem:
    filename: str
    label: str
    aliases: list[str]
    source: str = ""


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "item"


def load_catalog_items(catalog_dir: Path) -> list[CatalogItem]:
    manifest_path = catalog_dir / MANIFEST_NAME
    items_by_file: dict[str, CatalogItem] = {}

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for raw in data.get("items", []):
            filename = str(raw.get("filename", "")).strip()
            if not filename:
                continue
            items_by_file[filename] = CatalogItem(
                filename=filename,
                label=str(raw.get("label", Path(filename).stem.replace("_", " ").title())).strip(),
                aliases=[str(a).strip() for a in raw.get("aliases", []) if str(a).strip()],
                source=str(raw.get("source", "")).strip(),
            )

    for path in sorted(catalog_dir.iterdir()) if catalog_dir.exists() else []:
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if path.suffix.lower() not in SUPPORTED_CATALOG_EXTS:
            continue
        if path.name not in items_by_file:
            items_by_file[path.name] = CatalogItem(
                filename=path.name,
                label=path.stem.replace("_", " ").title(),
                aliases=[],
            )

    return list(items_by_file.values())


def save_catalog_manifest(catalog_dir: Path, items: list[CatalogItem]) -> Path:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = catalog_dir / MANIFEST_NAME
    payload = {
        "items": [asdict(item) for item in sorted(items, key=lambda x: x.label.lower())]
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def build_choice_map(items: list[CatalogItem]) -> dict[str, str]:
    return {item.label: item.filename for item in items}


def build_nl_lookup(items: list[CatalogItem]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in items:
        terms = [item.label, Path(item.filename).stem.replace("_", " ")] + item.aliases
        for term in terms:
            norm = " ".join(term.lower().replace("_", " ").split())
            if norm:
                lookup[norm] = item.filename
    return lookup
