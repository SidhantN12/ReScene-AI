"""
Fetch or import furniture assets into data/catalog/ for the Add tab.

Supports two practical paths:
1. Import from a local dataset directory.
2. Download from a JSON source manifest with direct image URLs or page scraping.

The script normalizes assets to transparent PNGs and writes
data/catalog/catalog_manifest.json so the app can show human labels and use
aliases in the Add tab / NL parser.

Usage:
    python fetch_catalog_assets.py --dataset-dir C:\\path\\to\\dataset --clear
    python fetch_catalog_assets.py --sources catalog_sources.example.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cv2
import numpy as np
import requests

from catalog_utils import CatalogItem, save_catalog_manifest, slugify

CATALOG_DIR = Path(__file__).parent / "data" / "catalog"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
USER_AGENT = "ReSceneAI/1.0 (+catalog importer)"


@dataclass
class PreparedAsset:
    filename: str
    label: str
    aliases: list[str]
    rgba: np.ndarray
    source: str = ""


def _clean_stem_label(stem: str) -> str:
    stem = stem.strip()
    stem = stem.removesuffix("_img").removesuffix("_image").removesuffix("-img").removesuffix("-image")
    return stem.replace("_", " ").replace("-", " ").strip()


def _image_to_rgba(src: np.ndarray, white_thresh: int = 240) -> np.ndarray:
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


def _decode_image_bytes(content: bytes) -> np.ndarray | None:
    arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _prepare_asset(
    raw: np.ndarray,
    *,
    slug: str,
    label: str,
    aliases: list[str],
    source: str = "",
    white_thresh: int = 240,
) -> PreparedAsset:
    rgba = _image_to_rgba(raw, white_thresh=white_thresh)
    rgba = _crop_to_alpha(rgba)
    return PreparedAsset(
        filename=f"{slug}.png",
        label=label.strip(),
        aliases=sorted({a.strip() for a in aliases if a and a.strip()}),
        rgba=rgba,
        source=source,
    )


def _save_assets(output_dir: Path, assets: list[PreparedAsset], clear: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()

    manifest_items: list[CatalogItem] = []
    for asset in assets:
        dest = output_dir / asset.filename
        bgra = cv2.cvtColor(asset.rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(dest), bgra)
        manifest_items.append(
            CatalogItem(
                filename=asset.filename,
                label=asset.label,
                aliases=asset.aliases,
                source=asset.source,
            )
        )
        print(f"OK   {asset.label} -> {asset.filename}")

    manifest_path = save_catalog_manifest(output_dir, manifest_items)
    print(f"\nWrote {len(assets)} assets to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


def _import_dataset_dir(dataset_dir: Path, white_thresh: int) -> list[PreparedAsset]:
    assets: list[PreparedAsset] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"SKIP {path.name} - unreadable")
            continue

        rel_parts = path.relative_to(dataset_dir).parts
        category = rel_parts[0].replace("_", " ").replace("-", " ") if len(rel_parts) > 1 else ""
        clean_stem = _clean_stem_label(path.stem)
        stem_label = clean_stem.title()
        label = stem_label if not category else f"{category.title()} - {stem_label}"
        aliases = [clean_stem, path.stem.replace("_", " "), category]
        slug = slugify(f"{category}_{clean_stem}" if category else clean_stem)
        assets.append(
            _prepare_asset(
                raw,
                slug=slug,
                label=label,
                aliases=aliases,
                source=str(path),
                white_thresh=white_thresh,
            )
        )
    return assets


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _download_image(session: requests.Session, url: str, timeout_s: int) -> np.ndarray | None:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return _decode_image_bytes(resp.content)


def _load_sources(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data)
    if not isinstance(items, list):
        raise ValueError("Source manifest must be a JSON list or an object with an 'items' list.")
    return items


def _fetch_direct_assets(
    entry: dict[str, Any],
    session: requests.Session,
    white_thresh: int,
    timeout_s: int,
) -> list[PreparedAsset]:
    url = str(entry["url"]).strip()
    raw = _download_image(session, url, timeout_s)
    if raw is None:
        return []

    label = str(entry.get("label") or Path(url).stem.replace("_", " ").title()).strip()
    slug = slugify(str(entry.get("slug") or label))
    aliases = list(entry.get("aliases", [])) + [label]
    return [
        _prepare_asset(
            raw,
            slug=slug,
            label=label,
            aliases=aliases,
            source=url,
            white_thresh=white_thresh,
        )
    ]


def _fetch_page_assets(
    entry: dict[str, Any],
    session: requests.Session,
    white_thresh: int,
    timeout_s: int,
) -> list[PreparedAsset]:
    from bs4 import BeautifulSoup

    page_url = str(entry["page_url"]).strip()
    selector = str(entry.get("image_selector") or "img").strip()
    url_attr = str(entry.get("url_attr") or "src").strip()
    limit = int(entry.get("limit", 10))
    slug_prefix = slugify(str(entry.get("slug_prefix") or entry.get("label_prefix") or "item"))
    label_prefix = str(entry.get("label_prefix") or slug_prefix.replace("_", " ").title()).strip()
    aliases = list(entry.get("aliases", []))

    html = session.get(page_url, timeout=timeout_s)
    html.raise_for_status()
    soup = BeautifulSoup(html.text, "html.parser")

    assets: list[PreparedAsset] = []
    seen_urls: set[str] = set()
    for idx, tag in enumerate(soup.select(selector), start=1):
        raw_url = tag.get(url_attr) or tag.get("src")
        if not raw_url:
            continue
        image_url = urljoin(page_url, raw_url)
        if image_url in seen_urls:
            continue
        seen_urls.add(image_url)

        raw = _download_image(session, image_url, timeout_s)
        if raw is None:
            continue

        alt = str(tag.get("alt") or "").strip()
        label = alt or f"{label_prefix} {idx}"
        slug = slugify(f"{slug_prefix}_{idx:02d}")
        assets.append(
            _prepare_asset(
                raw,
                slug=slug,
                label=label,
                aliases=aliases + [label_prefix, alt],
                source=image_url,
                white_thresh=white_thresh,
            )
        )
        if len(assets) >= limit:
            break
    return assets


def _fetch_sources(path: Path, white_thresh: int, timeout_s: int) -> list[PreparedAsset]:
    entries = _load_sources(path)
    session = _session()
    assets: list[PreparedAsset] = []
    for entry in entries:
        mode = str(entry.get("mode", "direct")).strip().lower()
        try:
            if mode == "direct":
                assets.extend(_fetch_direct_assets(entry, session, white_thresh, timeout_s))
            elif mode == "page":
                assets.extend(_fetch_page_assets(entry, session, white_thresh, timeout_s))
            else:
                print(f"SKIP unknown mode: {mode}")
        except Exception as exc:
            print(f"SKIP source entry failed ({mode}): {exc}")
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or import real furniture catalog assets")
    parser.add_argument("--dataset-dir", type=Path,
                        help="Local dataset directory. Subfolders become category aliases.")
    parser.add_argument("--sources", type=Path,
                        help="JSON file with direct download URLs or page scraping rules.")
    parser.add_argument("--output-dir", type=Path, default=CATALOG_DIR)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--white-thresh", type=int, default=240)
    parser.add_argument("--timeout-s", type=int, default=30)
    args = parser.parse_args()

    if not args.dataset_dir and not args.sources:
        raise SystemExit("Provide at least one of --dataset-dir or --sources.")

    assets: list[PreparedAsset] = []
    if args.dataset_dir:
        if not args.dataset_dir.exists():
            raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")
        assets.extend(_import_dataset_dir(args.dataset_dir, args.white_thresh))

    if args.sources:
        if not args.sources.exists():
            raise SystemExit(f"Source manifest not found: {args.sources}")
        assets.extend(_fetch_sources(args.sources, args.white_thresh, args.timeout_s))

    deduped: dict[str, PreparedAsset] = {}
    for asset in assets:
        deduped[asset.filename] = asset

    _save_assets(args.output_dir, list(deduped.values()), clear=args.clear)


if __name__ == "__main__":
    main()
