"""
ReScene AI - model weight downloader.

Run once before using the pipeline:
    python download_models.py

Downloads the required checkpoints into the local models/ directory and prints
an actionable summary for optional/manual integrations.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

_ZOEDEPTH_RELEASE = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    dest: Path
    url: str | None
    category: str
    required: bool
    sha256: str | None = None
    notes: str = ""


@dataclass
class DownloadResult:
    entry: ModelEntry
    status: str
    detail: str = ""


MODELS: list[ModelEntry] = [
    ModelEntry(
        name="SAM ViT-H",
        dest=Path("sam_vit_h_4b8939.pth"),
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        category="required",
        required=True,
        # Correct hash observed from the official SAM checkpoint download.
        sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        notes="Used by Remove, Move selection, scene understanding, and optional Restyle segmentation.",
    ),
    ModelEntry(
        name="ZoeDepth ZoeD_NK",
        dest=Path("ZoeD_M12_NK.pt"),
        # This URL is inferred from the official ZoeDepth v1.0 release asset naming:
        # ZoeD_M12_N.pt, ZoeD_M12_K.pt, ZoeD_M12_NK.pt.
        url=f"{_ZOEDEPTH_RELEASE}/ZoeD_M12_NK.pt",
        category="required",
        required=True,
        notes="Used by Add, Move scaling, Depth Preview, and scene understanding.",
    ),
    ModelEntry(
        name="LaMa big-lama",
        dest=Path("lama"),
        url=None,
        category="optional",
        required=False,
        notes=(
            "Not downloaded here. Runtime uses simple-lama-inpainting, which fetches "
            "big-lama into its own cache on first use. Remove/Move fall back to OpenCV if absent."
        ),
    ),
    ModelEntry(
        name="MAT Places-512",
        dest=Path("mat") / "places_512_G.pkl",
        url=None,
        category="manual",
        required=False,
        notes="Configured path exists for future use, but the current app does not load MAT weights.",
    ),
    ModelEntry(
        name="ST-GAN",
        dest=Path("stgan") / "checkpoint.pth",
        url=None,
        category="manual",
        required=False,
        notes="Current PerspectiveWarpWrapper is OpenCV-based and does not load ST-GAN weights.",
    ),
    ModelEntry(
        name="ARShadowGAN",
        dest=Path("arshadowgan") / "net_G.pth",
        url=None,
        category="manual",
        required=False,
        notes="Current ShadowGenerationWrapper is a no-weight placeholder.",
    ),
    ModelEntry(
        name="SPADE / GauGAN2",
        dest=Path("spade") / "latest_net_G.pth",
        url=None,
        category="manual",
        required=False,
        notes="Current StyleTransferWrapper uses no-weight Reinhard LAB transfer instead.",
    ),
    ModelEntry(
        name="iHarmony4 CDTNet",
        dest=Path("iharmony4") / "iHarmony4_model.pth",
        url=None,
        category="manual",
        required=False,
        notes="Current HarmonizationWrapper uses no-weight LAB + Poisson blending.",
    ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB", end="")
    else:
        print(f"\r  Downloaded {downloaded/1e6:.1f} MB", end="")


def _download_binary(entry: ModelEntry) -> DownloadResult:
    dest = MODELS_DIR / entry.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.is_dir():
        return DownloadResult(entry, "ok", "directory already present")

    if dest.exists():
        if entry.sha256:
            actual = sha256_file(dest)
            if actual == entry.sha256:
                return DownloadResult(entry, "ok", "already present, checksum matches")
            logger.warning("Checksum mismatch for %s - re-downloading.", entry.name)
            logger.warning("Expected: %s", entry.sha256)
            logger.warning("Got:      %s", actual)
            dest.unlink()
        else:
            return DownloadResult(entry, "ok", "already present")

    if not entry.url:
        return DownloadResult(entry, entry.category, entry.notes)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        logger.info("DL   %s", entry.name)
        logger.info("     %s", entry.url)
        urllib.request.urlretrieve(entry.url, tmp, reporthook=_progress_hook)
        print()

        if entry.sha256:
            actual = sha256_file(tmp)
            if actual != entry.sha256:
                tmp.unlink(missing_ok=True)
                return DownloadResult(
                    entry,
                    "failed",
                    f"checksum mismatch after download (expected {entry.sha256}, got {actual})",
                )

        shutil.move(str(tmp), str(dest))
        return DownloadResult(entry, "ok", f"saved to {dest.relative_to(MODELS_DIR)}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return DownloadResult(entry, "failed", str(exc))


def _summarise(results: list[DownloadResult]) -> int:
    buckets: dict[str, list[DownloadResult]] = {
        "required": [],
        "optional": [],
        "manual": [],
        "failed": [],
    }

    for result in results:
        if result.status == "failed":
            buckets["failed"].append(result)
        elif result.entry.category in buckets:
            buckets[result.entry.category].append(result)

    def _print_bucket(title: str, items: list[DownloadResult]) -> None:
        print(f"\n{title}")
        if not items:
            print("  - none")
            return
        for item in items:
            suffix = f" ({item.detail})" if item.detail else ""
            print(f"  - {item.entry.name}{suffix}")

    print("\n=== Summary ===")
    _print_bucket("REQUIRED models", buckets["required"])
    _print_bucket("OPTIONAL models", buckets["optional"])
    _print_bucket("MANUAL models", buckets["manual"])
    _print_bucket("FAILED models", buckets["failed"])

    if buckets["manual"]:
        print("\nManual/inactive notes:")
        for item in buckets["manual"]:
            print(f"  - {item.entry.name}: {item.entry.notes}")

    if buckets["optional"]:
        print("\nOptional notes:")
        for item in buckets["optional"]:
            print(f"  - {item.entry.name}: {item.entry.notes}")

    required_failures = [r for r in buckets["failed"] if r.entry.required]
    if required_failures:
        print("\nRequired downloads failed:")
        for item in required_failures:
            print(f"  - {item.entry.name}: {item.detail}")
        return 1

    if buckets["failed"]:
        print("\nNon-required downloads failed, but required app models are available.")

    print("\nRequired model setup completed.")
    return 0


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model download directory: %s", MODELS_DIR)

    results: list[DownloadResult] = []
    for entry in MODELS:
        result = _download_binary(entry)
        results.append(result)

        if result.status == "ok":
            logger.info("OK   %s - %s", entry.name, result.detail)
        elif result.status == "failed":
            logger.error("FAIL %s - %s", entry.name, result.detail)
        else:
            logger.warning("SKIP %s - %s", entry.name, result.detail)

    sys.exit(_summarise(results))


if __name__ == "__main__":
    main()
