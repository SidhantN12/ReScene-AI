"""
ReScene AI — central configuration.

All model paths, device settings, VRAM budgets, and inference hyper-parameters
live here so every module imports from a single source of truth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _configure_windows_cuda_dll_path() -> None:
    """Ensure Conda CUDA DLLs are discoverable on Windows.

    Some launch paths (notably VS Code / external terminals) start Python with
    the environment's interpreter but without prepending `<env>\\bin` to PATH.
    ZoeDepth's NVRTC JIT path then fails to locate `nvrtc-builtins64_121.dll`
    even though it is installed inside the Conda env.
    """
    if os.name != "nt":
        return

    env_root = Path(os.environ.get("CONDA_PREFIX", Path(sys.executable).resolve().parent))
    candidates = [env_root / "bin", env_root / "Library" / "bin"]

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in reversed([str(p) for p in candidates if p.exists()]):
        if dll_dir not in path_entries:
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(dll_dir)
        except (AttributeError, FileNotFoundError):
            pass


_configure_windows_cuda_dll_path()

import torch

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent.resolve()
MODELS_DIR: Path = BASE_DIR / "models"
DATA_DIR: Path = BASE_DIR / "data"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

# Ensure runtime dirs exist
for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@dataclass
class DeviceConfig:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # Hard ceiling — model_manager enforces this before any load
    vram_limit_gb: float = 8.0
    # Use fp16 on GPU to halve VRAM; fp32 on CPU
    dtype: torch.dtype = field(
        default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32
    )

    def vram_limit_bytes(self) -> int:
        return int(self.vram_limit_gb * 1024 ** 3)


# ---------------------------------------------------------------------------
# Model paths  (weights land in models/ after download_models.py runs)
# ---------------------------------------------------------------------------

@dataclass
class ModelPaths:
    # SAM ViT-H
    sam_checkpoint: Path = field(default_factory=lambda: MODELS_DIR / "sam_vit_h_4b8939.pth")
    sam_model_type: str = "vit_h"

    # ZoeDepth — NK = joint NYUv2+KITTI, best for indoor scenes
    zoedepth_checkpoint: Path = field(default_factory=lambda: MODELS_DIR / "ZoeD_M12_NK.pt")
    zoedepth_model_name: str = "ZoeD_NK"

    # LaMa inpainting (preferred over MAT for speed).
    # Current runtime uses simple-lama-inpainting's own cache; this path is kept
    # only as a conventional local location if you later choose to mirror weights.
    lama_dir: Path = field(default_factory=lambda: MODELS_DIR / "lama")

    # MAT inpainting (fallback / high-quality mode)
    mat_checkpoint: Path = field(default_factory=lambda: MODELS_DIR / "mat" / "places_512_G.pkl")
    mat_image_size: int = 512

    # ST-GAN perspective warp
    stgan_checkpoint: Path = field(default_factory=lambda: MODELS_DIR / "stgan" / "checkpoint.pth")

    # ARShadowGAN contact-shadow generation
    arshadowgan_checkpoint: Path = field(
        default_factory=lambda: MODELS_DIR / "arshadowgan" / "net_G.pth"
    )

    # SPADE / GauGAN2 style transfer
    spade_checkpoint: Path = field(
        default_factory=lambda: MODELS_DIR / "spade" / "latest_net_G.pth"
    )
    spade_config: Path = field(default_factory=lambda: MODELS_DIR / "spade" / "opt.pkl")

    # iHarmony4 harmonization
    iharmony4_checkpoint: Path = field(
        default_factory=lambda: MODELS_DIR / "iharmony4" / "iHarmony4_model.pth"
    )


# ---------------------------------------------------------------------------
# Per-model VRAM estimates (GB) — used by model_manager for headroom checks
# ---------------------------------------------------------------------------

MODEL_VRAM_GB: dict[str, float] = {
    "sam": 3.5,          # ViT-H image encoder in fp16; decoder+prompt encoder fp32
    "zoedepth": 1.2,     # ZoeD_NK; inference with autocast fp16 → ~0.7 GB
    "clip": 0.6,         # ViT-B/32; loaded on CPU or GPU, fp32
    "lama": 1.0,
    "mat": 2.0,
    "stgan": 0.8,
    "arshadowgan": 0.5,
    "spade": 3.5,
    "iharmony4": 0.4,
}


# ---------------------------------------------------------------------------
# Inference hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    # SAM
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    sam_box_nms_thresh: float = 0.7

    # Inpainting
    lama_pad_modulo: int = 8
    inpaint_size: int = 512         # resize before inpaint, restore after

    # General image processing
    max_image_size: int = 1024      # long-edge cap before pipeline

    # Style transfer
    style_image_size: int = 512
    num_style_epochs: int = 100     # iterations for optimization-based methods

    # Harmonization
    harmony_image_size: int = 256

    # Depth-guided scaling: meters per pixel at reference depth
    depth_reference_m: float = 2.5


# ---------------------------------------------------------------------------
# Style catalogue (maps UI name → reference image path)
# ---------------------------------------------------------------------------

STYLES: dict[str, Path] = {
    "Scandinavian": DATA_DIR / "styles" / "scandinavian.jpg",
    "Industrial": DATA_DIR / "styles" / "industrial.jpg",
    "Bohemian": DATA_DIR / "styles" / "bohemian.jpg",
    "Mid-Century Modern": DATA_DIR / "styles" / "midcentury.jpg",
    "Minimalist": DATA_DIR / "styles" / "minimalist.jpg",
    "Coastal": DATA_DIR / "styles" / "coastal.jpg",
}


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    debug: bool = False
    max_upload_size_mb: int = 20


# ---------------------------------------------------------------------------
# Scene understanding (Phase 8)
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    # CLIP zero-shot classifier for mask labelling
    clip_model_name: str = "openai/clip-vit-base-patch32"
    # Max entries in the SceneContext LRU cache
    scene_cache_max: int = 4
    # Whether to build SceneContext on upload (set False to disable for speed)
    build_on_upload: bool = True


# ---------------------------------------------------------------------------
# Performance / profiling
# ---------------------------------------------------------------------------

@dataclass
class PerformanceConfig:
    # Enable torch.compile() on PyTorch ≥ 2.0 (can reduce inference latency ~15–25 %)
    # Disabled by default: first-call compilation adds ~30 s overhead on some backends.
    use_torch_compile: bool = False
    compile_backend: str = "inductor"   # "inductor" | "eager" | "aot_eager"
    # Collect per-stage wall-clock time and VRAM snapshots in compound()
    profile_stages: bool = True


# ---------------------------------------------------------------------------
# Root config object
# ---------------------------------------------------------------------------

@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    models: ModelPaths = field(default_factory=ModelPaths)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    app: AppConfig = field(default_factory=AppConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


# Singleton — import this everywhere
config = Config()
