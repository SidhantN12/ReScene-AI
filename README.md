# ReScene AI

**Intelligent room rearrangement, furniture placement, and restyling from a single photograph.**

Built with SAM · ZoeDepth · LaMa · ARShadowGAN · Reinhard LAB style transfer · Poisson harmonization.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Environment Setup](#environment-setup)
3. [Download Model Weights](#download-model-weights)
4. [Generate Assets](#generate-assets)
5. [Running the App](#running-the-app)
6. [Feature Tour](#feature-tour)
7. [Demo Guide](#demo-guide)
8. [Performance Notes](#performance-notes)
9. [Project Structure](#project-structure)

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 (CUDA 12.1 recommended) |
| CUDA | 12.1+ (CPU fallback available, much slower) |
| VRAM | ≥ 6 GB (8 GB recommended) |
| RAM | ≥ 16 GB |
| Disk | ~10 GB for model weights |

---

## Environment Setup

```bash
# Create and activate the conda environment
conda create -n torch-cu121 python=3.11 -y
conda activate torch-cu121

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install gradio opencv-python-headless pillow numpy scipy segment-anything \
            simple-lama-inpainting

# Clone the repo (if not already)
git clone <repo-url>
cd rescene-ai
```

---

## Download Model Weights

```bash
conda activate torch-cu121
python download_models.py
```

This downloads:
- `models/sam_vit_h_4b8939.pth` — SAM ViT-H (~2.4 GB)
- `models/ZoeD_M12_NK.pt` — ZoeDepth-NK (~400 MB)

LaMa is handled separately:
- `simple-lama-inpainting` downloads `big-lama` into its own cache on first use.
- The app falls back to OpenCV inpainting if LaMa is unavailable.

---

## Generate Assets

**Style reference palettes** (required for Restyle tab):
```bash
python generate_styles.py
```
Writes 6 × 512×512 JPEG palettes to `data/styles/`.

**Furniture catalog** (required for Add tab):
```bash
python generate_catalog.py
```
Writes 14 furniture PNGs to `data/catalog/`.

To replace the synthetic catalog with real furniture cutouts:
```bash
python fetch_catalog_assets.py --dataset-dir "C:\path\to\furniture_dataset" --clear
```
or use a URL/page manifest:
```bash
python fetch_catalog_assets.py --sources catalog_sources.example.json --clear
```
This writes normalized PNGs plus `data/catalog/catalog_manifest.json`, which the
Add tab and NL parser read automatically for labels and aliases.

If you use the local `Furniture_images/` folder instead, refresh its manifest with:
```bash
python sync_furniture_manifest.py
```

**Demo examples** (optional, populates the Examples gallery):
```bash
python generate_examples.py
```
Place your own room photos in `data/examples/input/` first, or let the script
generate synthetic rooms automatically.

---

## Running the App

```bash
conda activate torch-cu121
python app.py
```

Open `http://localhost:7860` in your browser.

To expose publicly (e.g. for a demo):
```python
# config.py → AppConfig
share: bool = True
```

---

## Feature Tour

### Remove tab
1. Upload a room photo.
2. Click the object to remove — SAM highlights it with a red mask.
3. Click again to refine, or press **Remove Object**.
4. Optionally choose a design style to apply after removal.

### Move tab
1. Upload a room photo.
2. Click the object in the **left panel** — SAM highlights it.
3. Click the destination in the **right panel** — a blue crosshair marks it.
4. Press **Move Object**.

### Add tab
1. Upload a room photo; click the insertion point.
2. Pick furniture from the 14-item catalog **or** upload a PNG.
3. Press **Add Furniture** — ZoeDepth scales it to match room depth.

### Restyle tab
Apply one of six design palettes (Scandinavian, Industrial, Bohemian,
Mid-Century Modern, Minimalist, Coastal) using Reinhard LAB colour statistics.
Drag the **before/after slider** to compare.

### Compound Operations tab

**Natural Language mode** (new in Phase 7):
```
add a sofa on the left, then make it scandinavian
place an armchair in the corner, apply coastal style
add a lamp, restyle as industrial
```
Press **Preview parse** to see the interpreted instruction list, then
**Run NL Compound** to execute. A stage gallery shows each intermediate result.

> Note: `remove` and `move` operations require clicking to select an object and
> cannot be automated from text alone. Use the dedicated tabs or JSON mode for these.

**JSON mode**:
```json
[
  {"operation": "restyle", "style_name": "Bohemian"},
  {"operation": "add", "furniture_image": "...", "target_position": [400, 600]}
]
```

### Depth Preview tab
ZoeDepth-NK metric depth map visualisation — warm colours = near, cool = far.

### Examples tab
Pre-computed before/after demo pairs. Populate with `python generate_examples.py`.

---

## Demo Guide

**Recommended demo sequence** (works even without ML weights for style/harmonization):

1. Open the **Restyle** tab → upload any room photo → choose *Scandinavian* → show the slider.
2. Open **Compound** → NL tab → type `add a sofa on the left, make it coastal` → run.
3. Open **Depth Preview** → show the depth heatmap.
4. Show the **Examples** gallery for backup visuals.

**With full model weights:**
- Remove tab: demonstrate click-to-select SAM segmentation, then inpainting.
- Move tab: pick a chair, drag it to another corner.

---

## Performance Notes

| Stage | Approximate time (RTX 3060, fp16) |
|---|---|
| SAM segmentation | 1–3 s |
| ZoeDepth estimation | 0.5–1 s |
| LaMa inpainting | 2–5 s |
| Restyle (Reinhard LAB) | < 0.1 s |
| Harmonization (Poisson) | < 0.1 s |
| Shadow generation | 0.5–1 s |
| **Full add/move pipeline** | **~5–10 s** |

### Enabling torch.compile (PyTorch ≥ 2.0)

In `config.py`:
```python
performance: PerformanceConfig = field(default_factory=lambda: PerformanceConfig(
    use_torch_compile=True,
    compile_backend="inductor",
))
```
First call incurs a ~30 s compilation cost; subsequent calls are 15–25 % faster.

---

## Project Structure

```
rescene-ai/
├── app.py                    # Gradio UI (Phase 7)
├── config.py                 # Central config (models, inference, app, performance)
├── download_models.py        # Model weight downloader
├── generate_styles.py        # Generates data/styles/ reference palettes
├── generate_catalog.py       # Generates data/catalog/ furniture PNGs
├── generate_examples.py      # Generates data/examples/output/ demo pairs
├── pipeline/
│   ├── orchestrator.py       # High-level ops: remove/move/add/restyle/compound
│   ├── model_manager.py      # VRAM-aware load/unload
│   ├── segmentation.py       # SAM wrapper
│   ├── depth_estimation.py   # ZoeDepth wrapper
│   ├── inpainting.py         # LaMa / MAT wrapper
│   ├── perspective_warp.py   # ST-GAN perspective warp
│   ├── shadow_generation.py  # ARShadowGAN wrapper
│   ├── style_transfer.py     # Reinhard LAB colour transfer
│   ├── harmonization.py      # Poisson + adaptive LAB harmonization
│   └── nl_parser.py          # Natural-language → instruction list parser (Phase 7)
├── utils/
│   ├── image_utils.py
│   ├── depth_utils.py
│   └── composite_utils.py
├── data/
│   ├── styles/               # 6 × 512×512 style reference images
│   ├── catalog/              # 14 × furniture PNG cutouts
│   └── examples/
│       ├── input/            # Place your room photos here
│       └── output/           # Generated before/after demo pairs
├── models/                   # Downloaded weights go here
└── outputs/                  # Pipeline outputs saved here
```

---

*ReScene AI — Phase 7 · April 2026*
