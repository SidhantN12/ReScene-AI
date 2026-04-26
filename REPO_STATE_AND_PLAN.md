# ReScene AI: Current Repo State and Improvement Plan

## Checkpoint

- Current lightweight code checkpoint:
  - `checkpoints/pre-architecture-20260426-153507/`
- This snapshot includes:
  - UI and pipeline code
  - config and downloader scripts
  - catalog scripts and manifest
  - tests
- This snapshot excludes:
  - `models/`
  - `outputs/`
  - `__pycache__/`
  - full `Furniture_images/` asset files
  - runtime data assets

Note: this repository is not currently a Git repository, so there is no branch/commit history yet.

## Repo Overview

ReScene AI is a Gradio-based room-editing prototype. It combines segmentation, monocular depth estimation, inpainting, compositing, and lightweight style transfer into a single interactive UI.

The app is strongest as a proof-of-concept orchestration layer. It is not yet a robust scene-understanding or photorealistic editing system.

## Top-Level Structure

- `app.py`
  - Main Gradio UI.
  - Tabs: `Remove`, `Move`, `Add`, `Mood Shift`, `Compound Operations`, `Depth Preview`, `Examples`.
  - Handles click interactions, catalog selection, compound parsing previews, and before/after displays.

- `config.py`
  - Central config for model paths, runtime device selection, VRAM limits, inference sizes, and style presets.
  - Includes Windows CUDA DLL path fix for the Conda environment.

- `download_models.py`
  - Downloads required weights and reports required/manual/optional/failed status.
  - Current required path is based around `SAM` and `ZoeDepth-NK`.

- `pipeline/`
  - Core pipeline wrappers and orchestrator logic.

- `utils/`
  - Image/compositing/depth helper utilities.

- `Furniture_images/`
  - External local furniture catalog directory currently preferred by the app over `data/catalog/`.
  - Uses `catalog_manifest.json` for display labels and aliases.

- `data/`
  - Style images, example assets, and other runtime data.

- Catalog tooling
  - `catalog_utils.py`
  - `prepare_catalog.py`
  - `fetch_catalog_assets.py`
  - `sync_furniture_manifest.py`

## Pipeline Modules

- `pipeline/model_manager.py`
  - Sequential, VRAM-aware model loading and unloading.
  - Only one heavyweight model is meant to stay resident at a time.

- `pipeline/orchestrator.py`
  - Main high-level operations:
    - `select_object`
    - `remove`
    - `move`
    - `add`
    - `restyle`
    - `understand_scene`
    - `compound`

- `pipeline/segmentation.py`
  - SAM-based object selection and segmentation.

- `pipeline/depth_estimation.py`
  - ZoeDepth-based monocular depth estimation.
  - Adjusted for current checkpoint compatibility and Windows CUDA edge cases.

- `pipeline/inpainting.py`
  - Inpainting wrapper.
  - Currently oriented around the `simple-lama-inpainting` path and fallback behavior.

- `pipeline/perspective_warp.py`
  - Current object warp / resize / perspective approximation.
  - This is not a full generative placement model.

- `pipeline/shadow_generation.py`
  - Lightweight or placeholder shadow generation.

- `pipeline/harmonization.py`
  - Lightweight harmonization/blending pass.

- `pipeline/style_transfer.py`
  - Current "Mood Shift" implementation.
  - This is mostly LAB/statistical color transfer, not real generative interior redesign.

- `pipeline/nl_parser.py`
  - Natural-language instruction parsing for compound operations.
  - Supports multi-add parsing, aliases, and heuristic position hints.

## What the App Can Actually Do Right Now

### Remove

- Uses SAM click-to-select.
- Uses inpainting to fill the removed region.
- Works best for small, isolated objects with simple backgrounds.

### Move

- Selects an existing object with SAM.
- Estimates scale with ZoeDepth.
- Removes from source via inpainting.
- Re-inserts via perspective warp, color matching, shadow, and harmonization.
- Works, but visual quality is weak on complex scenes.

### Add

- Adds local catalog furniture or an uploaded furniture asset.
- Uses depth-based scale plus a manual size multiplier.
- Crops transparent/white padding before scaling.
- Better than before, but still not truly scene-aware.

### Mood Shift

- Renamed in the UI from `Restyle`.
- Performs palette/mood transfer, not structural redesign.
- Should be described as color or tone adaptation, not full room restyling.

### Compound Operations

- Supports NL and JSON instruction chains.
- Multi-add parsing works better than before.
- Placement is still heuristic and can mix up destinations or overlap objects.

### Depth Preview

- Shows a ZoeDepth heatmap and is one of the more reliable technical tabs.

## Current Technical Reality

### Real / actively used models

- `SAM`
- `ZoeDepth-NK`
- `simple-lama-inpainting` path

### Real but lightweight / CV-based implementations

- perspective warp
- color matching
- harmonization
- shadow approximation
- mood transfer

### Placeholder or effectively inactive model references

- `MAT`
- `ST-GAN`
- `ARShadowGAN`
- `SPADE / GauGAN`
- `iHarmony4 CDTNet`

Some of those names exist in config or wrapper structure, but the current user-visible behavior is not equivalent to integrating the original research models end to end.

## Current Shortcomings

### Scene understanding

- No persistent semantic scene representation is built on room upload.
- No floor-plane / wall-plane model is retained for downstream actions.
- No occupancy map is used during multi-object compound placement.
- No support-surface reasoning such as:
  - on floor
  - against wall
  - on table
  - next to chair

### Add / Move placement

- NL placement resolves to heuristic 2D anchor percentages.
- Placement is not validated against free floor space.
- Multiple added objects can overlap or drift too close to image edges.
- The app does not reserve occupied regions after each compound placement step.

### Inpainting

- Large removals still degrade quickly.
- Wall/floor boundaries are not explicitly respected.
- The current backend is not strong enough to produce consistently professional results for furniture removal.

### Move realism

- Move is visually the weakest major feature.
- It combines removal, scaling, warping, insertion, shadowing, and harmonization without a true geometric scene model.
- The current pipeline is fragile on beds, trim, corners, and cluttered scenes.

### Mood Shift

- It mostly looks like color grading because that is effectively what it is.
- It does not change furniture style, materials, layout, or architecture.

### Catalog placement quality

- Better now due to alpha/white-background cropping and manual size multiplier.
- Still limited by:
  - product cutout quality
  - lack of scene semantics
  - limited physical-size reasoning

## Improvements Already Made Recently

- Fixed SAM checkpoint SHA mismatch handling in the downloader.
- Standardized ZoeDepth to `ZoeD_M12_NK.pt`.
- Made ZoeDepth loading more tolerant to checkpoint/key mismatches.
- Fixed Windows CUDA DLL discovery for ZoeDepth on Conda.
- Added fallback behavior for CUDA-specific ZoeDepth runtime issues.
- Installed and validated SAM usage in the target environment.
- Added external furniture catalog support from `Furniture_images/`.
- Added catalog manifest support and alias mapping.
- Added catalog prep/import/sync scripts.
- Renamed UI `Restyle` to `Mood Shift`.
- Improved NL parsing for multi-add commands and position hints.
- Added add-item alpha/white-padding cropping and a manual size multiplier.

## Proposed Next Improvements

### 1. Add a cached scene-understanding layer on room upload

Minimum useful version:

- run segmentation and depth once on upload
- derive a coarse floor region
- derive coarse wall region
- derive occupied-object mask
- build safe candidate placement anchors

This would improve:

- `Add`
- `Compound Operations`
- multi-object placement
- location resolution for NL commands

### 2. Build an occupancy-aware placement planner

- convert `left`, `right`, `corner`, `center` into valid scene anchors
- avoid placing objects outside the usable floor area
- avoid collisions with existing and newly placed objects
- reserve occupied regions after each compound step

### 3. Upgrade inpainting

- replace or augment the current inpainting backend with a stronger model
- use scene structure to preserve floor/wall boundaries
- make remove and move source cleanup materially better

This is likely the highest-value visual-quality improvement.

### 4. Improve Add and Move with floor-aware placement

- infer base contact with the floor
- use more believable default physical-size priors per furniture class
- improve occlusion and contact-shadow handling

### 5. Reposition or reduce weak features in the demo

If presentation quality matters more than breadth:

- keep `Add`, `Depth Preview`, and strong curated examples
- treat `Remove` as limited/experimental
- treat `Move` as prototype/future work unless improved
- present `Mood Shift` as palette transfer, not redesign

## Hypothetical 3D Reconstruction Direction

### Feasible near-term direction

Not full 3D reconstruction, but:

- monocular depth
- room layout estimation
- coarse floor/wall plane fitting
- occupancy and anchor generation
- 2.5D placement reasoning

This is realistic for local execution on a strong consumer GPU and would improve `Add` and `Move`.

### Harder long-term direction

- actual editable 3D room reconstruction
- stable object-aware scene graph
- camera intrinsics estimation
- geometric reprojection for inserted or moved furniture
- stronger lighting and shadow consistency

This is a bigger research project and not a small extension of the current repo.

## Recommended Practical Roadmap

1. Keep the current checkpoint and avoid broad refactors without a rollback point.
2. Add cached scene understanding on room upload.
3. Add floor/occupancy-aware compound placement.
4. Upgrade inpainting.
5. Improve `Add` placement realism and `Move` only after the scene planner exists.
6. Reframe or de-emphasize `Mood Shift` and weak live-demo features unless they are rebuilt.

## Present-State Summary

ReScene AI currently works best as:

- an interactive CV prototype
- a SAM + ZoeDepth + compositing demo
- a room-editing workflow experiment

It does not yet behave like:

- a robust semantic interior-planning tool
- a true generative restyling system
- a photorealistic geometry-aware object editor

That gap is exactly where the next architecture changes should focus.
