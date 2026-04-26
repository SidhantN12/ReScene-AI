"""
ReScene AI — Gradio web interface.

Tabs:
  1. Remove   — upload → click object → SAM highlights → Remove → before/after
  2. Move     — pick object + destination → composited result
  3. Add      — upload/catalog furniture + click position → composited result
  4. Mood Shift  — choose mood → full-room style transfer → before/after slider
  5. Compound — NL text or raw JSON → chain multiple ops → stage gallery
  6. Depth    — ZoeDepth heatmap preview
  7. Examples — pre-computed demo gallery

Run with:
    conda activate torch-cu121
    python app.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import config, STYLES
from catalog_utils import load_catalog_items, build_choice_map, build_nl_lookup
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.nl_parser import parse_nl
from pipeline.scene_context import SceneContext, SceneContextBuilder
from pipeline.scene_cache import get_scene_cache
from utils.image_utils import ensure_rgb
from utils.depth_utils import depth_colourmap

DEFAULT_CATALOG_DIR = Path(__file__).parent / "data" / "catalog"
EXTERNAL_CATALOG_DIR = Path(__file__).parent / "Furniture_images"
CATALOG_DIR = EXTERNAL_CATALOG_DIR if EXTERNAL_CATALOG_DIR.exists() else DEFAULT_CATALOG_DIR
EXAMPLES_DIR = Path(__file__).parent / "data" / "examples" / "output"
_CATALOG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
/* ── Header ─────────────────────────────────────────────────── */
.rs-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 12px; padding: 20px 28px 16px; margin-bottom: 8px;
    color: white;
}
.rs-header h1 { margin: 0 0 4px; font-size: 2rem; font-weight: 700; }
.rs-header p  { margin: 0; opacity: .78; font-size: .95rem; }

/* ── Stage gallery caption ───────────────────────────────────── */
.stage-caption { font-size: 11px; color: #555; text-align: center; }

/* ── Status boxes ────────────────────────────────────────────── */
.status-ok   textarea { border-left: 3px solid #22c55e !important; }
.status-warn textarea { border-left: 3px solid #f59e0b !important; }
.status-err  textarea { border-left: 3px solid #ef4444 !important; }

/* ── Timing badge ────────────────────────────────────────────── */
.timing-badge {
    display: inline-block; background: #e0f2fe; color: #0369a1;
    border-radius: 4px; padding: 1px 7px; font-size: 11px;
    font-family: monospace;
}
"""

# ---------------------------------------------------------------------------
# Lazy pipeline singleton
# ---------------------------------------------------------------------------

_orchestrator: PipelineOrchestrator | None = None


def _orch() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising pipeline orchestrator…")
        _orchestrator = PipelineOrchestrator(config)
    return _orchestrator


# ---------------------------------------------------------------------------
# Lazy scene-context singleton (Phase 8)
# ---------------------------------------------------------------------------

_scene_builder: SceneContextBuilder | None = None


def _get_scene_builder() -> SceneContextBuilder:
    global _scene_builder
    if _scene_builder is None:
        mm = _orch()._mm if _orchestrator is not None else None
        _scene_builder = SceneContextBuilder(model_manager=mm, config=config)
    return _scene_builder


def _build_scene_ctx_safe(image: np.ndarray) -> SceneContext:
    """Build (or return cached) SceneContext for *image*. Never raises."""
    try:
        cache = get_scene_cache()
        builder = _get_scene_builder()
        return cache.get_or_build(image, builder)
    except Exception as exc:
        logger.warning("[SCENE] Context build failed (%s) — returning empty context.", exc)
        import hashlib
        empty = SceneContext(image=image,
                             image_hash=hashlib.sha256(image.tobytes()).hexdigest())
        return empty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_catalog() -> list[str]:
    return [item.label for item in load_catalog_items(CATALOG_DIR)]


def _catalog_choice_map() -> dict[str, str]:
    return build_choice_map(load_catalog_items(CATALOG_DIR))


def _catalog_nl_lookup() -> dict[str, str]:
    return build_nl_lookup(load_catalog_items(CATALOG_DIR))


def _catalog_asset_to_rgba(src: np.ndarray) -> np.ndarray:
    """Normalize a catalog asset into RGBA.

    Real furniture datasets often ship RGB product shots on white backgrounds
    instead of transparent PNG cutouts. Preserve alpha when it already exists;
    otherwise, make near-white pixels transparent so catalog items can be used
    without manual editing.
    """
    if src.ndim == 2:
        rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        alpha = np.full(src.shape[:2], 255, dtype=np.uint8)
        return np.dstack([rgb, alpha])

    if src.shape[2] == 4:
        return cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA)

    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    is_bg = (r > 240) & (g > 240) & (b > 240)
    alpha = np.where(is_bg, np.uint8(0), np.uint8(255))
    return np.dstack([rgb, alpha])


def _crop_rgba_to_alpha(rgba: np.ndarray, pad: int = 6) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(rgba.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(rgba.shape[0], int(ys.max()) + 1 + pad)
    return rgba[y0:y1, x0:x1]


def _make_before_after(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return a side-by-side labelled before/after composite."""
    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (240, 240, 240), 2, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    target_h = max(before.shape[0], after.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        sc = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * sc), target_h))

    b = _label(_fit(before), "Before")
    a = _label(_fit(after), "After")
    div = np.full((target_h, 4, 3), 180, dtype=np.uint8)
    return np.concatenate([b, div, a], axis=1)


def _make_slider_html(before: np.ndarray, after: np.ndarray, height: int = 380) -> str:
    """Return an HTML drag-to-compare before/after slider."""

    def _enc(img: np.ndarray) -> str:
        from PIL import Image as _PIL
        buf = io.BytesIO()
        _PIL.fromarray(img).save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    b64b = _enc(before)
    b64a = _enc(after)
    uid = abs(hash(id(before)) ^ hash(id(after)) ^ id(b64b)) % 10_000_000

    return f"""
<div id="sc{uid}" style="
    position:relative; width:100%; height:{height}px;
    overflow:hidden; cursor:ew-resize; user-select:none;
    border-radius:10px; box-shadow:0 3px 14px rgba(0,0,0,.25);
    background:#111;
">
  <!-- After (right) -->
  <img src="data:image/jpeg;base64,{b64a}"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;">
  <!-- Before (left, clipped) -->
  <img id="bi{uid}" src="data:image/jpeg;base64,{b64b}"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;
              clip-path:inset(0 50% 0 0);pointer-events:none;">
  <!-- Divider -->
  <div id="dv{uid}" style="
      position:absolute;top:0;left:50%;width:2px;height:100%;
      background:rgba(255,255,255,.88);transform:translateX(-50%);
      pointer-events:none;box-shadow:0 0 8px rgba(0,0,0,.4);
  ">
    <div style="position:absolute;top:50%;left:50%;
                transform:translate(-50%,-50%);width:34px;height:34px;
                background:#fff;border-radius:50%;
                box-shadow:0 2px 8px rgba(0,0,0,.35);
                display:flex;align-items:center;justify-content:center;
                font-size:16px;">⇔</div>
  </div>
  <!-- Labels -->
  <span style="position:absolute;top:9px;left:10px;
               background:rgba(0,0,0,.62);color:#fff;
               padding:2px 9px;border-radius:4px;font:bold 11px sans-serif;
               pointer-events:none;">BEFORE</span>
  <span style="position:absolute;top:9px;right:10px;
               background:rgba(0,0,0,.62);color:#fff;
               padding:2px 9px;border-radius:4px;font:bold 11px sans-serif;
               pointer-events:none;">AFTER</span>
</div>
<script>
(function(){{
  var el=document.getElementById('sc{uid}'),
      bi=document.getElementById('bi{uid}'),
      dv=document.getElementById('dv{uid}'),
      dr=false;
  function mv(x){{
    var r=el.getBoundingClientRect(),
        p=Math.min(Math.max((x-r.left)/r.width,.01),.99);
    bi.style.clipPath='inset(0 '+(100-p*100)+'% 0 0)';
    dv.style.left=(p*100)+'%';
  }}
  el.addEventListener('mousedown',function(e){{dr=true;mv(e.clientX);e.preventDefault();}});
  window.addEventListener('mouseup',function(){{dr=false;}});
  window.addEventListener('mousemove',function(e){{if(dr)mv(e.clientX);}});
  el.addEventListener('touchstart',function(e){{dr=true;mv(e.touches[0].clientX);e.preventDefault();}},{{passive:false}});
  window.addEventListener('touchend',function(){{dr=false;}});
  window.addEventListener('touchmove',function(e){{if(dr)mv(e.touches[0].clientX);}},{{passive:false}});
}})();
</script>
"""


def _fmt_status(msg: str, elapsed: float | None = None) -> str:
    if elapsed is not None:
        return f"{msg}  [{elapsed:.1f}s]"
    return msg


def _render_scene_inspector(ctx: SceneContext) -> tuple[np.ndarray, str]:
    """Render a single composite overlay image and a summary text from *ctx*."""
    image = ctx.image.copy()
    h, w = image.shape[:2]
    overlay = image.copy().astype(np.float32)

    # Floor — semi-transparent green
    if ctx.floor_mask is not None and ctx.floor_mask.any():
        fm = ctx.floor_mask.astype(bool)
        overlay[fm] = overlay[fm] * 0.55 + np.array([30, 220, 80], dtype=np.float32) * 0.45

    # Walls — per-wall tint (cycle through blue shades)
    wall_colors = [
        np.array([60, 120, 220], dtype=np.float32),
        np.array([100, 160, 255], dtype=np.float32),
        np.array([40,  90, 200], dtype=np.float32),
    ]
    for i, wm in enumerate(ctx.wall_masks):
        if wm is not None and wm.astype(bool).any():
            wc = wall_colors[i % len(wall_colors)]
            wm_b = wm.astype(bool)
            overlay[wm_b] = overlay[wm_b] * 0.60 + wc * 0.40

    # Occupancy — semi-transparent red
    if ctx.occupancy_mask is not None and ctx.occupancy_mask.astype(bool).any():
        om = ctx.occupancy_mask.astype(bool)
        overlay[om] = overlay[om] * 0.55 + np.array([220, 50, 50], dtype=np.float32) * 0.45

    canvas = np.clip(overlay, 0, 255).astype(np.uint8).copy()
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Anchor candidate dots — yellow filled circles
    for a in ctx.anchor_candidates:
        ax, ay = int(a["x"]), int(a["y"])
        cv2.circle(bgr, (ax, ay), 6, (0, 220, 220), -1)
        cv2.circle(bgr, (ax, ay), 7, (0, 0, 0), 1)

    # Vanishing point crosses — white crosshairs
    cross_r = 22
    for vx, vy in ctx.vanishing_points:
        ivx, ivy = int(round(vx)), int(round(vy))
        cv2.line(bgr, (ivx - cross_r, ivy), (ivx + cross_r, ivy), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(bgr, (ivx, ivy - cross_r), (ivx, ivy + cross_r), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, (ivx, ivy), 5, (0, 200, 255), -1)

    # Legend strip at bottom
    legend_h = 26
    cv2.rectangle(bgr, (0, h - legend_h), (w, h), (20, 20, 20), -1)
    cv2.putText(bgr, "Green=floor  Blue=wall  Red=furniture  Yellow=anchor  White=VP",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Summary text
    intrinsics = ctx.camera_intrinsics_estimate
    f_str = f"{intrinsics['f_px']:.1f} px" if intrinsics else "n/a"
    floor_pct = (ctx.floor_mask.sum() / (h * w) * 100) if ctx.floor_mask is not None else 0
    occ_pct = (ctx.occupancy_mask.astype(bool).sum() / (h * w) * 100
               if ctx.occupancy_mask is not None else 0)
    summary_lines = [
        f"Image: {w}×{h}  |  Hash: {ctx.image_hash[:12]}…",
        f"Masks (SAM): {len(ctx.panoptic_masks)}  |  Labeled: {len(ctx.panoptic_labels)}",
        f"Floor coverage: {floor_pct:.1f}%  |  Furniture coverage: {occ_pct:.1f}%",
        f"Wall regions: {len(ctx.wall_masks)}",
        f"Vanishing points: {len(ctx.vanishing_points)}" +
        (f"  {[f'({vx:.0f},{vy:.0f})' for vx,vy in ctx.vanishing_points]}"
         if ctx.vanishing_points else ""),
        f"Focal length estimate: {f_str}",
        f"Anchor candidates: {len(ctx.anchor_candidates)}",
        f"Depth map: {'yes' if ctx.depth_map is not None else 'no'}",
    ]
    return result_rgb, "\n".join(summary_lines)


def _load_examples() -> list[tuple[np.ndarray, str]]:
    if not EXAMPLES_DIR.exists():
        return []
    items = []
    for p in sorted(EXAMPLES_DIR.glob("*.jpg")) + sorted(EXAMPLES_DIR.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            items.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), p.stem.replace("_", " ")))
    return items


# ---------------------------------------------------------------------------
# Remove tab handlers
# ---------------------------------------------------------------------------

def on_room_upload(image: np.ndarray | None) -> tuple[Any, Any, str]:
    if image is None:
        return None, None, "Upload a room photo to begin."
    return ensure_rgb(image), None, "Click on the object you want to remove."


def on_click_select(
    image: np.ndarray | None,
    mask_state: np.ndarray | None,
    evt: gr.SelectData,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        return None, None, "Please upload an image first."
    cx, cy = int(evt.index[0]), int(evt.index[1])
    try:
        res = _orch().select_object(image=ensure_rgb(image), click_point=(cx, cy))
        mask, score = res["mask"], res["score"]
        area_pct = mask.astype(bool).sum() / mask.size * 100
        return (res["overlay"], mask,
                f"Object selected — SAM confidence: {score:.2f}, area: {area_pct:.1f}%.  "
                "Click again to reselect, or press 'Remove Object'.")
    except Exception as exc:
        logger.exception("select_object failed")
        return ensure_rgb(image), None, f"Selection failed: {exc}"


def on_remove(
    image: np.ndarray | None,
    mask_state: np.ndarray | None,
    style_after: str = "None",
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        return None, None, "Please upload an image first."
    if mask_state is None:
        return None, None, "Please click on an object to select it first."
    try:
        t0 = time.perf_counter()
        rgb = ensure_rgb(image)
        result = _orch().remove(image=rgb, object_mask=mask_state)
        if style_after and style_after != "None":
            result = _orch().restyle(image=result, style_name=style_after)
        elapsed = time.perf_counter() - t0
        before = _make_before_after(rgb, result)
        status = _fmt_status(
            f"Object removed." + (f"  Mood shift: {style_after}." if style_after != "None" else ""),
            elapsed,
        )
        return result, before, status
    except Exception as exc:
        logger.exception("remove failed")
        return None, None, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Move tab handlers
# ---------------------------------------------------------------------------

def on_move_upload(image: np.ndarray | None) -> tuple:
    if image is None:
        return None, None, None, (1.0, 1.0), None, None, "Upload a room photo to begin."
    rgb = ensure_rgb(image)
    return rgb, rgb, None, (1.0, 1.0), None, rgb, "Step 2 — Click the object to move."


def on_move_object_click(
    image: np.ndarray | None, evt: gr.SelectData,
) -> tuple[np.ndarray | None, np.ndarray | None, Any, str]:
    if image is None:
        return None, None, (1.0, 1.0), "Upload an image first."
    cx, cy = int(evt.index[0]), int(evt.index[1])
    try:
        res = _orch().select_object(image=ensure_rgb(image), click_point=(cx, cy))
        mask, score = res["mask"], res["score"]
        area_pct = mask.astype(bool).sum() / mask.size * 100
        return (res["overlay"], mask, res["scale_xy"],
                f"Object selected — confidence: {score:.2f}, area: {area_pct:.1f}%.  "
                "Step 3 — Click the destination on the right panel.")
    except Exception as exc:
        logger.exception("move object click failed")
        return ensure_rgb(image), None, (1.0, 1.0), f"Selection failed: {exc}"


def on_move_target_click(
    original_image: np.ndarray | None, scale_state: tuple, evt: gr.SelectData,
) -> tuple[np.ndarray | None, Any, str]:
    if original_image is None:
        return None, None, "Upload an image first."
    sx, sy = scale_state if scale_state else (1.0, 1.0)
    cx, cy = int(evt.index[0]), int(evt.index[1])
    tx = int(np.clip(cx * sx, 0, 32767))
    ty = int(np.clip(cy * sy, 0, 32767))
    marked = ensure_rgb(original_image).copy()
    bgr = cv2.cvtColor(marked, cv2.COLOR_RGB2BGR)
    r = 18
    cv2.circle(bgr, (cx, cy), r, (50, 50, 220), 3)
    cv2.line(bgr, (cx - r, cy), (cx + r, cy), (50, 50, 220), 2)
    cv2.line(bgr, (cx, cy - r), (cx, cy + r), (50, 50, 220), 2)
    cv2.putText(bgr, "Destination", (cx + 22, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 220), 2, cv2.LINE_AA)
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tx, ty),
            f"Destination ({tx}, {ty}) set.  Click again to adjust or press 'Move Object'.")


def on_move_execute(
    original_image: np.ndarray | None,
    mask_state: np.ndarray | None,
    target_state: tuple | None,
    style_after: str = "None",
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if original_image is None:
        return None, None, "Upload a room image first."
    if mask_state is None:
        return None, None, "Click an object to select it (Step 2)."
    if target_state is None:
        return None, None, "Click a destination on the right panel (Step 3)."
    try:
        t0 = time.perf_counter()
        rgb = ensure_rgb(original_image)
        tx, ty = target_state
        scene_ctx = _build_scene_ctx_safe(rgb)
        result = _orch().move(image=rgb, object_name="", target_position=(tx, ty),
                              source_mask=mask_state, scene_context=scene_ctx)
        if style_after and style_after != "None":
            result = _orch().restyle(image=result, style_name=style_after)
        elapsed = time.perf_counter() - t0
        comparison = _make_before_after(rgb, result)
        status = _fmt_status(
            f"Object moved to ({tx}, {ty})." + (f"  Mood shift: {style_after}." if style_after != "None" else ""),
            elapsed,
        )
        return result, comparison, status
    except Exception as exc:
        logger.exception("move execute failed")
        return None, None, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Add tab handlers
# ---------------------------------------------------------------------------

def on_add_room_upload(room_img: np.ndarray | None) -> tuple:
    if room_img is None:
        return None, None, (1.0, 1.0), "Upload a room photo to begin."
    rgb = ensure_rgb(room_img)
    h, w = rgb.shape[:2]
    max_size = config.inference.max_image_size
    scale = min(1.0, max_size / max(h, w))
    return (rgb, rgb, (scale, scale),
            "Room uploaded.  Step 2 — choose furniture.  Step 3 — click to set position.")


def on_add_catalog_select(choice: str) -> tuple:
    if not choice or choice == "(upload instead)":
        return None, None, "Upload your own furniture image below."
    choice_map = _catalog_choice_map()
    filename = choice_map.get(choice, choice)
    furn_path = CATALOG_DIR / filename
    if not furn_path.exists():
        return None, None, f"Catalog file not found: {choice}"
    asset = cv2.imread(str(furn_path), cv2.IMREAD_UNCHANGED)
    if asset is None:
        return None, None, f"Failed to load {choice}."
    rgba = _crop_rgba_to_alpha(_catalog_asset_to_rgba(asset))
    label = choice if choice in choice_map else furn_path.stem.replace("_", " ").title()
    return rgba[:, :, :3].copy(), rgba, f"Catalog: {label} loaded."


def on_add_furn_upload(furn: np.ndarray | None) -> tuple:
    if furn is None:
        return None, "Upload a furniture image."
    if furn.ndim == 2:
        furn = cv2.cvtColor(furn, cv2.COLOR_GRAY2RGBA)
    elif furn.ndim == 3 and furn.shape[2] == 3:
        furn = _crop_rgba_to_alpha(_catalog_asset_to_rgba(cv2.cvtColor(furn, cv2.COLOR_RGB2BGR)))
    elif furn.ndim == 3 and furn.shape[2] == 4:
        furn = _crop_rgba_to_alpha(furn)
    return furn, (
        f"Furniture uploaded ({furn.shape[1]}x{furn.shape[0]}). "
        "Adjust the size slider if it still looks too small or too large."
    )


def on_add_room_click(
    original_image: np.ndarray | None, scale_state: tuple, evt: gr.SelectData,
) -> tuple:
    if original_image is None:
        return None, None, "Upload a room image first."
    sx, sy = scale_state if scale_state else (1.0, 1.0)
    cx, cy = int(evt.index[0]), int(evt.index[1])
    tx = int(np.clip(cx * sx, 0, 32767))
    ty = int(np.clip(cy * sy, 0, 32767))
    marked = ensure_rgb(original_image).copy()
    bgr = cv2.cvtColor(marked, cv2.COLOR_RGB2BGR)
    r = 20
    cv2.circle(bgr, (cx, cy), r, (30, 220, 80), 3)
    cv2.line(bgr, (cx - r - 5, cy), (cx + r + 5, cy), (30, 220, 80), 2)
    cv2.line(bgr, (cx, cy - r - 5), (cx, cy + r + 5), (30, 220, 80), 2)
    cv2.putText(bgr, "Insert here", (cx + r + 6, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 80), 2, cv2.LINE_AA)
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tx, ty),
            f"Insertion point ({tx}, {ty}).  Click again to adjust or press 'Add Furniture'.")


def on_add_execute(
    add_original: np.ndarray | None,
    add_furn: Any,
    add_target: tuple | None,
    add_size_multiplier: float,
    style_after: str = "None",
) -> tuple:
    if add_original is None:
        return None, None, "Upload a room image first."
    if add_furn is None:
        return None, None, "Select or upload a furniture image."
    if add_target is None:
        return None, None, "Click the room to set the insertion point."
    tx, ty = add_target
    try:
        t0 = time.perf_counter()
        rgb_add = ensure_rgb(add_original)
        scene_ctx = _build_scene_ctx_safe(rgb_add)
        result = _orch().add(image=rgb_add, furniture_image=add_furn,
                             target_position=(tx, ty),
                             size_multiplier=float(add_size_multiplier),
                             scene_context=scene_ctx)
        if style_after and style_after != "None":
            result = _orch().restyle(image=result, style_name=style_after)
        elapsed = time.perf_counter() - t0
        comparison = _make_before_after(rgb_add, result)
        status = _fmt_status(
            f"Furniture added at ({tx}, {ty}) with size ×{add_size_multiplier:.2f}."
            + (f"  Mood shift: {style_after}." if style_after != "None" else ""),
            elapsed,
        )
        return result, comparison, status
    except Exception as exc:
        logger.exception("add failed")
        return None, None, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Restyle tab handler
# ---------------------------------------------------------------------------

def handle_restyle(
    room_image: np.ndarray | None,
    style_name: str,
) -> tuple[np.ndarray | None, str, str]:
    """Returns (result_image, slider_html, status)."""
    if room_image is None:
        return None, "", "Please upload a room image."
    try:
        t0 = time.perf_counter()
        rgb = ensure_rgb(room_image)
        result = _orch().restyle(image=rgb, style_name=style_name)
        elapsed = time.perf_counter() - t0
        slider_html = _make_slider_html(rgb, result)
        return result, slider_html, _fmt_status(f"Mood shift '{style_name}' applied.", elapsed)
    except Exception as exc:
        logger.exception("restyle failed")
        return None, "", f"Error: {exc}"


# ---------------------------------------------------------------------------
# Compound tab — JSON mode handler
# ---------------------------------------------------------------------------

def handle_compound(
    room_image: np.ndarray | None,
    instructions_json: str,
) -> tuple[np.ndarray | None, list, str]:
    """Returns (result_image, stages_gallery, status)."""
    if room_image is None:
        return None, [], "Please upload a room image."
    try:
        instructions: list[dict[str, Any]] = json.loads(instructions_json)
        if not isinstance(instructions, list):
            return None, [], "Instructions must be a JSON array."
    except json.JSONDecodeError as exc:
        return None, [], f"Invalid JSON: {exc}"
    try:
        stages: list[dict] = []
        t0 = time.perf_counter()
        result = _orch().compound(image=ensure_rgb(room_image),
                                  instructions=instructions,
                                  stage_sink=stages)
        elapsed = time.perf_counter() - t0
        gallery = [(s["image"], f"{s['name']} · {s['time_s']:.1f}s") for s in stages]
        timing = " | ".join(f"{s['name']}: {s['time_s']:.1f}s" for s in stages)
        status = _fmt_status(f"Completed {len(instructions)} operation(s).", elapsed)
        if timing:
            status += f"\n{timing}"
        return result, gallery, status
    except Exception as exc:
        logger.exception("compound failed")
        return None, [], f"Error: {exc}"


# ---------------------------------------------------------------------------
# Compound tab — NL mode handler
# ---------------------------------------------------------------------------

def handle_nl_parse(
    nl_text: str,
    room_image: np.ndarray | None,
) -> tuple[str, str]:
    """Parse NL text → return (parsed_json_preview, parse_warnings)."""
    if not nl_text or not nl_text.strip():
        return "[]", "Enter instructions above then click Parse."
    catalog = _catalog_nl_lookup()
    img_size = (1024, 1024)
    if room_image is not None:
        rgb = ensure_rgb(room_image)
        img_size = (rgb.shape[1], rgb.shape[0])
    instructions, warnings = parse_nl(nl_text, catalog, STYLES, image_size=img_size)
    reverse_choice_map = {filename: label for label, filename in _catalog_choice_map().items()}
    # Strip furniture_image_path from JSON preview (path is internal)
    preview: list[dict[str, Any]] = []
    for instr in instructions:
        item = dict(instr)
        filename = item.pop("furniture_image_path", None)
        if filename and "catalog_label" in item:
            item["catalog_label"] = reverse_choice_map.get(filename, item["catalog_label"])
        preview.append(item)
    warn_text = "\n".join(f"⚠  {w}" for w in warnings) if warnings else ""
    return json.dumps(preview, indent=2), warn_text


def handle_nl_compound(
    room_image: np.ndarray | None,
    nl_text: str,
) -> tuple[np.ndarray | None, list, str, str]:
    """Parse NL → execute → return (result_image, stages_gallery, status, slider_html)."""
    if room_image is None:
        return None, [], "Upload a room image first.", ""
    if not nl_text or not nl_text.strip():
        return None, [], "Enter instructions then click 'Run NL Compound'.", ""

    catalog = _catalog_nl_lookup()
    rgb = ensure_rgb(room_image)
    img_size = (rgb.shape[1], rgb.shape[0])
    instructions, warnings = parse_nl(nl_text, catalog, STYLES, image_size=img_size)

    # Resolve furniture_image_path → numpy array
    preprocessed: list[dict] = []
    load_errs: list[str] = []
    for instr in instructions:
        instr = dict(instr)
        if "target_position" in instr:
            instr["target_position"] = tuple(instr["target_position"])
        if "furniture_image_path" in instr:
            fpath = CATALOG_DIR / instr.pop("furniture_image_path")
            bgra = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
            if bgra is None:
                load_errs.append(f"Could not load catalog item: {fpath.name}")
                continue
            if bgra.ndim == 3 and bgra.shape[2] == 4:
                furn = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
            elif bgra.ndim == 3:
                furn = cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB)
            else:
                furn = cv2.cvtColor(bgra, cv2.COLOR_GRAY2RGB)
            instr["furniture_image"] = furn
        instr.pop("catalog_label", None)
        instr.pop("position_hint", None)
        preprocessed.append(instr)

    all_warnings = warnings + load_errs

    if not preprocessed:
        msg = "\n".join(f"⚠  {w}" for w in all_warnings) if all_warnings else "No executable operations found."
        return None, [], msg, ""

    # Execute
    stages: list[dict] = []
    try:
        t0 = time.perf_counter()
        result = _orch().compound(image=rgb, instructions=preprocessed, stage_sink=stages)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        logger.exception("NL compound failed")
        return None, [], f"Error: {exc}", ""

    gallery = [(s["image"], f"{s['name']} · {s['time_s']:.1f}s") for s in stages]
    slider_html = _make_slider_html(rgb, result) if result is not None else ""

    status_lines = [_fmt_status(f"Completed {len(preprocessed)} operation(s).", elapsed)]
    if stages:
        status_lines.append("  ".join(f"{s['name']}: {s['time_s']:.1f}s" for s in stages))
    if all_warnings:
        status_lines += [f"⚠  {w}" for w in all_warnings]

    return result, gallery, "\n".join(status_lines), slider_html


# ---------------------------------------------------------------------------
# Scene Inspector handler (Phase 8)
# ---------------------------------------------------------------------------

def handle_scene_inspector(
    room_image: np.ndarray | None,
) -> tuple[np.ndarray | None, str, str]:
    """Build scene context and render annotated overlay.

    Returns (overlay_image, summary_text, status).
    """
    if room_image is None:
        return None, "", "Upload a room photo to begin."
    try:
        t0 = time.perf_counter()
        rgb = ensure_rgb(room_image)
        ctx = _build_scene_ctx_safe(rgb)
        elapsed = time.perf_counter() - t0
        overlay, summary = _render_scene_inspector(ctx)
        status = _fmt_status(
            f"Scene context built.  "
            f"Masks: {len(ctx.panoptic_masks)}, "
            f"VPs: {len(ctx.vanishing_points)}, "
            f"Anchors: {len(ctx.anchor_candidates)}.",
            elapsed,
        )
        return overlay, summary, status
    except Exception as exc:
        logger.exception("scene inspector failed")
        return None, "", f"Error: {exc}"


# ---------------------------------------------------------------------------
# Depth preview handler
# ---------------------------------------------------------------------------

def handle_depth_preview(room_image: np.ndarray | None) -> np.ndarray | None:
    if room_image is None:
        return None
    try:
        room = ensure_rgb(room_image)
        orch = _orch()
        de = orch._mm.get("zoedepth")
        depth = de.predict(room)
        orch._mm.unload_current()
        return depth_colourmap(depth)
    except Exception as exc:
        logger.exception("depth preview failed")
        return None


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

_COMPOUND_PLACEHOLDER = json.dumps([
    {"operation": "restyle", "style_name": "Bohemian"}
], indent=2)

_COMPOUND_EXAMPLE = json.dumps([
    {"operation": "restyle", "style_name": "Scandinavian"},
    {"operation": "add", "furniture_image": "<<numpy array>>",
     "target_position": [400, 600]},
], indent=2)

_NL_PLACEHOLDER = (
    'e.g. "add a sofa on the left, then make it scandinavian" or '
    '"place an armchair in the corner, apply coastal style"'
)


def build_ui() -> gr.Blocks:
    style_choices = list(STYLES.keys())

    with gr.Blocks(title="ReScene AI") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="rs-header">
          <h1>ReScene AI</h1>
          <p>Intelligent room rearrangement, furniture placement, and restyling
             from a single photograph.</p>
        </div>
        """)

        # ── Tab 1: Remove ─────────────────────────────────────────────────
        with gr.Tab("Remove"):
            gr.Markdown(
                "### Remove an object from the room\n"
                "**Step 1** Upload a photo.  "
                "**Step 2** Click the object (SAM highlights it).  "
                "**Step 3** Press *Remove Object*."
            )
            rm_mask_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    rm_room = gr.Image(label="Room Photo — click to select",
                                       type="numpy", height=420, sources=["upload"])
                    rm_overlay = gr.Image(label="Selection preview",
                                          type="numpy", height=420, interactive=False)
                    rm_status = gr.Textbox(label="Status",
                                           value="Upload a room photo to begin.",
                                           interactive=False, lines=2)
                    rm_style_after = gr.Dropdown(
                        choices=["None"] + style_choices, value="None",
                        label="Apply mood shift after remove (optional)")
                    rm_btn = gr.Button("Remove Object", variant="primary", size="lg")
                with gr.Column(scale=1):
                    rm_result = gr.Image(label="Result", type="numpy",
                                         height=420, interactive=False)
                    rm_comparison = gr.Image(label="Before / After", type="numpy",
                                             height=420, interactive=False)

            rm_room.upload(fn=on_room_upload, inputs=[rm_room],
                           outputs=[rm_overlay, rm_mask_state, rm_status])
            rm_room.select(fn=on_click_select, inputs=[rm_room, rm_mask_state],
                           outputs=[rm_overlay, rm_mask_state, rm_status])
            rm_overlay.select(fn=on_click_select, inputs=[rm_room, rm_mask_state],
                              outputs=[rm_overlay, rm_mask_state, rm_status])
            rm_btn.click(fn=on_remove, inputs=[rm_room, rm_mask_state, rm_style_after],
                         outputs=[rm_result, rm_comparison, rm_status])

        # ── Tab 2: Move ───────────────────────────────────────────────────
        with gr.Tab("Move"):
            gr.Markdown(
                "### Relocate an object within the scene\n"
                "**Step 1** Upload.  **Step 2** Click object (left panel).  "
                "**Step 3** Click destination (right panel).  **Step 4** Move."
            )
            mv_mask_state     = gr.State(None)
            mv_scale_state    = gr.State((1.0, 1.0))
            mv_target_state   = gr.State(None)
            mv_original_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    mv_source = gr.Image(label="Room Photo — click object to move",
                                         type="numpy", height=400, sources=["upload"])
                    mv_target = gr.Image(label="Destination — click where to place",
                                         type="numpy", height=400, interactive=True,
                                         sources=["upload"])
                    mv_status = gr.Textbox(label="Status",
                                           value="Upload a room photo to begin.",
                                           interactive=False, lines=2)
                    mv_style_after = gr.Dropdown(
                        choices=["None"] + style_choices, value="None",
                        label="Apply mood shift after move (optional)")
                    mv_btn = gr.Button("Move Object", variant="primary", size="lg")
                with gr.Column(scale=1):
                    mv_result     = gr.Image(label="Result", type="numpy",
                                             height=400, interactive=False)
                    mv_comparison = gr.Image(label="Before / After", type="numpy",
                                             height=400, interactive=False)

            mv_source.upload(fn=on_move_upload, inputs=[mv_source],
                             outputs=[mv_source, mv_target, mv_mask_state,
                                      mv_scale_state, mv_target_state,
                                      mv_original_state, mv_status])
            mv_target.upload(fn=on_move_upload, inputs=[mv_target],
                             outputs=[mv_source, mv_target, mv_mask_state,
                                      mv_scale_state, mv_target_state,
                                      mv_original_state, mv_status])
            mv_source.select(fn=on_move_object_click, inputs=[mv_source],
                             outputs=[mv_source, mv_mask_state,
                                      mv_scale_state, mv_status])
            mv_target.select(fn=on_move_target_click,
                             inputs=[mv_original_state, mv_scale_state],
                             outputs=[mv_target, mv_target_state, mv_status])
            mv_btn.click(fn=on_move_execute,
                         inputs=[mv_original_state, mv_mask_state,
                                  mv_target_state, mv_style_after],
                         outputs=[mv_result, mv_comparison, mv_status])

        # ── Tab 3: Add ────────────────────────────────────────────────────
        with gr.Tab("Add"):
            gr.Markdown(
                "### Insert furniture into the scene\n"
                "**Step 1** Upload room.  **Step 2** Pick/upload furniture.  "
                "**Step 3** Click insertion point.  **Step 4** Add."
            )
            add_original_state = gr.State(None)
            add_scale_state    = gr.State((1.0, 1.0))
            add_target_state   = gr.State(None)
            add_furn_state     = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    add_room = gr.Image(label="Room Photo — click insertion point",
                                        type="numpy", height=400, sources=["upload"])
                    add_status = gr.Textbox(label="Status",
                                            value="Upload a room photo to begin.",
                                            interactive=False, lines=2)
                    gr.Markdown("**Step 2: Furniture source**")
                    _catalog_choices = ["(upload instead)"] + _list_catalog()
                    add_catalog = gr.Dropdown(choices=_catalog_choices,
                                              value=_catalog_choices[0],
                                              label="Pick from catalog")
                    add_furn = gr.Image(
                        label="— or upload a product photo (PNG preferred)",
                        type="numpy", height=180, sources=["upload"])
                    add_size = gr.Slider(
                        minimum=0.6,
                        maximum=2.6,
                        value=1.35,
                        step=0.05,
                        label="Furniture Size Multiplier",
                        info="Increase this when the inserted furniture looks too small for the room.",
                    )
                    add_style_after = gr.Dropdown(
                        choices=["None"] + style_choices, value="None",
                        label="Apply mood shift after add (optional)")
                    add_btn = gr.Button("Add Furniture", variant="primary", size="lg")
                with gr.Column(scale=1):
                    add_result     = gr.Image(label="Result", type="numpy",
                                              height=400, interactive=False)
                    add_comparison = gr.Image(label="Before / After", type="numpy",
                                              height=400, interactive=False)

            add_room.upload(fn=on_add_room_upload, inputs=[add_room],
                            outputs=[add_room, add_original_state,
                                     add_scale_state, add_status])
            add_room.select(fn=on_add_room_click,
                            inputs=[add_original_state, add_scale_state],
                            outputs=[add_room, add_target_state, add_status])
            add_catalog.change(fn=on_add_catalog_select, inputs=[add_catalog],
                               outputs=[add_furn, add_furn_state, add_status])
            add_furn.upload(fn=on_add_furn_upload, inputs=[add_furn],
                            outputs=[add_furn_state, add_status])
            add_btn.click(fn=on_add_execute,
                          inputs=[add_original_state, add_furn_state,
                                   add_target_state, add_size, add_style_after],
                          outputs=[add_result, add_comparison, add_status])

        # ── Tab 4: Mood Shift ─────────────────────────────────────────────
        with gr.Tab("Mood Shift"):
            gr.Markdown(
                "### Apply a room mood shift\n"
                "Reinhard LAB colour-statistics transfer from curated reference palettes.  "
                "Drag the slider in the result panel to compare before/after."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rs_room  = gr.Image(label="Room Photo", type="numpy", height=400)
                    rs_style = gr.Dropdown(choices=style_choices, value=style_choices[0],
                                           label="Mood Preset")
                    rs_btn    = gr.Button("Apply Mood Shift", variant="primary", size="lg")
                    rs_status = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column(scale=1):
                    rs_out    = gr.Image(label="Result", type="numpy",
                                         height=400, interactive=False)
                    rs_slider = gr.HTML(label="Before / After — drag to compare")

            rs_btn.click(fn=handle_restyle, inputs=[rs_room, rs_style],
                         outputs=[rs_out, rs_slider, rs_status])

        # ── Tab 5: Compound ───────────────────────────────────────────────
        with gr.Tab("Compound Operations"):
            gr.Markdown(
                "### Chain multiple operations\n"
                "**Natural Language mode** — describe what you want in plain English.  "
                "**JSON mode** — supply a raw instruction array for full control."
            )

            with gr.Row():
                cp_room = gr.Image(label="Room Photo", type="numpy",
                                   height=360, sources=["upload"])

            with gr.Tabs():
                # ── NL sub-tab ──────────────────────────────────────────
                with gr.Tab("Natural Language"):
                    cp_nl_input = gr.Textbox(
                        label="Describe your changes",
                        placeholder=_NL_PLACEHOLDER,
                        lines=3,
                    )
                    with gr.Row():
                        cp_nl_parse_btn = gr.Button("Preview parse", size="sm")
                        cp_nl_run_btn   = gr.Button("Run NL Compound",
                                                     variant="primary", size="lg")
                    cp_nl_preview = gr.Code(
                        label="Parsed instructions (preview — editable in JSON mode)",
                        language="json", value="[]", lines=8, interactive=False,
                    )
                    cp_nl_warnings = gr.Textbox(
                        label="Parse notes / warnings",
                        interactive=False, lines=3, visible=True,
                    )

                # ── JSON sub-tab ────────────────────────────────────────
                with gr.Tab("JSON / Advanced"):
                    cp_json = gr.Code(
                        label="Instructions (JSON array)",
                        language="json",
                        value=_COMPOUND_PLACEHOLDER,
                        lines=14,
                    )
                    cp_json_run_btn = gr.Button("Run JSON Compound",
                                                 variant="primary", size="lg")
                    gr.Code(value=_COMPOUND_EXAMPLE, language="json",
                            interactive=False, label="Example instructions")

            # ── Shared results ──────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    cp_out    = gr.Image(label="Final result", type="numpy",
                                         height=380, interactive=False)
                    cp_slider = gr.HTML(label="Before / After slider")
                with gr.Column(scale=1):
                    cp_stages = gr.Gallery(
                        label="Pipeline stages",
                        columns=2,
                        height=380,
                        object_fit="contain",
                    )
            cp_status = gr.Textbox(label="Status / timing", interactive=False, lines=4)

            # ── NL wiring ───────────────────────────────────────────────
            cp_nl_parse_btn.click(
                fn=handle_nl_parse,
                inputs=[cp_nl_input, cp_room],
                outputs=[cp_nl_preview, cp_nl_warnings],
            )
            cp_nl_run_btn.click(
                fn=handle_nl_compound,
                inputs=[cp_room, cp_nl_input],
                outputs=[cp_out, cp_stages, cp_status, cp_slider],
            )

            # ── JSON wiring ─────────────────────────────────────────────
            cp_json_run_btn.click(
                fn=handle_compound,
                inputs=[cp_room, cp_json],
                outputs=[cp_out, cp_stages, cp_status],
            )

        # ── Tab 6: Depth Preview ──────────────────────────────────────────
        with gr.Tab("Depth Preview"):
            gr.Markdown(
                "### ZoeDepth-NK depth estimation\n"
                "TURBO colormap — warm = near, cool = far."
            )
            with gr.Row():
                dp_room = gr.Image(label="Room Photo", type="numpy", height=400)
                dp_out  = gr.Image(label="Depth Map", type="numpy",
                                   height=400, interactive=False)
            dp_btn = gr.Button("Estimate Depth")
            dp_btn.click(fn=handle_depth_preview, inputs=[dp_room], outputs=[dp_out])

        # ── Tab 7: Examples ───────────────────────────────────────────────
        with gr.Tab("Examples"):
            gr.Markdown(
                "### Pre-computed demo examples\n"
                "Run `python generate_examples.py` to populate this gallery "
                "with before/after pairs from the full pipeline."
            )
            _ex = _load_examples()
            if _ex:
                gr.Gallery(value=_ex, label="Demo gallery",
                            columns=3, height=500, object_fit="contain")
            else:
                gr.Markdown(
                    "> **No examples yet.** "
                    "Place room images in `data/examples/input/` and run "
                    "`python generate_examples.py` to generate them."
                )

        # ── Tab 8: Scene Inspector ────────────────────────────────────────
        with gr.Tab("Scene Inspector"):
            gr.Markdown(
                "### Cached scene understanding (Phase 8)\n"
                "Uploads the room to the SAM → ZoeDepth → CLIP pipeline and "
                "shows the derived floor / wall / furniture masks, vanishing points, "
                "camera intrinsics estimate, and anchor candidates.\n\n"
                "> **Note:** Full analysis requires SAM and ZoeDepth weights. "
                "Without weights, heuristic fallbacks are used (position-based "
                "floor/wall masks, no depth)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    si_room = gr.Image(label="Room Photo", type="numpy",
                                       height=420, sources=["upload"])
                    si_btn  = gr.Button("Analyse Scene", variant="primary", size="lg")
                    si_status = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column(scale=1):
                    si_overlay = gr.Image(label="Scene overlay", type="numpy",
                                          height=420, interactive=False)
            si_summary = gr.Textbox(
                label="Scene context summary",
                interactive=False,
                lines=10,
                placeholder="Upload a photo and click 'Analyse Scene'.",
            )
            gr.Markdown(
                "**Overlay legend:** "
                "Green = floor · Blue = wall(s) · Red = furniture/occupancy · "
                "Yellow dots = anchor candidates · White crosses = vanishing points"
            )
            si_btn.click(
                fn=handle_scene_inspector,
                inputs=[si_room],
                outputs=[si_overlay, si_summary, si_status],
            )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n*ReScene AI · Phase 8.  "
            "Cached scene context · CLIP semantic labels · vanishing points · "
            "Caprile-Torre intrinsics · anchor candidates · Scene Inspector tab.*"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.share,
        debug=config.app.debug,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
