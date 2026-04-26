"""
ReScene AI — Cached Scene Understanding Layer (Phase 8).

SceneContextBuilder analyses an uploaded room image once and stores the result
in a SceneContext dataclass.  Every subsequent pipeline operation (move, add,
restyle) can read depth, masks, vanishing points and anchor candidates directly
from the context instead of re-running heavy models.

Pipeline sequence
-----------------
1. SAM automatic segmentation → panoptic masks (fail-safe: empty dict)
2. ZoeDepth metric depth       → depth_map   (fail-safe: None)
3. CLIP zero-shot labels       → panoptic_labels (fail-safe: heuristic)
4. Structural masks derived    → floor / wall / ceiling / occupancy
5. Vanishing-point estimation  → vanishing_points (HoughLinesP + RANSAC voting)
6. Camera intrinsics           → camera_intrinsics_estimate (Caprile-Torre simplified)
7. Anchor candidates           → anchor_candidates (free-floor sampling)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic label vocabulary
# ---------------------------------------------------------------------------

LABELS: list[str] = [
    "floor", "wall", "ceiling",
    "sofa", "chair", "table", "bed", "lamp", "rug",
    "cabinet", "bookshelf", "window", "door",
    "plant", "tv", "painting", "other",
]

_STRUCTURAL_LABELS: frozenset[str] = frozenset({"floor", "wall", "ceiling", "window", "door"})
_OCCUPANCY_LABELS:  frozenset[str] = frozenset(LABELS) - _STRUCTURAL_LABELS


# ---------------------------------------------------------------------------
# SceneContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class SceneContext:
    """Immutable (in practice) snapshot of a room image's semantic structure."""

    image:         np.ndarray          # H×W×3 uint8 RGB
    image_hash:    str                 # sha256 hex used as cache key

    # Depth
    depth_map: np.ndarray | None = None   # H×W float32, metric metres

    # SAM panoptic
    panoptic_masks:  dict[int, np.ndarray] = field(default_factory=dict)  # id → H×W bool
    panoptic_labels: dict[int, str]        = field(default_factory=dict)  # id → label str

    # Structural layers
    floor_mask:    np.ndarray | None = None   # H×W bool
    wall_masks:    list[np.ndarray] = field(default_factory=list)  # one per detected wall
    ceiling_mask:  np.ndarray | None = None
    occupancy_mask: np.ndarray | None = None  # union of all furniture masks
    free_floor_mask: np.ndarray | None = None # floor AND NOT occupancy

    # Geometry
    vanishing_points: list[tuple[float, float]] = field(default_factory=list)
    camera_intrinsics_estimate: dict[str, float] = field(default_factory=dict)
    # {"fx": …, "fy": …, "cx": …, "cy": …, "f_px": …}

    # Placement helpers
    anchor_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"x": int, "y": int, "depth_m": float|None, "score": float, "source": str}

    # Free-form diagnostics
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SceneContextBuilder
# ---------------------------------------------------------------------------

class SceneContextBuilder:
    """Builds a SceneContext for a room image.

    Models (SAM, ZoeDepth, CLIP) are accessed lazily through the provided
    ModelManager.  Every stage is wrapped in try/except so the builder always
    returns a usable (possibly degraded) context even when models are absent.
    """

    def __init__(self, model_manager=None, config=None) -> None:
        self._mm = model_manager
        self._cfg = config
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, image: np.ndarray) -> SceneContext:
        """Build and return a SceneContext for *image* (H×W×3 uint8 RGB)."""
        image_hash = _sha256(image)

        ctx = SceneContext(image=image, image_hash=image_hash)
        meta: dict[str, Any] = {}

        # 1. SAM
        masks, sam_ok, sam_error, sam_oom = self._run_sam(image)
        ctx.panoptic_masks = masks
        meta["sam_available"] = sam_ok
        meta["sam_mask_count"] = len(masks)
        meta["sam_error"] = sam_error
        meta["sam_oom"] = sam_oom

        # 2. ZoeDepth
        depth, depth_ok, depth_error, depth_oom = self._run_zoedepth(image)
        ctx.depth_map = depth
        meta["depth_available"] = depth_ok
        meta["depth_error"] = depth_error
        meta["depth_oom"] = depth_oom

        # 3. Classify masks → semantic labels
        if masks:
            labels, label_mode, clip_error = self._classify_masks(image, masks)
        else:
            labels, label_mode, clip_error = {}, "none", ""
        ctx.panoptic_labels = labels
        meta["label_mode"] = label_mode
        meta["clip_error"] = clip_error
        meta["analysis_shape"] = list(image.shape[:2])
        meta["resource_limited"] = bool(sam_oom or depth_oom)

        # 4. Derive structural / occupancy masks
        self._derive_structural_masks(ctx, image.shape[:2])

        # 5. Vanishing points
        vps = self._compute_vanishing_points(image)
        ctx.vanishing_points = vps
        meta["vanishing_point_count"] = len(vps)

        # 6. Camera intrinsics
        if len(vps) >= 2:
            ctx.camera_intrinsics_estimate = self._estimate_intrinsics(
                vps[0], vps[1], image.shape
            )

        # 7. Anchor candidates
        ctx.anchor_candidates = self._build_anchor_candidates(ctx)
        meta["anchor_count"] = len(ctx.anchor_candidates)

        ctx.metadata = meta
        return ctx

    # ------------------------------------------------------------------
    # Stage 1: SAM
    # ------------------------------------------------------------------

    def _run_sam(self, image: np.ndarray) -> tuple[dict[int, np.ndarray], bool, str, bool]:
        if self._mm is None:
            return {}, False, "model manager unavailable", False
        try:
            seg = self._mm.get("sam")
            try:
                result = seg.predict(image=image, automatic=True)
            finally:
                self._mm.unload_current()
            masks_list: list[np.ndarray] = result.get("masks", [])
            masks = {i: m.astype(bool) for i, m in enumerate(masks_list)}
            return masks, True, "", False
        except Exception as exc:
            logger.warning("[SCENE] SAM unavailable (%s) — skipping.", exc)
            return {}, False, str(exc), _is_cuda_oom_message(exc)

    # ------------------------------------------------------------------
    # Stage 2: ZoeDepth
    # ------------------------------------------------------------------

    def _run_zoedepth(self, image: np.ndarray) -> tuple[np.ndarray | None, bool, str, bool]:
        if self._mm is None:
            return None, False, "model manager unavailable", False
        try:
            zoedepth = self._mm.get("zoedepth")
            try:
                depth = zoedepth.predict(image=image)          # H×W float32 metres
            finally:
                self._mm.unload_current()
            if isinstance(depth, dict):
                depth = depth.get("depth_map", depth.get("depth"))
            depth = np.asarray(depth, dtype=np.float32)
            # Resize to match image if needed
            h, w = image.shape[:2]
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            return depth, True, "", False
        except Exception as exc:
            logger.warning("[SCENE] ZoeDepth unavailable (%s) — skipping.", exc)
            return None, False, str(exc), _is_cuda_oom_message(exc)

    # ------------------------------------------------------------------
    # Stage 3: CLIP zero-shot classification
    # ------------------------------------------------------------------

    def _classify_masks(
        self, image: np.ndarray, masks: dict[int, np.ndarray]
    ) -> tuple[dict[int, str], str, str]:
        try:
            return self._clip_classify(image, masks), "clip", ""
        except Exception as exc:
            logger.warning("[SCENE] CLIP unavailable (%s) — using heuristic labels.", exc)
            return self._heuristic_classify(image, masks), "heuristic", str(exc)

    def _clip_classify(
        self, image: np.ndarray, masks: dict[int, np.ndarray]
    ) -> dict[int, str]:
        from safetensors.torch import load_file
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor
        import torch

        processor_path, config_path, weights_path = _resolve_local_clip_assets()
        if self._clip_model is None:
            self._clip_processor = CLIPProcessor.from_pretrained(
                processor_path,
                local_files_only=True,
            )
            device = "cpu"
            if self._cfg is not None:
                device = getattr(self._cfg.device, "device", "cpu")
            clip_config = CLIPConfig.from_pretrained(config_path, local_files_only=True)
            self._clip_model = CLIPModel(clip_config)
            state_dict = load_file(weights_path)
            self._clip_model.load_state_dict(state_dict, strict=False)
            self._clip_model = self._clip_model.to(device)
            self._clip_model.eval()

        device = next(self._clip_model.parameters()).device
        text_prompts = [f"a photo of a {lbl} in a room" for lbl in LABELS]

        labels_out: dict[int, str] = {}
        for mask_id, mask in masks.items():
            crop = _mask_crop(image, mask)
            if crop is None:
                labels_out[mask_id] = "other"
                continue
            from PIL import Image as PILImage
            pil_crop = PILImage.fromarray(crop)
            inputs = self._clip_processor(
                text=text_prompts, images=pil_crop, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image[0]
                idx = int(logits.argmax().item())
            labels_out[mask_id] = LABELS[idx]

        return labels_out

    def _heuristic_classify(
        self, image: np.ndarray, masks: dict[int, np.ndarray]
    ) -> dict[int, str]:
        """Position-based heuristic: large-bottom → floor, large-sides → wall, etc."""
        h, w = image.shape[:2]
        labels_out: dict[int, str] = {}

        for mask_id, mask in masks.items():
            area = mask.sum()
            if area == 0:
                labels_out[mask_id] = "other"
                continue

            ys, xs = np.where(mask)
            cy = float(ys.mean()) / h  # 0=top, 1=bottom
            cx = float(xs.mean()) / w  # 0=left, 1=right
            area_frac = area / (h * w)

            # Ceiling: large area in top third
            if cy < 0.30 and area_frac > 0.10:
                labels_out[mask_id] = "ceiling"
            # Floor: large area in bottom third
            elif cy > 0.70 and area_frac > 0.08:
                labels_out[mask_id] = "floor"
            # Wall: tall/wide shape spanning significant height
            elif area_frac > 0.12 and (ys.max() - ys.min()) / h > 0.50:
                labels_out[mask_id] = "wall"
            # Small items still matter for occupancy and "next_to" placement.
            elif area_frac < 0.05:
                labels_out[mask_id] = _guess_furniture_by_color(image, mask)
            else:
                # Mid-area mid-height → generic furniture
                labels_out[mask_id] = _guess_furniture_by_color(image, mask)

        return labels_out

    # ------------------------------------------------------------------
    # Stage 4: Derive structural masks
    # ------------------------------------------------------------------

    def _derive_structural_masks(
        self, ctx: SceneContext, shape: tuple[int, int]
    ) -> None:
        h, w = shape
        masks = ctx.panoptic_masks
        labels = ctx.panoptic_labels

        floor_parts: list[np.ndarray] = []
        wall_parts: list[np.ndarray] = []
        ceiling_parts: list[np.ndarray] = []
        occ_parts: list[np.ndarray] = []

        for mid, lbl in labels.items():
            m = masks.get(mid)
            if m is None:
                continue
            if lbl == "floor":
                floor_parts.append(m)
            elif lbl == "wall":
                wall_parts.append(m)
            elif lbl == "ceiling":
                ceiling_parts.append(m)
            elif lbl in _OCCUPANCY_LABELS:
                occ_parts.append(m)

        # Fall back to position heuristics if masks completely absent
        if not floor_parts and not masks:
            floor_parts = [_position_floor_mask(h, w)]
        if not wall_parts and not masks:
            wall_parts = [_position_wall_mask(h, w)]

        ctx.floor_mask = _union_masks(floor_parts, h, w) if floor_parts else _position_floor_mask(h, w)
        ctx.ceiling_mask = _union_masks(ceiling_parts, h, w) if ceiling_parts else None
        ctx.wall_masks = wall_parts or [_position_wall_mask(h, w)]

        # Each wall segment gets its own entry; merge wall_parts into individual masks
        # (already one mask per SAM segment — keep them separate)
        ctx.wall_masks = wall_parts if wall_parts else [_position_wall_mask(h, w)]

        ctx.occupancy_mask = _union_masks(occ_parts, h, w) if occ_parts else np.zeros((h, w), dtype=bool)

        if ctx.floor_mask is not None:
            ctx.free_floor_mask = ctx.floor_mask & ~ctx.occupancy_mask
        else:
            ctx.free_floor_mask = None

    # ------------------------------------------------------------------
    # Stage 5: Vanishing points
    # ------------------------------------------------------------------

    def _compute_vanishing_points(
        self, image: np.ndarray
    ) -> list[tuple[float, float]]:
        try:
            return _detect_vanishing_points(image)
        except Exception as exc:
            logger.warning("[SCENE] VP detection failed (%s).", exc)
            return []

    # ------------------------------------------------------------------
    # Stage 6: Camera intrinsics (Caprile-Torre simplified)
    # ------------------------------------------------------------------

    def _estimate_intrinsics(
        self,
        vp1: tuple[float, float],
        vp2: tuple[float, float],
        shape: tuple[int, ...],
    ) -> dict[str, float]:
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0
        x1, y1 = vp1
        x2, y2 = vp2
        # f² = –(v1 – c) · (v2 – c)
        dot = (x1 - cx) * (x2 - cx) + (y1 - cy) * (y2 - cy)
        f_sq = -dot
        f_px = float(np.sqrt(max(f_sq, 1.0)))
        return {"fx": f_px, "fy": f_px, "cx": cx, "cy": cy, "f_px": f_px}

    # ------------------------------------------------------------------
    # Stage 7: Anchor candidates
    # ------------------------------------------------------------------

    def _build_anchor_candidates(self, ctx: SceneContext) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        h, w = ctx.image.shape[:2]
        depth = ctx.depth_map
        ffm = ctx.free_floor_mask

        if ffm is None or ffm.sum() == 0:
            # No free floor info — fall back to a 3×3 grid in the lower half
            for r in [0.60, 0.72, 0.84]:
                for c in [0.25, 0.50, 0.75]:
                    x, y = int(c * w), int(r * h)
                    candidates.append(_make_anchor(x, y, depth, "grid_fallback"))
            return candidates

        # Centroid of free floor
        ys, xs = np.where(ffm)
        cx, cy = int(xs.mean()), int(ys.mean())
        candidates.append(_make_anchor(cx, cy, depth, "floor_centroid"))

        # 3×3 grid sampled within free-floor bounding box
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        for ri in range(3):
            for ci in range(3):
                gx = x_min + int((x_max - x_min) * (ci + 0.5) / 3)
                gy = y_min + int((y_max - y_min) * (ri + 0.5) / 3)
                gy_clamp = max(0, min(h - 1, gy))
                gx_clamp = max(0, min(w - 1, gx))
                if ffm[gy_clamp, gx_clamp]:
                    candidates.append(_make_anchor(gx_clamp, gy_clamp, depth, "grid"))

        # Floor-wall boundary points (top edge of floor mask)
        boundary = _floor_wall_boundary(ctx.floor_mask, ffm, w)
        for bx, by in boundary:
            candidates.append(_make_anchor(bx, by, depth, "boundary"))

        # De-duplicate and score (prefer deeper / lower in frame)
        seen: set[tuple[int, int]] = set()
        unique: list[dict[str, Any]] = []
        for c in candidates:
            key = (c["x"] // 20, c["y"] // 20)  # bucket ~20 px
            if key not in seen:
                seen.add(key)
                c["score"] = float(c["y"]) / h  # lower = better for placement
                unique.append(c)

        unique.sort(key=lambda c: -c["score"])
        return unique[:20]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sha256(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _union_masks(parts: list[np.ndarray], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    for m in parts:
        out |= m.astype(bool)
    return out


def _mask_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = image[y0:y1, x0:x1].copy()
    # Zero out pixels outside mask
    local_mask = mask[y0:y1, x0:x1]
    crop[~local_mask] = 0
    return crop


def _make_anchor(
    x: int, y: int, depth: np.ndarray | None, source: str
) -> dict[str, Any]:
    d = None
    if depth is not None and 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
        d = float(depth[y, x])
    return {"x": x, "y": y, "depth_m": d, "score": 0.5, "source": source}


def _position_floor_mask(h: int, w: int) -> np.ndarray:
    """Heuristic: lower 35 % of the image is the floor."""
    m = np.zeros((h, w), dtype=bool)
    m[int(h * 0.65):, :] = True
    return m


def _position_wall_mask(h: int, w: int) -> np.ndarray:
    """Heuristic: middle band (20–65 % height) is the back wall."""
    m = np.zeros((h, w), dtype=bool)
    m[int(h * 0.20): int(h * 0.65), :] = True
    return m


def _floor_wall_boundary(
    floor_mask: np.ndarray | None,
    free_floor_mask: np.ndarray,
    w: int,
    n_points: int = 5,
) -> list[tuple[int, int]]:
    """Return points along the top edge of the free-floor mask."""
    if free_floor_mask is None or free_floor_mask.sum() == 0:
        return []
    points: list[tuple[int, int]] = []
    step = max(1, w // (n_points + 1))
    for col in range(step, w - step, step):
        col_slice = free_floor_mask[:, col]
        rows = np.where(col_slice)[0]
        if len(rows):
            points.append((col, int(rows.min())))
    return points


def _guess_furniture_by_color(image: np.ndarray, mask: np.ndarray) -> str:
    """Heuristic fallback when CLIP labels are unavailable."""
    pixels = image[mask]
    if len(pixels) == 0:
        return "other"

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    img_h, img_w = mask.shape
    area = float(mask.sum())
    bbox_area = float(bbox_h * bbox_w)
    fill_ratio = area / max(bbox_area, 1.0)
    width_frac = bbox_w / max(img_w, 1)
    height_frac = bbox_h / max(img_h, 1)
    cy = float(ys.mean()) / max(img_h, 1)
    mean_rgb = pixels.mean(axis=0)

    if mean_rgb[1] > mean_rgb[0] + 20 and mean_rgb[1] > mean_rgb[2] + 10:
        return "plant"

    if width_frac > 0.22 and height_frac > 0.14 and cy > 0.50:
        return "bed"
    if height_frac > 0.18 and width_frac < 0.12 and cy > 0.35:
        return "lamp" if fill_ratio < 0.42 else "bookshelf"
    if width_frac > 0.10 and height_frac < 0.12 and cy > 0.50:
        return "table"
    if width_frac > 0.10 and height_frac > 0.12 and cy > 0.45:
        return "chair" if fill_ratio < 0.58 else "cabinet"
    if mean_rgb.mean() < 80:
        return "cabinet"
    return "other"


def _resolve_local_clip_assets() -> tuple[str, str, str]:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots"
    if not root.exists():
        raise FileNotFoundError("No local cache found for openai/clip-vit-base-patch32.")

    processor_dir: Path | None = None
    config_dir: Path | None = None
    weights_file: Path | None = None

    for snapshot in sorted(root.iterdir(), reverse=True):
        if not snapshot.is_dir():
            continue
        if processor_dir is None and (snapshot / "preprocessor_config.json").exists():
            processor_dir = snapshot
        if config_dir is None and (snapshot / "config.json").exists():
            config_dir = snapshot
        if weights_file is None and (snapshot / "model.safetensors").exists():
            weights_file = snapshot / "model.safetensors"

    if processor_dir is None or config_dir is None or weights_file is None:
        raise FileNotFoundError(
            "The local CLIP cache is incomplete. Expected preprocessor files, config.json, and model.safetensors."
        )

    return str(processor_dir), str(config_dir), str(weights_file)


def _is_cuda_oom_message(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


# ---------------------------------------------------------------------------
# Vanishing-point detection
# ---------------------------------------------------------------------------

def _detect_vanishing_points(
    image: np.ndarray,
    min_votes: int = 3,
) -> list[tuple[float, float]]:
    """Estimate up to 3 vanishing points via HoughLinesP + RANSAC voting."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, min(h, w) // 12),
        maxLineGap=10,
    )
    if lines is None or len(lines) < 4:
        return []

    segs = lines[:, 0, :]  # (N, 4): x1, y1, x2, y2

    # Group lines by angle into H / V / diag buckets
    groups = _group_lines_by_angle(segs)

    vps: list[tuple[float, float]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        vp = _ransac_vp(group, w, h, min_votes=min_votes)
        if vp is not None:
            vps.append(vp)

    # Sort by proximity to image interior (closer to image center = more reliable)
    cx, cy = w / 2.0, h / 2.0
    vps.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
    return vps[:3]


def _group_lines_by_angle(segs: np.ndarray) -> dict[str, list[np.ndarray]]:
    groups: dict[str, list[np.ndarray]] = {
        "horizontal": [], "vertical": [], "diag_pos": [], "diag_neg": [],
    }
    for seg in segs:
        x1, y1, x2, y2 = seg
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 180
        if angle < 22.5 or angle >= 157.5:
            groups["horizontal"].append(seg)
        elif 67.5 <= angle < 112.5:
            groups["vertical"].append(seg)
        elif 22.5 <= angle < 67.5:
            groups["diag_pos"].append(seg)
        else:
            groups["diag_neg"].append(seg)
    return groups


def _line_intersection(s1: np.ndarray, s2: np.ndarray) -> tuple[float, float] | None:
    """Line–line intersection of two segments (infinite lines)."""
    x1, y1, x2, y2 = s1.astype(float)
    x3, y3, x4, y4 = s2.astype(float)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return ix, iy


def _ransac_vp(
    group: list[np.ndarray],
    w: int,
    h: int,
    min_votes: int = 3,
    n_iter: int = 200,
    inlier_px: float = 15.0,
) -> tuple[float, float] | None:
    """RANSAC-style VP estimation from a group of line segments."""
    import random

    best_vp: tuple[float, float] | None = None
    best_count = 0

    segs = group
    n = len(segs)
    if n < 2:
        return None

    for _ in range(min(n_iter, n * (n - 1) // 2)):
        i, j = random.sample(range(n), 2)
        pt = _line_intersection(segs[i], segs[j])
        if pt is None:
            continue
        vx, vy = pt
        # Count inliers: lines whose distance from vp is < inlier_px
        count = 0
        for seg in segs:
            if _dist_point_to_line(vx, vy, seg) < inlier_px:
                count += 1
        if count > best_count:
            best_count = count
            best_vp = (vx, vy)

    if best_count < min_votes:
        return None
    return best_vp


def _dist_point_to_line(px: float, py: float, seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg.astype(float)
    dx, dy = x2 - x1, y2 - y1
    denom = np.sqrt(dx * dx + dy * dy)
    if denom < 1e-6:
        return float(np.hypot(px - x1, py - y1))
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom
