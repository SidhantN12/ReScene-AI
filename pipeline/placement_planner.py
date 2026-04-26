"""Floor-aware placement planner for ReScene AI v2 (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np

from pipeline.object_size_priors import canonical_object_class, object_size_for
from pipeline.scene_context import SceneContext


@dataclass
class PlacementRequest:
    object_class: str
    object_size_estimate: tuple[float, float, float]
    location_hint: str | tuple[float, float]
    clearance_required: float = 0.3


@dataclass
class PlacementResult:
    position_px: tuple[int, int]
    footprint_mask: np.ndarray
    depth_at_position: float
    scale_factor: float
    facing_direction: tuple[float, float]
    confidence: float
    reasoning: str
    clearance_mask: np.ndarray | None = None
    score_heatmap: np.ndarray | None = None


class OccupancyTracker:
    """Tracks reserved floor regions across consecutive add operations."""

    def __init__(self, scene: SceneContext) -> None:
        self.scene = scene
        h, w = scene.image.shape[:2]
        base_free = scene.free_floor_mask
        if base_free is None:
            base_free = np.zeros((h, w), dtype=bool)
        self._base_free_floor = base_free.astype(bool).copy()
        self._reserved_mask = np.zeros((h, w), dtype=bool)
        self.reset()

    @property
    def reserved_mask(self) -> np.ndarray:
        return self._reserved_mask.copy()

    def reserve(self, footprint_mask: np.ndarray) -> None:
        self._reserved_mask |= footprint_mask.astype(bool)
        self.scene.free_floor_mask = self._base_free_floor & ~self._reserved_mask

    def release(self, footprint_mask: np.ndarray) -> None:
        self._reserved_mask &= ~footprint_mask.astype(bool)
        self.scene.free_floor_mask = self._base_free_floor & ~self._reserved_mask

    def reset(self) -> None:
        self._reserved_mask[:] = False
        self.scene.free_floor_mask = self._base_free_floor.copy()


class PlacementPlanner:
    """Plans floor-aware object placement using a cached SceneContext."""

    def __init__(self) -> None:
        self.last_debug: dict[str, Any] = {}

    def plan(self, scene: SceneContext, request: PlacementRequest) -> PlacementResult:
        image_h, image_w = scene.image.shape[:2]
        free_floor = _effective_free_floor(scene)
        if free_floor.sum() == 0:
            raise ValueError("No free floor region available for placement.")

        canonical_class = canonical_object_class(request.object_class)
        size_m = request.object_size_estimate or object_size_for(canonical_class)
        base_depth = _scene_median_depth(scene)
        fx = float(scene.camera_intrinsics_estimate.get("fx", image_w * 1.15))
        fy = float(scene.camera_intrinsics_estimate.get("fy", fx))

        candidates = _collect_candidates(scene, request, canonical_class)
        if not candidates:
            raise ValueError("Planner could not generate any placement candidates.")

        reasoning_lines = [
            f"class={canonical_class}",
            f"size_m=({size_m[0]:.2f}, {size_m[1]:.2f}, {size_m[2]:.2f})",
            f"hint={request.location_hint}",
            f"free_floor_px={int(free_floor.sum())}",
            f"initial_candidates={len(candidates)}",
        ]
        allowed_clearance_region = free_floor | _wall_union(scene)

        scored: list[dict[str, Any]] = []
        heatmap = np.zeros((image_h, image_w), dtype=np.float32)

        for candidate in candidates:
            x = int(np.clip(candidate["x"], 0, image_w - 1))
            y = int(np.clip(candidate["y"], 0, image_h - 1))
            depth_here = _depth_at(scene, x, y, base_depth)
            px_per_m = max(10.0, fx / max(depth_here, 0.2))
            footprint_w_px = max(8, int(round(size_m[0] * px_per_m)))
            footprint_d_px = max(8, int(round(size_m[1] * px_per_m * 0.30)))
            clearance_px = max(4, int(round(request.clearance_required * px_per_m * 0.20)))
            footprint = _footprint_mask(
                image_h, image_w, x, y,
                footprint_w_px, footprint_d_px,
            )
            clearance_mask = _footprint_mask(
                image_h, image_w, x, y,
                footprint_w_px + clearance_px * 2,
                footprint_d_px + clearance_px * 2,
            )

            if not _mask_within(footprint, free_floor):
                continue
            if not _mask_within(clearance_mask, allowed_clearance_region):
                continue

            score, details = _score_candidate(
                scene=scene,
                request=request,
                candidate=(x, y),
                canonical_class=canonical_class,
                depth_here=depth_here,
                source=candidate.get("source", "candidate"),
            )
            scored.append({
                "x": x,
                "y": y,
                "depth": depth_here,
                "px_per_m": px_per_m,
                "footprint": footprint,
                "clearance": clearance_mask,
                "score": score,
                "details": details,
                "source": candidate.get("source", "candidate"),
            })
            heatmap = np.maximum(
                heatmap,
                _candidate_heat(heatmap.shape, x, y, score),
            )

        reasoning_lines.append(f"valid_candidates={len(scored)}")
        if not scored:
            raise ValueError(
                "Planner found no placement large enough for the requested footprint and clearance."
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        chosen = scored[0]
        facing_direction = _resolve_facing_direction(
            scene,
            request,
            chosen["x"],
            chosen["y"],
        )
        confidence = float(np.clip(chosen["score"], 0.0, 1.0))

        reasoning_lines.extend([
            f"chosen_source={chosen['source']}",
            f"chosen_position=({chosen['x']}, {chosen['y']})",
            f"depth_at_position={chosen['depth']:.3f}m",
            f"scale_factor_px_per_m={chosen['px_per_m']:.2f}",
            f"score={chosen['score']:.3f}",
            f"details={chosen['details']}",
            f"facing=({facing_direction[0]:.3f}, {facing_direction[1]:.3f})",
        ])

        self.last_debug = {
            "heatmap": _normalize_heatmap(heatmap),
            "candidates": scored,
            "request": request,
            "chosen": chosen,
        }
        return PlacementResult(
            position_px=(chosen["x"], chosen["y"]),
            footprint_mask=chosen["footprint"].astype(np.uint8),
            depth_at_position=float(chosen["depth"]),
            scale_factor=float(chosen["px_per_m"]),
            facing_direction=facing_direction,
            confidence=confidence,
            reasoning="\n".join(reasoning_lines),
            clearance_mask=chosen["clearance"].astype(np.uint8),
            score_heatmap=self.last_debug["heatmap"],
        )


def _effective_free_floor(scene: SceneContext) -> np.ndarray:
    h, w = scene.image.shape[:2]
    if scene.free_floor_mask is not None:
        return scene.free_floor_mask.astype(bool)
    if scene.floor_mask is not None:
        return scene.floor_mask.astype(bool)
    mask = np.zeros((h, w), dtype=bool)
    mask[int(h * 0.62):, :] = True
    return mask


def _scene_median_depth(scene: SceneContext) -> float:
    if scene.depth_map is None:
        return 2.5
    depth = scene.depth_map.astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0.05)]
    if len(finite) == 0:
        return 2.5
    return float(np.median(finite))


def _depth_at(scene: SceneContext, x: int, y: int, fallback: float) -> float:
    if scene.depth_map is None:
        return fallback
    depth = scene.depth_map
    x = int(np.clip(x, 0, depth.shape[1] - 1))
    y = int(np.clip(y, 0, depth.shape[0] - 1))
    value = float(depth[y, x])
    if not np.isfinite(value) or value <= 0.05:
        return fallback
    return value


def _collect_candidates(
    scene: SceneContext,
    request: PlacementRequest,
    canonical_class: str,
) -> list[dict[str, Any]]:
    h, w = scene.image.shape[:2]
    free_floor = _effective_free_floor(scene)
    ys, xs = np.where(free_floor)
    if len(xs) == 0 or len(ys) == 0:
        return []

    candidates: list[dict[str, Any]] = []
    for anchor in scene.anchor_candidates:
        candidates.append({"x": int(anchor["x"]), "y": int(anchor["y"]), "source": anchor.get("source", "anchor")})

    candidates.append({"x": int(xs.mean()), "y": int(ys.mean()), "source": "free_floor_centroid"})
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    for row_frac in (0.28, 0.48, 0.68, 0.82):
        for col_frac in (0.18, 0.34, 0.50, 0.66, 0.82):
            gx = x_min + int((x_max - x_min) * col_frac)
            gy = y_min + int((y_max - y_min) * row_frac)
            if free_floor[np.clip(gy, 0, h - 1), np.clip(gx, 0, w - 1)]:
                candidates.append({"x": gx, "y": gy, "source": "free_floor_grid"})

    if isinstance(request.location_hint, tuple):
        x = int(np.clip(request.location_hint[0] * w, 0, w - 1))
        y = int(np.clip(request.location_hint[1] * h, 0, h - 1))
        candidates.insert(0, {"x": x, "y": y, "source": "raw_hint"})

    hint = request.location_hint if isinstance(request.location_hint, str) else "raw"
    hint_target = _hint_target_class(hint)

    if hint == "corner":
        for x, y in _room_corner_candidates(scene):
            candidates.insert(0, {"x": x, "y": y, "source": "corner"})
    elif hint == "against_wall":
        for x, y in _wall_adjacent_candidates(scene):
            candidates.insert(0, {"x": x, "y": y, "source": "against_wall"})
    elif hint.startswith("next_to:") and hint_target:
        target_mask = _find_object_mask(scene, hint_target)
        for x, y in _adjacent_candidates(target_mask):
            candidates.insert(0, {"x": x, "y": y, "source": f"next_to:{hint_target}"})
    elif hint.startswith("in_front_of:") and hint_target:
        target_mask = _find_object_mask(scene, hint_target)
        for x, y in _in_front_candidates(target_mask, free_floor):
            candidates.insert(0, {"x": x, "y": y, "source": f"in_front_of:{hint_target}"})
    elif hint == "left":
        candidates.insert(0, {"x": int(w * 0.18), "y": int(ys.mean()), "source": "left_bias"})
    elif hint == "right":
        candidates.insert(0, {"x": int(w * 0.82), "y": int(ys.mean()), "source": "right_bias"})
    elif hint == "center":
        candidates.insert(0, {"x": int(xs.mean()), "y": int(ys.mean()), "source": "center"})
    elif hint.startswith("facing:"):
        candidates.insert(0, {"x": int(xs.mean()), "y": int(ys.mean()), "source": "facing_center"})

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        x = int(np.clip(item["x"], 0, w - 1))
        y = int(np.clip(item["y"], 0, h - 1))
        key = (x // 12, y // 12)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"x": x, "y": y, "source": item["source"]})

    return deduped[:80]


def _score_candidate(
    scene: SceneContext,
    request: PlacementRequest,
    candidate: tuple[int, int],
    canonical_class: str,
    depth_here: float,
    source: str,
) -> tuple[float, str]:
    x, y = candidate
    h, w = scene.image.shape[:2]
    free_floor = _effective_free_floor(scene)
    ys, xs = np.where(free_floor)
    centroid = (float(xs.mean()), float(ys.mean()))

    score = 0.25
    notes: list[str] = [source]
    hint = request.location_hint if isinstance(request.location_hint, str) else "raw"

    if isinstance(request.location_hint, tuple):
        tx = request.location_hint[0] * w
        ty = request.location_hint[1] * h
        dist = math.hypot(x - tx, y - ty)
        score += max(0.0, 0.55 - dist / max(w, h))
        notes.append("raw_hint_distance")
        return float(np.clip(score, 0.0, 1.0)), ", ".join(notes)

    if hint == "center":
        dist = math.hypot(x - centroid[0], y - centroid[1])
        score += max(0.0, 0.60 - dist / max(w, h))
        notes.append("near_floor_centroid")
    elif hint == "left":
        score += max(0.0, 0.55 * (1.0 - x / max(w - 1, 1)))
        notes.append("left_bias")
    elif hint == "right":
        score += max(0.0, 0.55 * (x / max(w - 1, 1)))
        notes.append("right_bias")
    elif hint == "corner":
        corners = _room_corner_candidates(scene)
        if corners:
            dist = min(math.hypot(x - cx, y - cy) for cx, cy in corners)
            score += max(0.0, 0.70 - dist / max(w, h))
            notes.append("corner_proximity")
    elif hint == "against_wall":
        wall_dist = _distance_to_wall(scene, x, y)
        score += max(0.0, 0.65 - wall_dist / max(w, h / 2))
        notes.append("wall_proximity")
    elif hint.startswith("next_to:"):
        target = _hint_target_class(hint)
        target_mask = _find_object_mask(scene, target)
        if target_mask is not None:
            dist = _distance_to_mask(target_mask, x, y)
            score += max(0.0, 0.72 - dist / max(w, h))
            notes.append(f"next_to:{target}")
    elif hint.startswith("in_front_of:"):
        target = _hint_target_class(hint)
        target_mask = _find_object_mask(scene, target)
        if target_mask is not None:
            t_cx, t_cy = _mask_centroid(target_mask)
            t_depth = _depth_at(scene, t_cx, t_cy, depth_here + 0.5)
            x_align = 1.0 - min(1.0, abs(x - t_cx) / max(w * 0.35, 1))
            closer = 1.0 if depth_here < t_depth else max(0.0, 1.0 - (depth_here - t_depth) / max(t_depth, 0.2))
            below = 1.0 if y >= t_cy else 0.35
            score += 0.25 * x_align + 0.25 * closer + 0.10 * below
            notes.append(f"in_front_of:{target}")
    elif hint.startswith("facing:"):
        dist = math.hypot(x - centroid[0], y - centroid[1])
        score += max(0.0, 0.50 - dist / max(w, h))
        notes.append("facing_position_resolved")
    else:
        dist = math.hypot(x - centroid[0], y - centroid[1])
        score += max(0.0, 0.40 - dist / max(w, h))
        notes.append("default_floor_fit")

    # Mild preference for candidates lower in the image because they typically lie on visible floor.
    score += 0.10 * (y / max(h - 1, 1))
    return float(np.clip(score, 0.0, 1.0)), ", ".join(notes)


def _hint_target_class(hint: str) -> str:
    if ":" not in hint:
        return ""
    return canonical_object_class(hint.split(":", 1)[1])


def _room_corner_candidates(scene: SceneContext) -> list[tuple[int, int]]:
    free_floor = _effective_free_floor(scene)
    ys, xs = np.where(free_floor)
    if len(xs) == 0 or len(ys) == 0:
        return []
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    points = [
        (x_min + int((x_max - x_min) * 0.26), y_min + int((y_max - y_min) * 0.22)),
        (x_max - int((x_max - x_min) * 0.26), y_min + int((y_max - y_min) * 0.22)),
    ]
    return points


def _wall_adjacent_candidates(scene: SceneContext) -> list[tuple[int, int]]:
    free_floor = _effective_free_floor(scene)
    h, w = free_floor.shape
    points: list[tuple[int, int]] = []
    ys, xs = np.where(free_floor)
    if len(xs) == 0 or len(ys) == 0:
        return []
    x_min, x_max = int(xs.min()), int(xs.max())
    for col in np.linspace(0.12, 0.88, 5):
        x = x_min + int((x_max - x_min) * col)
        rows = np.where(free_floor[:, x])[0]
        if len(rows):
            points.append((x, int(rows.min() + min(24, max(6, (rows.max() - rows.min()) * 0.08)))))
    return points


def _adjacent_candidates(target_mask: np.ndarray | None) -> list[tuple[int, int]]:
    if target_mask is None or target_mask.sum() == 0:
        return []
    ys, xs = np.where(target_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cy = int(ys.mean())
    pad = max(24, int((x1 - x0) * 0.35))
    return [
        (x0 - pad, min(y1 + 16, target_mask.shape[0] - 1)),
        (x1 + pad, min(y1 + 16, target_mask.shape[0] - 1)),
        (int(xs.mean()), min(y1 + pad, target_mask.shape[0] - 1)),
    ]


def _in_front_candidates(target_mask: np.ndarray | None, free_floor: np.ndarray) -> list[tuple[int, int]]:
    if target_mask is None or target_mask.sum() == 0:
        return []
    ys, xs = np.where(target_mask)
    cx = int(xs.mean())
    x0, x1 = int(xs.min()), int(xs.max())
    y1 = int(ys.max())
    step = max(20, int((ys.max() - ys.min()) * 0.35))
    candidates: list[tuple[int, int]] = []
    x_positions = [
        x0 - 12,
        x0 + int((x1 - x0) * 0.25),
        cx,
        x1 - int((x1 - x0) * 0.25),
        x1 + 12,
    ]
    y_positions = [y1 + step, y1 + step * 2, y1 + step * 3]
    for y in y_positions:
        for x in x_positions:
            x = int(np.clip(x, 0, free_floor.shape[1] - 1))
            y = int(np.clip(y, 0, free_floor.shape[0] - 1))
            if free_floor[y, x]:
                candidates.append((x, y))
    return candidates


def _find_object_mask(scene: SceneContext, target_class: str) -> np.ndarray | None:
    target_class = canonical_object_class(target_class)
    masks: list[np.ndarray] = []
    label_map = {
        "floor_lamp": {"lamp"},
        "coffee_table": {"table"},
        "side_table": {"table", "cabinet"},
        "armchair": {"chair"},
        "dining_chair": {"chair"},
        "bookshelf": {"bookshelf", "cabinet"},
        "sofa": {"sofa"},
        "bed_queen": {"bed"},
        "plant_pot": {"plant"},
        "tv_stand": {"tv", "cabinet"},
    }
    wanted = label_map.get(target_class, {target_class})
    for mask_id, label in scene.panoptic_labels.items():
        if label in wanted and mask_id in scene.panoptic_masks:
            masks.append(scene.panoptic_masks[mask_id].astype(bool))
    if not masks:
        return None
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m
    return out


def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def _distance_to_mask(mask: np.ndarray, x: int, y: int) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("inf")
    return float(np.min(np.hypot(xs - x, ys - y)))


def _distance_to_wall(scene: SceneContext, x: int, y: int) -> float:
    if not scene.wall_masks:
        return float(scene.image.shape[1])
    wall_union = _wall_union(scene)
    ys, xs = np.where(wall_union)
    if len(xs) == 0:
        return float(scene.image.shape[1])
    return float(np.min(np.hypot(xs - x, ys - y)))


def _wall_union(scene: SceneContext) -> np.ndarray:
    wall_union = np.zeros(scene.image.shape[:2], dtype=bool)
    for wall in scene.wall_masks:
        wall_union |= wall.astype(bool)
    return wall_union


def _resolve_facing_direction(
    scene: SceneContext,
    request: PlacementRequest,
    x: int,
    y: int,
) -> tuple[float, float]:
    if isinstance(request.location_hint, str) and request.location_hint.startswith("facing:"):
        target = _hint_target_class(request.location_hint)
        target_mask = _find_object_mask(scene, target)
        if target_mask is not None:
            tx, ty = _mask_centroid(target_mask)
            return _unit_vector(tx - x, ty - y)
    return (0.0, -1.0)


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return (0.0, -1.0)
    return (float(dx / norm), float(dy / norm))


def _footprint_mask(
    h: int,
    w: int,
    x: int,
    y: int,
    footprint_w_px: int,
    footprint_d_px: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(x), int(max(0, y - footprint_d_px // 3)))
    axes = (max(2, footprint_w_px // 2), max(2, footprint_d_px // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _mask_within(mask: np.ndarray, container: np.ndarray) -> bool:
    mask_b = mask.astype(bool)
    return bool(np.all(container[mask_b]))


def _candidate_heat(shape: tuple[int, int], x: int, y: int, score: float) -> np.ndarray:
    heat = np.zeros(shape, dtype=np.float32)
    cv2.circle(heat, (int(x), int(y)), 20, float(score), -1)
    heat = cv2.GaussianBlur(heat, (0, 0), 12)
    return heat


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    if heatmap.max() <= 1e-6:
        return np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
