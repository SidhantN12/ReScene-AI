"""
ReScene AI — Pipeline orchestrator.

Owns one ModelManager and one instance of each wrapper.  Exposes five
high-level operations:

    remove(image, mask)                        → inpainted_image
    move(image, object_name, target_pos)       → composited_image
    add(image, furniture_image, target_pos)    → composited_image
    restyle(image, style_name)                 → restyled_image
    compound(image, instructions)              → final_image

Each operation follows the same memory protocol:
    load model → run inference → unload model → repeat

This keeps peak VRAM below the configured limit even when chaining steps.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config, config as default_config
from pipeline.model_manager import ModelManager
from pipeline.object_size_priors import canonical_object_class, object_size_for
from pipeline.placement_planner import (
    OccupancyTracker,
    PlacementPlanner,
    PlacementRequest,
    PlacementResult,
)
from pipeline.scene_context import SceneContext
from pipeline.segmentation import SegmentationWrapper
from pipeline.depth_estimation import DepthEstimationWrapper
from pipeline.inpainting import InpaintingWrapper
from pipeline.perspective_warp import PerspectiveWarpWrapper
from pipeline.shadow_generation import ShadowGenerationWrapper
from pipeline.style_transfer import StyleTransferWrapper
from pipeline.harmonization import HarmonizationWrapper
from utils.image_utils import resize_long_edge, ensure_rgb, numpy_to_pil, pil_to_numpy
from utils.composite_utils import alpha_composite, paste_with_mask, simple_color_match

logger = logging.getLogger(__name__)

# Type alias for target position
Position = tuple[int, int]   # (x, y) in pixels of the final output image


class PipelineOrchestrator:
    """Coordinates all ReScene AI operations through sequential model execution."""

    def __init__(self, config: Config = default_config, use_v2_planner: bool = False) -> None:
        self._cfg = config
        self._use_v2_planner = use_v2_planner
        self._mm = ModelManager(config)
        self._placement_planner = PlacementPlanner() if use_v2_planner else None
        self._register_wrappers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_wrappers(self) -> None:
        self._mm.register("sam",         SegmentationWrapper(self._cfg))
        self._mm.register("zoedepth",    DepthEstimationWrapper(self._cfg))
        self._mm.register("lama",        InpaintingWrapper(self._cfg, backend="lama"))
        self._mm.register("mat",         InpaintingWrapper(self._cfg, backend="mat"))
        self._mm.register("stgan",       PerspectiveWarpWrapper(self._cfg))
        self._mm.register("arshadowgan", ShadowGenerationWrapper(self._cfg))
        self._mm.register("spade",       StyleTransferWrapper(self._cfg))
        self._mm.register("iharmony4",   HarmonizationWrapper(self._cfg))

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def select_object(
        self,
        image: np.ndarray,
        click_point: tuple[int, int],
        *,
        additional_points: list[tuple[int, int]] | None = None,
        point_labels: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run SAM with a click point and return the best matching mask.

        Intended for the Remove tab's click-to-select flow.  The caller
        stores the returned mask and passes it to remove() when the user
        confirms the selection.

        Parameters
        ----------
        image            : H×W×3 uint8 RGB (original, before pipeline resize)
        click_point      : (x, y) in *original* image coordinates
        additional_points: extra SAM prompt points (default: none)
        point_labels     : 1=foreground, 0=background per additional point

        Returns
        -------
        dict with:
            "processed_image" : np.ndarray (H'×W'×3) — resized to max_image_size
            "mask"            : np.ndarray (H'×W') uint8 — selected object mask
            "overlay"         : np.ndarray (H'×W'×3) — red highlight on processed_image
            "score"           : float — SAM confidence
            "box"             : tuple (x0,y0,x1,y1) — bounding box in processed coords
            "scale_xy"        : (sx, sy) — scale from original to processed coords
        """
        logger.info("[SELECT] click=(%d, %d)", *click_point)

        # Resize image the same way remove() will — must use identical scaling
        processed = _prep(image, self._cfg)
        ph, pw = processed.shape[:2]
        oh, ow = image.shape[:2]
        sx = pw / ow
        sy = ph / oh

        # Rescale click point to processed image space
        cx = int(np.clip(click_point[0] * sx, 0, pw - 1))
        cy = int(np.clip(click_point[1] * sy, 0, ph - 1))

        pts = [(cx, cy)]
        lbls = [1]   # foreground
        if additional_points:
            for p in additional_points:
                pts.append((int(p[0] * sx), int(p[1] * sy)))
            lbls += (point_labels or [1] * len(additional_points))

        seg = self._mm.get("sam")
        seg_result = seg.predict(
            image=processed,
            points=pts,
            point_labels=lbls,
            automatic=False,
        )
        self._mm.unload_current()

        # Pick the best mask: highest score among masks with reasonable area
        masks = seg_result["masks"]
        scores = seg_result["scores"]
        boxes = seg_result["boxes"]
        if not masks:
            empty = np.zeros(processed.shape[:2], dtype=np.uint8)
            return {
                "processed_image": processed, "mask": empty,
                "overlay": processed, "score": 0.0,
                "box": (0, 0, pw, ph), "scale_xy": (sx, sy),
            }

        total_px = ph * pw
        best_idx = _pick_best_mask(masks, scores, total_px)
        mask = masks[best_idx]
        score = scores[best_idx]
        box = boxes[best_idx]

        overlay = _make_selection_overlay(processed, mask)
        logger.info("[SELECT] Done — score=%.3f area=%.1f%%", score,
                    mask.astype(bool).sum() / total_px * 100)
        return {
            "processed_image": processed,
            "mask": mask,
            "overlay": overlay,
            "score": score,
            "box": box,
            "scale_xy": (sx, sy),
        }

    def remove(
        self,
        image: np.ndarray,
        object_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove an object from the scene and fill the background.

        Parameters
        ----------
        image       : H×W×3 uint8 RGB
        object_mask : H×W uint8 — 255 where the object to remove is

        Returns
        -------
        inpainted : H×W×3 uint8 RGB — object gone, background filled
        """
        logger.info("[REMOVE] Starting remove operation.")
        image = _prep(image, self._cfg)

        # Resize mask to match processed image if caller passed original-size mask
        if object_mask.shape[:2] != image.shape[:2]:
            object_mask = cv2.resize(
                object_mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Step 1: Inpaint (dilation handled inside InpaintingWrapper.predict)
        inpainter = self._mm.get("lama")
        result = inpainter.predict(image=image, mask=object_mask, dilation_px=10)
        self._mm.unload_current()

        # Step 2: Harmonize the newly-filled region so it blends with surroundings
        harmonizer = self._mm.get("iharmony4")
        result = harmonizer.predict(composite=result, foreground_mask=object_mask)
        self._mm.unload_current()

        logger.info("[REMOVE] Done.")
        return result

    def move(
        self,
        image: np.ndarray,
        object_name: str,
        target_position: Position,
        source_mask: np.ndarray | None = None,
        scene_context: SceneContext | None = None,
    ) -> np.ndarray:
        """Move a named object to a new position in the scene.

        Parameters
        ----------
        image           : H×W×3 uint8 RGB
        object_name     : free-text description; used to select among SAM masks
        target_position : (x, y) pixel coordinate for the object's centre
        source_mask     : H×W uint8 — pre-computed mask (skips SAM if provided)
        scene_context   : pre-built SceneContext; depth_map reused to skip ZoeDepth

        Returns
        -------
        composited : H×W×3 uint8 RGB
        """
        logger.info("[MOVE] '%s' → %s", object_name, target_position)
        image = _prep(image, self._cfg)
        h, w = image.shape[:2]

        # Resize source_mask to processed image space if caller passed original-size mask
        if source_mask is not None and source_mask.shape[:2] != (h, w):
            source_mask = cv2.resize(
                source_mask, (w, h), interpolation=cv2.INTER_NEAREST,
            )

        # Step 1: Segment object (skip if mask provided)
        if source_mask is None:
            seg = self._mm.get("sam")
            seg_result = seg.predict(image=image, automatic=True)
            self._mm.unload_current()
            source_mask = _pick_mask_by_name(seg_result, object_name, image)

        # Extract object crop
        obj_rgba = _extract_object_rgba(image, source_mask)

        # Step 2: Depth estimation for scale correction (reuse cached depth if available)
        tx, ty = int(np.clip(target_position[0], 0, w - 1)), int(np.clip(target_position[1], 0, h - 1))
        if scene_context is not None and scene_context.depth_map is not None:
            depth_map = cv2.resize(
                scene_context.depth_map, (w, h), interpolation=cv2.INTER_LINEAR
            )
            depth_est = self._mm.get("zoedepth")
            scale_factor = depth_est.scale_factor_at(depth_map, ty, tx)
            self._mm.unload_current()
            logger.debug("[MOVE] Reused cached depth map from SceneContext.")
        else:
            depth_est = self._mm.get("zoedepth")
            depth_map = depth_est.predict(image=image)
            scale_factor = depth_est.scale_factor_at(depth_map, ty, tx)
            self._mm.unload_current()

        # Step 3: Inpaint source region
        inpainter = self._mm.get("lama")
        scene_clean = inpainter.predict(image=image, mask=source_mask)
        self._mm.unload_current()

        # Step 4: Perspective warp object to destination
        target_size = (int(obj_rgba.shape[1] * scale_factor), int(obj_rgba.shape[0] * scale_factor))
        target_size = (max(target_size[0], 4), max(target_size[1], 4))
        warper = self._mm.get("stgan")
        h_p = min(h, max(8, int(h * 0.3)))
        w_p = min(w, max(8, int(w * 0.3)))
        scene_patch = scene_clean[
            max(0, ty - h_p // 2): ty + h_p // 2,
            max(0, tx - w_p // 2): tx + w_p // 2,
        ]
        warped_rgba = warper.predict(
            foreground_rgba=obj_rgba,
            scene_patch=scene_patch,
            target_size=target_size,
            scale_factor=scale_factor,
        )
        self._mm.unload_current()

        # Step 5: Composite warped object onto clean scene
        result, obj_mask = paste_with_mask(scene_clean, warped_rgba, (tx, ty))

        # Step 5b: Colour-match the composited object to its new surroundings.
        # Sample the ring of background pixels just outside the object boundary,
        # then apply a Reinhard mean+std transfer to the object's RGB region.
        result = _color_match_composite(result, scene_clean, obj_mask)

        # Step 6: Shadow
        shadow_gen = self._mm.get("arshadowgan")
        result = shadow_gen.predict(image=result, object_mask=obj_mask)
        self._mm.unload_current()

        # Step 7: Harmonize
        harmonizer = self._mm.get("iharmony4")
        result = harmonizer.predict(composite=result, foreground_mask=obj_mask)
        self._mm.unload_current()

        logger.info("[MOVE] Done.")
        return result

    def add(
        self,
        image: np.ndarray,
        furniture_image: np.ndarray,
        target_position: Position,
        size_multiplier: float = 1.0,
        scene_context: SceneContext | None = None,
        object_class: str | None = None,
        placement_request: PlacementRequest | None = None,
        occupancy_tracker: OccupancyTracker | None = None,
    ) -> np.ndarray:
        """Insert a furniture item into the scene at *target_position*.

        Parameters
        ----------
        image           : H×W×3 uint8 RGB — the room
        furniture_image : H_f×W_f×3 or ×4 uint8 — product photo (RGBA or RGB)
        target_position : (x, y) pixel coordinate for the furniture's base centre
        scene_context   : pre-built SceneContext; depth_map reused to skip ZoeDepth

        Returns
        -------
        composited : H×W×3 uint8 RGB — room with furniture added
        """
        logger.info("[ADD] Inserting furniture at %s with size multiplier %.2f",
                    target_position, size_multiplier)
        image = _prep(image, self._cfg)
        h, w = image.shape[:2]

        # Ensure furniture has alpha channel
        if furniture_image.ndim == 3 and furniture_image.shape[2] == 3:
            # White/near-white background → transparent (all channels > 235)
            r, g, b = furniture_image[:, :, 0], furniture_image[:, :, 1], furniture_image[:, :, 2]
            is_bg = (r > 235) & (g > 235) & (b > 235)
            alpha = np.where(is_bg, np.uint8(0), np.uint8(255))
            furniture_rgba = np.dstack([furniture_image, alpha])
        else:
            furniture_rgba = furniture_image.copy()
        furniture_rgba = _crop_rgba_to_alpha(furniture_rgba)

        plan_result: PlacementResult | None = None
        plan_request = placement_request
        tracker = occupancy_tracker
        tx, ty = int(np.clip(target_position[0], 0, w - 1)), int(np.clip(target_position[1], 0, h - 1))

        if self._use_v2_planner and scene_context is not None and self._placement_planner is not None:
            scaled_scene = _resize_scene_context(scene_context, (h, w))
            tracker = tracker or OccupancyTracker(scaled_scene)
            if plan_request is None:
                canonical_class = canonical_object_class(object_class)
                size_prior = object_size_for(canonical_class)
                scaled_size = tuple(float(v) * max(0.25, float(size_multiplier)) for v in size_prior)
                plan_request = PlacementRequest(
                    object_class=canonical_class,
                    object_size_estimate=scaled_size,
                    location_hint=(tx / max(w, 1), ty / max(h, 1)),
                )
            else:
                plan_request = PlacementRequest(
                    object_class=canonical_object_class(plan_request.object_class),
                    object_size_estimate=tuple(float(v) for v in plan_request.object_size_estimate),
                    location_hint=plan_request.location_hint,
                    clearance_required=plan_request.clearance_required,
                )
            plan_result = self._placement_planner.plan(tracker.scene, plan_request)
            tx, ty = plan_result.position_px

        # Step 1: Depth estimation for scale (reuse cached depth if available)
        if plan_result is not None:
            depth_map = tracker.scene.depth_map if tracker is not None else None
            scale_factor = float(plan_result.scale_factor)
        elif scene_context is not None and scene_context.depth_map is not None:
            depth_map = cv2.resize(
                scene_context.depth_map, (w, h), interpolation=cv2.INTER_LINEAR
            )
            depth_est = self._mm.get("zoedepth")
            scale_factor = depth_est.scale_factor_at(depth_map, ty, tx) * max(0.25, float(size_multiplier))
            self._mm.unload_current()
            logger.debug("[ADD] Reused cached depth map from SceneContext.")
        else:
            depth_est = self._mm.get("zoedepth")
            depth_map = depth_est.predict(image=image)
            scale_factor = depth_est.scale_factor_at(depth_map, ty, tx) * max(0.25, float(size_multiplier))
            self._mm.unload_current()

        # Step 2: Perspective warp furniture
        fw, fh = furniture_rgba.shape[1], furniture_rgba.shape[0]
        if plan_result is not None and plan_request is not None:
            desired_height_px = max(8, int(round(plan_request.object_size_estimate[2] * plan_result.scale_factor)))
            target_size = (max(int(round(desired_height_px * fw / max(fh, 1))), 4), desired_height_px)
        else:
            target_size = (max(int(fw * scale_factor), 4), max(int(fh * scale_factor), 4))
        h_p = min(h, max(8, int(h * 0.3)))
        w_p = min(w, max(8, int(w * 0.3)))
        scene_patch = image[
            max(0, ty - h_p // 2): ty + h_p // 2,
            max(0, tx - w_p // 2): tx + w_p // 2,
        ]
        warper = self._mm.get("stgan")
        warped_rgba = warper.predict(
            foreground_rgba=furniture_rgba,
            scene_patch=scene_patch,
            target_size=target_size,
            scale_factor=scale_factor,
        )
        self._mm.unload_current()

        # Step 3: Composite
        paste_position = (tx, ty)
        if plan_result is not None:
            paste_position = (tx, max(0, ty - warped_rgba.shape[0] // 2))
        result, obj_mask = paste_with_mask(image, warped_rgba, paste_position)

        # Step 3.5: Colour-match the inserted object to its surroundings
        result = _color_match_composite(result, image, obj_mask)

        # Step 4: Shadow
        shadow_gen = self._mm.get("arshadowgan")
        result = shadow_gen.predict(image=result, object_mask=obj_mask)
        self._mm.unload_current()

        # Step 5: Harmonize
        harmonizer = self._mm.get("iharmony4")
        result = harmonizer.predict(composite=result, foreground_mask=obj_mask)
        self._mm.unload_current()

        if tracker is not None and plan_result is not None:
            tracker.reserve(plan_result.footprint_mask)

        logger.info("[ADD] Done.")
        return result

    def restyle(
        self,
        image: np.ndarray,
        style_name: str,
    ) -> np.ndarray:
        """Apply an interior design style to the entire scene.

        Parameters
        ----------
        image      : H×W×3 uint8 RGB
        style_name : one of the keys in config.STYLES

        Returns
        -------
        restyled : H×W×3 uint8 RGB
        """
        logger.info("[RESTYLE] Applying style: %s", style_name)
        image = _prep(image, self._cfg)

        # Step 1: Segmentation (optional — style_transfer.predict ignores the label map;
        # skip gracefully when SAM weights or the package are not available)
        seg_map = None
        try:
            seg = self._mm.get("sam")
            seg_result = seg.predict(image=image, automatic=True)
            seg_map = _masks_to_label_map(seg_result["masks"], image.shape[:2])
            self._mm.unload_current()
        except Exception as _seg_exc:
            logger.warning("[RESTYLE] SAM unavailable (%s) — skipping segmentation.", _seg_exc)

        # Step 2: Style transfer
        styler = self._mm.get("spade")
        styled = styler.predict(image=image, style_name=style_name, segmentation_map=seg_map)
        self._mm.unload_current()

        # Step 3: Harmonize full image (foreground_mask = all ones)
        harmonizer = self._mm.get("iharmony4")
        full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        result = harmonizer.predict(composite=styled, foreground_mask=full_mask)
        self._mm.unload_current()

        logger.info("[RESTYLE] Done.")
        return result

    def plan_add_placement(
        self,
        scene_context: SceneContext,
        object_class: str,
        location_hint: str | tuple[float, float],
        *,
        size_multiplier: float = 1.0,
        clearance_required: float = 0.3,
        occupancy_tracker: OccupancyTracker | None = None,
    ) -> PlacementResult:
        """Return a planner result without executing the add composite path."""
        if not self._use_v2_planner or self._placement_planner is None:
            raise RuntimeError("V2 placement planner is disabled on this orchestrator.")
        canonical_class = canonical_object_class(object_class)
        size_prior = object_size_for(canonical_class)
        scaled_size = tuple(float(v) * max(0.25, float(size_multiplier)) for v in size_prior)
        request = PlacementRequest(
            object_class=canonical_class,
            object_size_estimate=scaled_size,
            location_hint=location_hint,
            clearance_required=clearance_required,
        )
        tracker = occupancy_tracker or OccupancyTracker(scene_context)
        return self._placement_planner.plan(tracker.scene, request)

    def understand_scene(self, image: np.ndarray) -> dict[str, Any]:
        """Run SAM + ZoeDepth to produce a complete scene understanding.

        This is the entry-point for Phase 2.  Downstream operations (remove,
        move, add) call the individual wrappers directly, but this method
        bundles both results for the test script and for any caller that needs
        both segmentation and depth in one shot.

        Parameters
        ----------
        image : H×W×3 uint8 RGB

        Returns
        -------
        dict with:
            "image"       : resized input (H×W×3 uint8)
            "segmentation": full predict() result dict (masks, scores, boxes, labels, label_map)
            "depth_map"   : (H,W) float32 metric depth in metres
            "depth_stats" : dict with min/max/mean/median depth values
        """
        logger.info("[UNDERSTAND] Starting scene understanding.")
        image = _prep(image, self._cfg)

        seg_wrapper = self._mm.get("sam")
        seg_result = seg_wrapper.predict(image=image, automatic=True)
        self._mm.unload_current()
        logger.info("[UNDERSTAND] Segmentation: %d objects found.", len(seg_result["masks"]))

        depth_wrapper = self._mm.get("zoedepth")
        depth_map = depth_wrapper.predict(image=image)
        depth_stats = depth_wrapper.depth_stats(depth_map)
        self._mm.unload_current()
        logger.info("[UNDERSTAND] Depth: %.2f – %.2f m", depth_stats["min_m"], depth_stats["max_m"])

        logger.info("[UNDERSTAND] Done.")
        return {
            "image": image,
            "segmentation": seg_result,
            "depth_map": depth_map,
            "depth_stats": depth_stats,
        }

    def compound(
        self,
        image: np.ndarray,
        instructions: list[dict[str, Any]],
        *,
        stage_sink: list | None = None,
    ) -> np.ndarray:
        """Apply a sequence of operations from a list of instruction dicts.

        Each instruction dict must have an "operation" key matching one of:
            "remove", "move", "add", "restyle"
        plus the relevant keyword arguments for that operation.

        Parameters
        ----------
        stage_sink : optional list; if provided, each completed step appends
                     {"name": str, "image": np.ndarray, "time_s": float,
                      "vram_mb": float} so callers can display intermediate results.

        Example
        -------
            instructions = [
                {"operation": "remove", "object_mask": mask},
                {"operation": "add",    "furniture_image": img, "target_position": (300, 400)},
                {"operation": "restyle","style_name": "Scandinavian"},
            ]
        """
        import torch as _torch
        logger.info("[COMPOUND] Running %d operations.", len(instructions))
        result = image.copy()
        occupancy_tracker: OccupancyTracker | None = None
        occupancy_hash: str | None = None
        for i, instr in enumerate(instructions):
            op = instr.get("operation")
            kw = {k: v for k, v in instr.items() if k != "operation"}
            logger.info("[COMPOUND] Step %d/%d: %s", i + 1, len(instructions), op)
            t0 = time.perf_counter()
            if op == "remove":
                result = self.remove(result, **kw)
            elif op == "move":
                result = self.move(result, **kw)
            elif op == "add":
                scene_ctx = kw.get("scene_context")
                if self._use_v2_planner and scene_ctx is not None:
                    scaled_scene = _resize_scene_context(scene_ctx, _prep(result, self._cfg).shape[:2])
                    if occupancy_tracker is None or occupancy_hash != scaled_scene.image_hash:
                        occupancy_tracker = OccupancyTracker(scaled_scene)
                        occupancy_hash = scaled_scene.image_hash
                    kw["scene_context"] = occupancy_tracker.scene
                    kw["occupancy_tracker"] = occupancy_tracker
                result = self.add(result, **kw)
            elif op == "restyle":
                result = self.restyle(result, **kw)
            else:
                raise ValueError(f"Unknown operation: '{op}'")
            elapsed = time.perf_counter() - t0
            vram_mb = (
                _torch.cuda.memory_allocated() / 1024 ** 2
                if _torch.cuda.is_available()
                else 0.0
            )
            logger.info("[COMPOUND] Step %d done in %.2fs (VRAM %.0f MB)", i + 1, elapsed, vram_mb)
            if stage_sink is not None:
                stage_sink.append({
                    "name": f"Step {i + 1}: {op}",
                    "image": result.copy(),
                    "time_s": elapsed,
                    "vram_mb": vram_mb,
                })
        logger.info("[COMPOUND] Done.")
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        self._mm.unload_all()

    def __enter__(self) -> "PipelineOrchestrator":
        return self

    def __exit__(self, *_) -> None:
        self.teardown()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _prep(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Ensure RGB, cap long edge."""
    image = ensure_rgb(image)
    return resize_long_edge(image, cfg.inference.max_image_size)


def _extract_object_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop the tightest bounding box around *mask* and add alpha channel."""
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return np.zeros((8, 8, 4), dtype=np.uint8)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_rgb = image[y0:y1, x0:x1].copy()
    crop_alpha = mask[y0:y1, x0:x1].copy()
    return np.dstack([crop_rgb, crop_alpha])


def _crop_rgba_to_alpha(rgba: np.ndarray, pad: int = 4) -> np.ndarray:
    """Trim transparent borders so scaling uses the visible silhouette, not the canvas."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return rgba
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(rgba.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(rgba.shape[0], int(ys.max()) + 1 + pad)
    return rgba[y0:y1, x0:x1]


def _resize_scene_context(scene: SceneContext, target_shape: tuple[int, int]) -> SceneContext:
    """Resize a SceneContext to match a processed pipeline image shape."""
    th, tw = target_shape
    sh, sw = scene.image.shape[:2]
    if (sh, sw) == (th, tw):
        return scene

    def _resize_bool(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        resized = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)

    resized_masks = {
        mid: _resize_bool(mask)
        for mid, mask in scene.panoptic_masks.items()
    }
    resized_image = cv2.resize(scene.image, (tw, th), interpolation=cv2.INTER_AREA)
    depth_map = None
    if scene.depth_map is not None:
        depth_map = cv2.resize(scene.depth_map.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)

    sx = tw / max(sw, 1)
    sy = th / max(sh, 1)
    intr = dict(scene.camera_intrinsics_estimate)
    if intr:
        intr["fx"] = float(intr.get("fx", tw * 1.15)) * sx
        intr["fy"] = float(intr.get("fy", th * 1.15)) * sy
        intr["cx"] = float(intr.get("cx", sw / 2.0)) * sx
        intr["cy"] = float(intr.get("cy", sh / 2.0)) * sy
        intr["f_px"] = float(intr.get("f_px", tw * 1.15)) * ((sx + sy) * 0.5)

    return SceneContext(
        image=resized_image,
        image_hash=scene.image_hash,
        depth_map=depth_map,
        panoptic_masks={mid: mask for mid, mask in resized_masks.items() if mask is not None},
        panoptic_labels=dict(scene.panoptic_labels),
        floor_mask=_resize_bool(scene.floor_mask),
        wall_masks=[wm for wm in (_resize_bool(mask) for mask in scene.wall_masks) if wm is not None],
        ceiling_mask=_resize_bool(scene.ceiling_mask),
        occupancy_mask=_resize_bool(scene.occupancy_mask),
        free_floor_mask=_resize_bool(scene.free_floor_mask),
        vanishing_points=[(vx * sx, vy * sy) for vx, vy in scene.vanishing_points],
        camera_intrinsics_estimate=intr,
        anchor_candidates=[
            {
                **anchor,
                "x": int(round(anchor["x"] * sx)),
                "y": int(round(anchor["y"] * sy)),
            }
            for anchor in scene.anchor_candidates
        ],
        metadata=dict(scene.metadata),
    )


def _pick_mask_by_name(
    seg_result: dict[str, Any],
    object_name: str,
    image: np.ndarray,
) -> np.ndarray:
    """Return the highest-scoring mask (real impl would match object_name)."""
    masks = seg_result["masks"]
    scores = seg_result["scores"]
    if not masks:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def _masks_to_label_map(
    masks: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Convert a list of binary masks into a single integer label map."""
    label_map = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks):
        label_map[m > 127] = i + 1
    return label_map


def _pick_best_mask(
    masks: list[np.ndarray],
    scores: list[float],
    total_px: int,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.90,
) -> int:
    """Return the index of the best mask for a click-to-select operation.

    Prefers high SAM score, but ignores masks that are implausibly tiny or
    that cover almost the entire image (likely a spurious whole-scene mask).
    """
    best_idx = 0
    best_score = -1.0
    for i, (m, s) in enumerate(zip(masks, scores)):
        area_ratio = m.astype(bool).sum() / total_px
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx


def _color_match_composite(
    composite: np.ndarray,
    background: np.ndarray,
    obj_mask: np.ndarray,
    dilation_size: int | None = None,
) -> np.ndarray:
    """Colour-match the inserted object region to its new surroundings.

    Samples the ring of background pixels just outside *obj_mask*, then applies
    a per-channel mean+std transfer to the object's bounding-box crop.

    Parameters
    ----------
    composite  : H×W×3 uint8 RGB — full composited image
    background : H×W×3 uint8 RGB — clean scene before the object was added
    obj_mask   : H×W uint8 — 255 where the inserted object is
    dilation_size : kernel size for computing the surrounding ring;
                    defaults to max(15, h // 20)
    """
    if obj_mask.max() == 0:
        return composite

    ys, xs = np.where(obj_mask > 127)
    if len(ys) < 10:
        return composite

    h = composite.shape[0]
    dil_sz = dilation_size or max(15, h // 20)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_sz * 2 + 1, dil_sz * 2 + 1)
    )
    # Ring = dilated mask minus the object itself → background context around insertion
    surround = cv2.dilate(obj_mask, kernel)
    bg_sample_mask = np.where(obj_mask > 127, np.uint8(0), surround)

    if bg_sample_mask.max() == 0:
        return composite  # object fills the whole image — skip

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    obj_crop = composite[y0:y1, x0:x1].copy()
    matched_crop = simple_color_match(obj_crop, background, bg_sample_mask)

    # Blend using the soft object mask (not hard binary) for smooth edges
    soft_alpha = (obj_mask[y0:y1, x0:x1].astype(np.float32) / 255.0)[:, :, None]
    blended = matched_crop * soft_alpha + composite[y0:y1, x0:x1] * (1.0 - soft_alpha)

    out = composite.copy()
    out[y0:y1, x0:x1] = blended.clip(0, 255).astype(np.uint8)
    return out


def _make_selection_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colour: tuple[int, int, int] = (220, 50, 50),
    alpha: float = 0.50,
) -> np.ndarray:
    """Draw a coloured highlight over the selected object region.

    Also draws a dashed bounding box and a small crosshair centroid marker.
    Returns H×W×3 uint8 RGB.
    """
    overlay = image.copy().astype(np.float32)
    bin_mask = mask > 127
    h, w = image.shape[:2]

    colour_layer = np.zeros_like(image, dtype=np.float32)
    colour_layer[bin_mask] = colour
    overlay[bin_mask] = overlay[bin_mask] * (1 - alpha) + colour_layer[bin_mask] * alpha
    canvas = overlay.clip(0, 255).astype(np.uint8)

    # Bounding box
    ys, xs = np.where(bin_mask)
    if len(ys):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.rectangle(canvas_bgr, (x0, y0), (x1, y1), (50, 50, 220), 2)

        # Centroid crosshair
        cx, cy = int(xs.mean()), int(ys.mean())
        r = 8
        cv2.line(canvas_bgr, (cx - r, cy), (cx + r, cy), (50, 50, 220), 2)
        cv2.line(canvas_bgr, (cx, cy - r), (cx, cy + r), (50, 50, 220), 2)

        canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    return canvas
