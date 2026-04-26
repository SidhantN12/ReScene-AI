from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.object_size_priors import object_size_for
from pipeline.placement_planner import OccupancyTracker, PlacementPlanner, PlacementRequest
from pipeline.scene_context import SceneContext


def _rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _make_scene() -> SceneContext:
    h, w = 480, 640
    image = np.full((h, w, 3), 220, dtype=np.uint8)
    floor_mask = np.zeros((h, w), dtype=bool)
    floor_mask[220:, 60:580] = True

    left_wall = _rect_mask(h, w, 40, 70, 120, 250)
    back_wall = _rect_mask(h, w, 120, 60, 520, 220)
    right_wall = _rect_mask(h, w, 520, 70, 600, 250)

    sofa_mask = _rect_mask(h, w, 285, 205, 495, 290)
    table_mask = _rect_mask(h, w, 120, 340, 215, 390)

    occupancy = sofa_mask | table_mask
    free_floor = floor_mask & ~occupancy

    depth_y = np.linspace(4.2, 1.7, h, dtype=np.float32)[:, None]
    depth_map = np.repeat(depth_y, w, axis=1)

    anchors = [
        {"x": 165, "y": 275, "depth_m": float(depth_map[275, 165]), "score": 0.8, "source": "left_corner"},
        {"x": 475, "y": 275, "depth_m": float(depth_map[275, 475]), "score": 0.8, "source": "right_corner"},
        {"x": 320, "y": 385, "depth_m": float(depth_map[385, 320]), "score": 0.7, "source": "center"},
        {"x": 240, "y": 400, "depth_m": float(depth_map[400, 240]), "score": 0.6, "source": "left_floor"},
        {"x": 430, "y": 400, "depth_m": float(depth_map[400, 430]), "score": 0.6, "source": "right_floor"},
    ]

    return SceneContext(
        image=image,
        image_hash="synthetic-scene",
        depth_map=depth_map,
        panoptic_masks={1: sofa_mask, 2: table_mask},
        panoptic_labels={1: "sofa", 2: "table"},
        floor_mask=floor_mask,
        wall_masks=[left_wall, back_wall, right_wall],
        ceiling_mask=None,
        occupancy_mask=occupancy,
        free_floor_mask=free_floor,
        vanishing_points=[(320.0, -180.0), (900.0, 220.0)],
        camera_intrinsics_estimate={"fx": 720.0, "fy": 720.0, "cx": 320.0, "cy": 240.0, "f_px": 720.0},
        anchor_candidates=anchors,
        metadata={},
    )


class PlacementPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _make_scene()
        self.planner = PlacementPlanner()

    def test_lamp_in_corner_prefers_wall_intersection(self) -> None:
        request = PlacementRequest(
            object_class="floor_lamp",
            object_size_estimate=object_size_for("floor_lamp"),
            location_hint="corner",
        )
        result = self.planner.plan(self.scene, request)
        self.assertTrue(result.position_px[0] < 220 or result.position_px[0] > 430)
        self.assertLess(result.position_px[1], 285)

    def test_table_in_front_of_sofa_prefers_shallower_depth(self) -> None:
        request = PlacementRequest(
            object_class="coffee_table",
            object_size_estimate=object_size_for("coffee_table"),
            location_hint="in_front_of:sofa",
            clearance_required=0.1,
        )
        result = self.planner.plan(self.scene, request)
        sofa_ys, sofa_xs = np.where(self.scene.panoptic_masks[1])
        sofa_cx = int(sofa_xs.mean())
        sofa_cy = int(sofa_ys.mean())
        sofa_depth = float(self.scene.depth_map[sofa_cy, sofa_cx])
        self.assertLess(result.depth_at_position, sofa_depth)

    def test_chair_next_to_table_prefers_adjacent_space(self) -> None:
        request = PlacementRequest(
            object_class="dining_chair",
            object_size_estimate=object_size_for("dining_chair"),
            location_hint="next_to:coffee_table",
        )
        result = self.planner.plan(self.scene, request)
        table_mask = self.scene.panoptic_masks[2]
        distance = _distance_to_mask(table_mask, *result.position_px)
        self.assertLess(distance, 140.0)

    def test_reserved_regions_are_not_reused(self) -> None:
        tracker = OccupancyTracker(self.scene)
        request = PlacementRequest(
            object_class="floor_lamp",
            object_size_estimate=object_size_for("floor_lamp"),
            location_hint="center",
        )
        first = self.planner.plan(tracker.scene, request)
        tracker.reserve(first.footprint_mask)
        second = self.planner.plan(tracker.scene, request)
        overlap = np.logical_and(first.footprint_mask.astype(bool), second.footprint_mask.astype(bool))
        self.assertFalse(overlap.any())
        self.assertNotEqual(first.position_px, second.position_px)


def _distance_to_mask(mask: np.ndarray, x: int, y: int) -> float:
    ys, xs = np.where(mask)
    return float(np.min(np.hypot(xs - x, ys - y)))


if __name__ == "__main__":
    unittest.main()
