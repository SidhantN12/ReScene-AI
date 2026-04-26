"""
ReScene AI — Segmentation wrapper (SAM ViT-H).

Provides:
  - Automatic mask generation (SamAutomaticMaskGenerator) — full scene
  - Prompted segmentation (SamPredictor) — point / box guided
  - segment_all() → {label: binary_mask} dict for downstream use
  - Mask post-processing: area filtering + overlap NMS

VRAM: ~3.5 GB with fp16 image encoder on RTX 4080.
Checkpoint: models/sam_vit_h_4b8939.pth (2.4 GB)
  python download_models.py   # downloads automatically
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config import Config

logger = logging.getLogger(__name__)

# Minimum mask area as fraction of total image pixels — discard noise fragments
_MIN_AREA_RATIO: float = 0.002
# Suppress a mask if it overlaps more than this fraction of a higher-area mask
_MAX_OVERLAP_RATIO: float = 0.85


class SegmentationWrapper:
    """SAM ViT-H segmentation model wrapper.

    Follows the standard ModelWrapper interface:
        load() / unload() / is_loaded() / predict()

    Additional public method:
        segment_all(image) → dict[str, np.ndarray]
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None
        self._predictor: Any = None
        self._use_real: bool = False

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load SAM ViT-H checkpoint onto GPU/CPU.

        Uses fp16 for the image encoder (the largest submodule, ~2 GB)
        while keeping the mask decoder and prompt encoder in fp32 for
        numerical stability.  Total VRAM: ~3.5 GB on RTX 4080.
        """
        checkpoint: Path = self._config.models.sam_checkpoint
        device: str = self._config.device.device
        logger.info("Loading SAM ViT-H from %s on %s", checkpoint, device)

        try:
            from segment_anything import (
                sam_model_registry,
                SamPredictor,
                SamAutomaticMaskGenerator,
            )
        except ImportError as exc:
            raise ImportError(
                "segment-anything is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}\n"
                "  Run: python download_models.py"
            )

        sam = sam_model_registry[self._config.models.sam_model_type](
            checkpoint=str(checkpoint)
        )
        sam.to(device=device)

        # Apply fp16 only to the image encoder (3× larger than decoder)
        if device == "cuda":
            sam.image_encoder = sam.image_encoder.half()

        self._model = sam
        self._predictor = SamPredictor(sam)
        self._use_real = True
        logger.info("SAM ViT-H loaded (real weights).")

    def unload(self) -> None:
        """Free GPU memory held by SAM."""
        self._model = None
        self._predictor = None
        self._use_real = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image: np.ndarray,
        *,
        points: list[tuple[int, int]] | None = None,
        point_labels: list[int] | None = None,
        box: tuple[int, int, int, int] | None = None,
        automatic: bool = False,
    ) -> dict[str, Any]:
        """Run segmentation inference.

        Parameters
        ----------
        image        : H×W×3 uint8 RGB
        points       : list of (x, y) foreground / background prompt points
        point_labels : 1=foreground, 0=background — one per point
        box          : (x0, y0, x1, y1) bounding-box prompt
        automatic    : if True, ignore prompts and segment the full scene

        Returns
        -------
        dict with:
            "masks"     : list[np.ndarray]  — (H,W) uint8, 255=object
            "scores"    : list[float]       — predicted IoU per mask
            "boxes"     : list[tuple]       — (x0,y0,x1,y1) per mask
            "labels"    : list[str]         — "object_N" sorted by area
            "label_map" : dict[str, ndarray]— {label: mask} convenience view
        """
        if not self.is_loaded():
            raise RuntimeError("SAM not loaded. Call load() first.")

        h, w = image.shape[:2]
        image = np.ascontiguousarray(image)

        if self._use_real:
            if automatic or (points is None and box is None):
                return self._predict_automatic(image)
            else:
                return self._predict_prompted(image, points, point_labels, box)

        # ---------- placeholder (package not installed) ----------
        dummy = np.ones((h, w), dtype=np.uint8) * 255
        result = {
            "masks": [dummy],
            "scores": [0.99],
            "boxes": [(0, 0, w, h)],
            "labels": ["object_0"],
            "label_map": {"object_0": dummy},
        }
        return result

    # ------------------------------------------------------------------
    # Convenience: {label: mask} dict
    # ------------------------------------------------------------------

    def segment_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Return {label: binary_mask} for every object found in *image*.

        Labels are "object_0", "object_1" … sorted by area descending.
        """
        result = self.predict(image, automatic=True)
        return result["label_map"]

    # ------------------------------------------------------------------
    # Internal: automatic mode
    # ------------------------------------------------------------------

    def _predict_automatic(self, image: np.ndarray) -> dict[str, Any]:
        from segment_anything import SamAutomaticMaskGenerator

        h, w = image.shape[:2]
        min_area_px = int(_MIN_AREA_RATIO * h * w)

        generator = SamAutomaticMaskGenerator(
            model=self._model,
            points_per_side=self._config.inference.sam_points_per_side,
            pred_iou_thresh=self._config.inference.sam_pred_iou_thresh,
            stability_score_thresh=self._config.inference.sam_stability_score_thresh,
            box_nms_thresh=self._config.inference.sam_box_nms_thresh,
            min_mask_region_area=min_area_px,
        )

        device = self._config.device.device
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device == "cuda"
            else torch.no_grad()
        )
        with torch.no_grad(), autocast_ctx:
            annotations = generator.generate(image)

        # Sort by area descending (largest objects first)
        annotations.sort(key=lambda a: a["area"], reverse=True)
        annotations = _suppress_overlapping(annotations)

        masks, scores, boxes = [], [], []
        for ann in annotations:
            m = ann["segmentation"].astype(np.uint8) * 255
            masks.append(m)
            scores.append(float(ann["predicted_iou"]))
            x, y, bw, bh = ann["bbox"]
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

        labels = [f"object_{i}" for i in range(len(masks))]
        label_map = dict(zip(labels, masks))
        return {"masks": masks, "scores": scores, "boxes": boxes,
                "labels": labels, "label_map": label_map}

    # ------------------------------------------------------------------
    # Internal: prompted mode
    # ------------------------------------------------------------------

    def _predict_prompted(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]] | None,
        point_labels: list[int] | None,
        box: tuple[int, int, int, int] | None,
    ) -> dict[str, Any]:
        device = self._config.device.device
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device == "cuda"
            else torch.no_grad()
        )
        with torch.no_grad(), autocast_ctx:
            self._predictor.set_image(image)
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_lbls = np.array(point_labels, dtype=np.int32) if point_labels else None
            box_arr = np.array(box, dtype=np.float32) if box else None

            raw_masks, raw_scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_lbls,
                box=box_arr,
                multimask_output=True,
            )

        masks, scores, boxes, labels = [], [], [], []
        for i, (m, s) in enumerate(zip(raw_masks, raw_scores)):
            bin_mask = m.astype(np.uint8) * 255
            ys, xs = np.where(bin_mask)
            if len(ys) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            masks.append(bin_mask)
            scores.append(float(s))
            boxes.append(bbox)
            labels.append(f"object_{i}")

        label_map = dict(zip(labels, masks))
        return {"masks": masks, "scores": scores, "boxes": boxes,
                "labels": labels, "label_map": label_map}


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _suppress_overlapping(
    annotations: list[dict],
    max_overlap: float = _MAX_OVERLAP_RATIO,
) -> list[dict]:
    """NMS-style suppression: discard a mask if it is largely contained inside
    a previously accepted (larger-area) mask."""
    kept: list[dict] = []
    for ann in annotations:
        m = ann["segmentation"]
        area = ann["area"]
        if area == 0:
            continue
        dominated = False
        for k in kept:
            intersection = np.logical_and(m, k["segmentation"]).sum()
            if intersection / area > max_overlap:
                dominated = True
                break
        if not dominated:
            kept.append(ann)
    return kept


def draw_segmentation_overlay(
    image: np.ndarray,
    result: dict[str, Any],
    alpha: float = 0.45,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Return an RGB visualization of all masks drawn over *image*.

    Parameters
    ----------
    image       : H×W×3 uint8 RGB
    result      : output of SegmentationWrapper.predict()
    alpha       : mask transparency (0=invisible, 1=opaque)
    draw_boxes  : overlay bounding boxes
    draw_labels : overlay "object_N" text labels

    Returns
    -------
    overlay : H×W×3 uint8 RGB
    """
    rng = np.random.default_rng(42)
    overlay = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    masks = result.get("masks", [])
    labels = result.get("labels", [f"object_{i}" for i in range(len(masks))])
    boxes = result.get("boxes", [])

    colours = rng.integers(64, 255, size=(len(masks), 3), dtype=np.uint8)

    for mask, colour in zip(masks, colours):
        bin_mask = mask > 127
        coloured = np.zeros((h, w, 3), dtype=np.float32)
        coloured[bin_mask] = colour.astype(np.float32)
        overlay[bin_mask] = (
            overlay[bin_mask] * (1 - alpha) + coloured[bin_mask] * alpha
        )

    canvas = overlay.clip(0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for i, (box, label, colour) in enumerate(zip(boxes, labels, colours)):
        if draw_boxes and box:
            x0, y0, x1, y1 = box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour.tolist(), 2)
        if draw_labels and box:
            x0, y0 = box[0], box[1]
            cv2.putText(
                canvas, label,
                (x0, max(y0 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                colour.tolist(), 1, cv2.LINE_AA,
            )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
