"""ReScene AI — general image utility functions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR, RGBA, or greyscale numpy array to RGB uint8."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    # Assume OpenCV BGR input if loaded via cv2.imread
    return image


def resize_long_edge(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize so the longer edge equals *max_size*, preserving aspect ratio."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_size:
        return image
    scale = max_size / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def resize_to_multiple(image: np.ndarray, modulo: int = 8) -> np.ndarray:
    """Pad image so H and W are multiples of *modulo* (required by some GANs)."""
    h, w = image.shape[:2]
    new_h = ((h + modulo - 1) // modulo) * modulo
    new_w = ((w + modulo - 1) // modulo) * modulo
    if new_h == h and new_w == w:
        return image
    padded = np.zeros((new_h, new_w, *image.shape[2:]), dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def numpy_to_pil(image: np.ndarray) -> PILImage.Image:
    """Convert H×W×3 uint8 numpy (RGB) → PIL Image."""
    return PILImage.fromarray(image.astype(np.uint8))


def pil_to_numpy(image: PILImage.Image) -> np.ndarray:
    """Convert PIL Image → H×W×3 uint8 numpy (RGB)."""
    return np.array(image.convert("RGB"))


def load_image(path: str | Path) -> np.ndarray:
    """Load image from *path* as H×W×3 uint8 RGB numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save H×W×3 uint8 RGB numpy array to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """Dilate a binary mask by *pixels* to ensure clean seam coverage."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, kernel)
