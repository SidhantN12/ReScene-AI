"""Object size priors for the v2 floor-aware placement planner."""

from __future__ import annotations

from typing import Final

OBJECT_SIZE_PRIORS: Final[dict[str, tuple[float, float, float]]] = {
    "floor_lamp": (0.4, 0.4, 1.6),
    "coffee_table": (1.1, 0.6, 0.45),
    "armchair": (0.85, 0.85, 0.9),
    "dining_chair": (0.5, 0.5, 0.9),
    "side_table": (0.5, 0.5, 0.6),
    "bookshelf": (0.8, 0.3, 1.8),
    "sofa": (2.0, 0.9, 0.85),
    "bed_queen": (1.6, 2.1, 0.6),
    "plant_pot": (0.3, 0.3, 0.8),
    "tv_stand": (1.6, 0.4, 0.5),
}

_ALIASES: Final[dict[str, str]] = {
    "lamp": "floor_lamp",
    "floor lamp": "floor_lamp",
    "standing lamp": "floor_lamp",
    "table lamp": "side_table",
    "coffee table": "coffee_table",
    "table": "coffee_table",
    "nightstand": "side_table",
    "side table": "side_table",
    "end table": "side_table",
    "arm chair": "armchair",
    "armchair": "armchair",
    "accent chair": "armchair",
    "chair": "dining_chair",
    "dining chair": "dining_chair",
    "bookshelf": "bookshelf",
    "book shelf": "bookshelf",
    "shelf": "bookshelf",
    "wall shelf": "bookshelf",
    "cupboard": "bookshelf",
    "cabinet": "bookshelf",
    "sofa": "sofa",
    "couch": "sofa",
    "loveseat": "sofa",
    "bed": "bed_queen",
    "queen bed": "bed_queen",
    "plant": "plant_pot",
    "potted plant": "plant_pot",
    "plant pot": "plant_pot",
    "tv stand": "tv_stand",
    "media console": "tv_stand",
    "console": "tv_stand",
}


def canonical_object_class(name: str | None) -> str:
    """Map free-form class names to a known size-prior key."""
    if not name:
        return "side_table"
    key = name.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key in OBJECT_SIZE_PRIORS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    words = key.split()
    for size_key in OBJECT_SIZE_PRIORS:
        if size_key.replace("_", " ") in key or key in size_key.replace("_", " "):
            return size_key
    for alias, canonical in _ALIASES.items():
        alias_words = set(alias.split())
        if alias_words and alias_words.issubset(words):
            return canonical
    return "side_table"


def object_size_for(name: str | None) -> tuple[float, float, float]:
    """Return a realistic default (width_m, depth_m, height_m)."""
    return OBJECT_SIZE_PRIORS[canonical_object_class(name)]

