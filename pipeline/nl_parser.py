"""
ReScene AI — Natural-language instruction parser for the Compound tab.

Converts sentences like:
    "remove the bookshelf, move the sofa left, add a lamp, make it scandinavian"
into a list of operation dicts that compound() can execute (or, for remove/move
which require a SAM mask, returns a human-readable warning).

No LLM required — rule-based regex matching against known styles and catalog items.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Position vocabulary  (text hint → (x_frac, y_frac) in [0, 1]²)
# ---------------------------------------------------------------------------

_POSITION_HINTS: dict[str, tuple[float, float]] = {
    "upper left":    (0.20, 0.25),
    "upper right":   (0.80, 0.25),
    "lower left":    (0.20, 0.82),
    "lower right":   (0.80, 0.82),
    "top left":      (0.20, 0.25),
    "top right":     (0.80, 0.25),
    "bottom left":   (0.20, 0.82),
    "bottom right":  (0.80, 0.82),
    "top center":    (0.50, 0.25),
    "top centre":    (0.50, 0.25),
    "bottom center": (0.50, 0.82),
    "bottom centre": (0.50, 0.82),
    "left":          (0.22, 0.65),
    "right":         (0.78, 0.65),
    "center":        (0.50, 0.65),
    "centre":        (0.50, 0.65),
    "middle":        (0.50, 0.60),
    "corner":        (0.15, 0.80),
    "background":    (0.50, 0.28),
    "foreground":    (0.50, 0.82),
    "front":         (0.50, 0.82),
    "back":          (0.50, 0.25),
    "far":           (0.50, 0.25),
    "near":          (0.50, 0.82),
    "near left":     (0.25, 0.82),
    "near right":    (0.75, 0.82),
    "far left":      (0.25, 0.25),
    "far right":     (0.75, 0.25),
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RESTYLE_PATS = [
    r"make\s+it\s+([\w][\w\s\-]*)",
    r"apply\s+([\w][\w\s\-]*?)\s*style",
    r"in\s+([\w][\w\s\-]*?)\s*style",
    r"(?:restyle|style)\s+(?:as\s+)?([\w][\w\s\-]*)",
    r"([\w][\w\s\-]*?)\s+(?:interior|design|aesthetic)",
]

_ADD_PATS = [
    r"(?:add|place|insert|put)\s+(?:a\s+|an\s+|the\s+)?([\w][\w\s]*)",
]

_REMOVE_PATS = [
    r"(?:remove|delete|erase|eliminate)\s+(?:the\s+)?([\w][\w\s]*)",
    r"get\s+rid\s+of\s+(?:the\s+)?([\w][\w\s]*)",
]

_MOVE_PATS = [
    r"(?:move|shift|relocate|push)\s+(?:the\s+)?([\w][\w\s]*)",
]

_ACTION_PREFIX_RE = re.compile(
    r"^(add|place|insert|put|remove|delete|erase|eliminate|move|shift|relocate|push|"
    r"restyle|style|apply)\b"
)

# Words that end an "object description" before a positional cue
_POS_STOPWORDS = {
    "to", "on", "in", "at", "from", "into", "onto", "toward", "towards",
    "the", "a", "an",
}

# ---------------------------------------------------------------------------
# Style aliases (lowercase → canonical style name)
# ---------------------------------------------------------------------------

_STYLE_ALIASES: dict[str, str] = {
    "scandi":           "Scandinavian",
    "scandinavian":     "Scandinavian",
    "nordic":           "Scandinavian",
    "hygge":            "Scandinavian",
    "industrial":       "Industrial",
    "factory":          "Industrial",
    "loft":             "Industrial",
    "bohemian":         "Bohemian",
    "boho":             "Bohemian",
    "eclectic":         "Bohemian",
    "mid-century":      "Mid-Century Modern",
    "mid century":      "Mid-Century Modern",
    "midcentury":       "Mid-Century Modern",
    "mid-century modern": "Mid-Century Modern",
    "retro":            "Mid-Century Modern",
    "fifties":          "Mid-Century Modern",
    "minimalist":       "Minimalist",
    "minimal":          "Minimalist",
    "clean":            "Minimalist",
    "modern":           "Minimalist",
    "coastal":          "Coastal",
    "beach":            "Coastal",
    "nautical":         "Coastal",
    "hamptons":         "Coastal",
    "ocean":            "Coastal",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_nl(
    text: str,
    catalog_items: list[str] | dict[str, str],
    styles: dict[str, Any],
    image_size: tuple[int, int] = (1024, 1024),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a natural-language instruction string into operation dicts.

    Parameters
    ----------
    text         : user's free-form instruction (e.g. "add a sofa, make it bohemian")
    catalog_items: list of catalog PNG filenames (e.g. ["armchair_1.png", ...])
    styles       : STYLES dict from config (keys are canonical style names)
    image_size   : (width, height) of the room image being edited

    Returns
    -------
    (instructions, warnings)
        instructions : list of dicts ready for compound(); "add" ops include
                       "furniture_image_path" (str) that the caller must resolve
                       to a numpy array before passing to compound().
        warnings     : list of human-readable strings for unexecutable ops
                       (remove/move which require a SAM mask click).
    """
    img_w, img_h = image_size

    # Build lookup tables
    catalog_lookup = _build_catalog_lookup(catalog_items)
    style_lookup = _build_style_lookup(styles)

    instructions: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Split into coarse clauses on comma / semicolon / "and then" / "then" / "after that"
    coarse_clauses = re.split(
        r"[,;]|\bthen\b|\band\s+then\b|\bafter\s+that\b|\bfinally\b",
        text.lower(),
    )
    clauses: list[str] = []
    for clause in coarse_clauses:
        clauses.extend(_expand_connected_actions(clause))

    for clause in clauses:
        _parse_clause(
            clause, img_w, img_h,
            catalog_lookup, style_lookup,
            instructions, warnings,
        )

    return instructions, warnings


# ---------------------------------------------------------------------------
# Internal clause parser
# ---------------------------------------------------------------------------

def _parse_clause(
    clause: str,
    img_w: int,
    img_h: int,
    catalog_lookup: dict[str, str],
    style_lookup: dict[str, str],
    instructions: list,
    warnings: list,
) -> None:
    # ── 0. Multiple add ops inside one clause ────────────────────────────
    if _looks_like_add_clause(clause):
        for add_clause in _split_add_continuations(clause):
            _parse_add_clause(add_clause, img_w, img_h, catalog_lookup, instructions, warnings)
        return

    # ── 1. Restyle ───────────────────────────────────────────────────────
    for pat in _RESTYLE_PATS:
        m = re.search(pat, clause)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            style = _match_style(candidate, style_lookup)
            if style:
                instructions.append({"operation": "restyle", "style_name": style})
                return

    # ── 2. Add ───────────────────────────────────────────────────────────
    if _parse_add_clause(clause, img_w, img_h, catalog_lookup, instructions, warnings):
        return

    # ── 3. Remove (needs SAM click — flag only) ──────────────────────────
    for pat in _REMOVE_PATS:
        m = re.search(pat, clause)
        if m:
            item = m.group(1).strip().rstrip(".")
            warnings.append(
                f"Remove '{item}' — removal requires clicking the object in the Remove tab "
                "or supplying an 'object_mask' array in JSON mode."
            )
            return

    # ── 4. Move (needs SAM click — flag only) ────────────────────────────
    for pat in _MOVE_PATS:
        m = re.search(pat, clause)
        if m:
            item = m.group(1).strip().rstrip(".")
            warnings.append(
                f"Move '{item}' — moving requires clicking the object in the Move tab "
                "or supplying a 'source_mask' array in JSON mode."
            )
            return

    # ── Unrecognised ─────────────────────────────────────────────────────
    if clause:
        warnings.append(f"Could not parse: '{clause}'")


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _build_style_lookup(styles: dict[str, Any]) -> dict[str, str]:
    """canonical_lower → canonical for each style, plus aliases."""
    lookup = {k.lower(): k for k in styles}
    for alias, canonical in _STYLE_ALIASES.items():
        if canonical in styles:
            lookup.setdefault(alias, canonical)
    return lookup


def _build_catalog_lookup(catalog_items: list[str] | dict[str, str]) -> dict[str, str]:
    """Build normalized text → filename lookup for catalog matching."""
    lookup: dict[str, str] = {}
    if isinstance(catalog_items, dict):
        for key, filename in catalog_items.items():
            norm = Path(key).stem.lower().replace("_", " ")
            lookup[norm] = filename
        return lookup

    for fname in catalog_items:
        stem = Path(fname).stem.lower().replace("_", " ")
        lookup[stem] = fname
    return lookup


def _expand_connected_actions(clause: str) -> list[str]:
    clause = clause.strip()
    if not clause:
        return []

    parts = re.split(
        r"\band\b(?=\s+(?:add|place|insert|put|remove|delete|erase|eliminate|move|shift|"
        r"relocate|push|make\s+it|apply|restyle|style|a|an|the))",
        clause,
    )
    expanded: list[str] = []
    previous_action: str | None = None

    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if previous_action in {"add", "place", "insert", "put"} and re.match(r"^(a|an|the)\b", part):
            part = f"{previous_action} {part}"
        match = _ACTION_PREFIX_RE.match(part)
        if match:
            previous_action = match.group(1)
        expanded.append(part)

    return expanded


def _looks_like_add_clause(clause: str) -> bool:
    return bool(re.search(r"\b(add|place|insert|put)\b", clause))


def _split_add_continuations(clause: str) -> list[str]:
    parts = re.split(r"\band\b(?=\s+(?:a|an|the|add|place|insert|put))", clause)
    out: list[str] = []
    previous_action: str = "add"
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        match = re.match(r"^(add|place|insert|put)\b", part)
        if match:
            previous_action = match.group(1)
        elif re.match(r"^(a|an|the)\b", part):
            part = f"{previous_action} {part}"
        out.append(part)
    return out


def _parse_add_clause(
    clause: str,
    img_w: int,
    img_h: int,
    catalog_lookup: dict[str, str],
    instructions: list,
    warnings: list,
) -> bool:
    for pat in _ADD_PATS:
        m = re.search(pat, clause)
        if not m:
            continue
        item_raw = m.group(1).strip().rstrip(".")
        item_name, pos_hint = _split_item_and_position(item_raw)
        catalog_match = _match_catalog(item_name, catalog_lookup)
        if catalog_match:
            px, py, resolved_hint = _resolve_pos(pos_hint, clause, img_w, img_h)
            instructions.append({
                "operation": "add",
                "furniture_image_path": catalog_match,
                "target_position": [px, py],
                "catalog_label": _catalog_display_label(catalog_match),
                "position_hint": resolved_hint,
            })
            if resolved_hint == "default":
                warnings.append(
                    f"Add '{item_name}' — no clear location phrase found; defaulting to lower-center."
                )
        else:
            warnings.append(
                f"Add '{item_name}' — no catalog match found. "
                f"Available items: {', '.join(sorted(catalog_lookup)) or 'none'}."
            )
        return True
    return False


def _match_style(candidate: str, style_lookup: dict[str, str]) -> str | None:
    c = candidate.strip().lower()
    if c in style_lookup:
        return style_lookup[c]
    # Partial match: candidate word appears in a key, or vice versa
    for key, canonical in style_lookup.items():
        if c in key or key in c:
            return canonical
    return None


def _match_catalog(item_text: str, catalog_lookup: dict[str, str]) -> str | None:
    text = item_text.strip().lower()
    # Exact match on stem
    if text in catalog_lookup:
        return catalog_lookup[text]
    # Word-intersection match
    item_words = set(text.split()) - {"a", "an", "the"}
    for stem, fname in catalog_lookup.items():
        stem_words = set(stem.split())
        if item_words & stem_words:
            return fname
    return None


def _split_item_and_position(item_raw: str) -> tuple[str, str]:
    """Extract the item name and an optional positional hint from a raw phrase.

    E.g. "lamp on the left" → ("lamp", "left")
         "armchair"         → ("armchair", "")
    """
    text = item_raw.lower()
    # Try two-word hints first (longer match wins)
    for hint in sorted(_POSITION_HINTS, key=len, reverse=True):
        if hint in text:
            item = re.sub(re.escape(hint), "", text)
            item = re.sub(r"\b(on|in|to|at|the|a|an)\b", "", item).strip()
            item = re.sub(r"\s{2,}", " ", item).strip(" -.,")
            return (item or item_raw), hint
    return item_raw, ""


def _resolve_pos(
    hint: str,
    clause: str,
    img_w: int,
    img_h: int,
) -> tuple[int, int, str]:
    """Convert a positional hint (or whole clause scan) to pixel coords."""
    if hint in _POSITION_HINTS:
        xf, yf = _POSITION_HINTS[hint]
        return int(xf * img_w), int(yf * img_h), hint
    # Scan full clause for any positional keyword
    for kw in sorted(_POSITION_HINTS, key=len, reverse=True):
        if kw in clause:
            xf, yf = _POSITION_HINTS[kw]
            return int(xf * img_w), int(yf * img_h), kw
    # Default: lower-centre (typical furniture placement)
    return img_w // 2, int(img_h * 0.65), "default"


def _catalog_display_label(filename: str) -> str:
    stem = Path(filename).stem
    for suffix in ("_img", "_image", "-img", "-image"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ").replace("-", " ").title()
