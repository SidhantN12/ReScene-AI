"""
generate_catalog.py — Synthetic furniture catalog for ReScene AI.

Generates 14 RGBA PNG files in data/catalog/ using pure cv2/numpy.
Each image has a transparent background so the pipeline's alpha-channel
compositing works without any background-removal step.

Run once before starting the app:
    conda run -n torch-cu121 python generate_catalog.py
or simply:
    python generate_catalog.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

CATALOG_DIR = Path(__file__).parent / "data" / "catalog"

# ---------------------------------------------------------------------------
# Low-level drawing helpers
# All functions operate on an RGBA (H×W×4) uint8 canvas.
# ---------------------------------------------------------------------------

def _canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.uint8)


def _rect(c: np.ndarray, x1: int, y1: int, x2: int, y2: int,
          color: tuple, a: int = 255) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(c.shape[1], x2), min(c.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    c[y1:y2, x1:x2, :3] = color
    c[y1:y2, x1:x2, 3]  = a


def _poly(c: np.ndarray, pts: list, color: tuple, a: int = 255) -> None:
    arr = np.array(pts, np.int32)
    mask = np.zeros(c.shape[:2], np.uint8)
    cv2.fillPoly(mask, [arr], 255)
    c[:, :, :3][mask > 0] = color
    c[:, :, 3][mask > 0]  = a


def _circle(c: np.ndarray, cx: int, cy: int, r: int,
            color: tuple, a: int = 255) -> None:
    mask = np.zeros(c.shape[:2], np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    c[:, :, :3][mask > 0] = color
    c[:, :, 3][mask > 0]  = a


def _ellipse(c: np.ndarray, cx: int, cy: int, rx: int, ry: int,
             color: tuple, angle: float = 0, a: int = 255) -> None:
    mask = np.zeros(c.shape[:2], np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), int(angle), 0, 360, 255, -1)
    c[:, :, :3][mask > 0] = color
    c[:, :, 3][mask > 0]  = a


def _line(c: np.ndarray, x1: int, y1: int, x2: int, y2: int,
          color: tuple, thickness: int = 2, a: int = 255) -> None:
    mask = np.zeros(c.shape[:2], np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
    c[:, :, :3][mask > 0] = color
    c[:, :, 3][mask > 0]  = a


# ---------------------------------------------------------------------------
# Furniture drawing functions
# Convention: colours listed as RGB tuples
# ---------------------------------------------------------------------------

def _draw_sofa() -> np.ndarray:
    W, H = 700, 370
    c = _canvas(W, H)
    FB  = (195, 158, 112)
    FBL = (225, 192, 150)
    FBD = (148, 115, 74)
    FBX = (105, 80, 50)
    LEG = (82, 52, 24)

    for lx in [105, 190, 490, 575]:
        _rect(c, lx, H - 72, lx + 20, H - 4, LEG)

    _rect(c, 65, H - 80, 630, H - 50, FBD)
    _poly(c, [(65, H-50), (630, H-50), (630, H-8), (65, H-8)], FBX)

    _rect(c, 20, 145, 85, H - 50, FB)
    _rect(c, 14, 130, 90, 158, FBL)
    _poly(c, [(85, 145), (90, 158), (90, H-50), (85, H-50)], FBD)

    _rect(c, 610, 145, 675, H - 50, FB)
    _rect(c, 605, 130, 680, 158, FBL)
    _poly(c, [(610, 145), (615, 158), (615, H-50), (610, H-50)], FBD)

    _rect(c, 65, 65, 630, 220, FBD)
    _rect(c, 65, 55, 630, 76, FBL)

    for x1, x2 in [(80, 268), (278, 418), (428, 616)]:
        _rect(c, x1 + 2, 76, x2 - 2, 215, FB)
        _rect(c, x1 + 2, 76, x2 - 2, 92, FBL)
        _poly(c, [(x2-2, 76), (x2-2, 215), (x2+5, 215), (x2+5, 76)], FBD)
    for dx in [272, 422]:
        _rect(c, dx - 2, 77, dx + 2, 214, FBX)

    for x1, x2 in [(80, 268), (278, 418), (428, 616)]:
        _rect(c, x1 + 2, 215, x2 - 2, H - 56, FB)
        _rect(c, x1 + 2, 215, x2 - 2, 228, FBL)
    for dx in [272, 422]:
        _rect(c, dx - 2, 216, dx + 2, H - 57, FBX)

    return c


def _draw_armchair() -> np.ndarray:
    W, H = 420, 390
    c = _canvas(W, H)
    FB  = (185, 145, 98)
    FBL = (218, 180, 132)
    FBD = (138, 105, 62)
    FBX = (98, 70, 40)
    LEG = (80, 50, 22)

    for lx in [110, 170, 240, 298]:
        _rect(c, lx, H - 70, lx + 18, H - 4, LEG)

    _rect(c, 80, H - 80, 335, H - 48, FBD)
    _poly(c, [(80, H-48), (335, H-48), (335, H-8), (80, H-8)], FBX)

    _rect(c, 32, 160, 88, H - 48, FB)
    _rect(c, 26, 145, 94, 168, FBL)
    _poly(c, [(88, 160), (94, 168), (94, H-48), (88, H-48)], FBD)

    _rect(c, 328, 160, 384, H - 48, FB)
    _rect(c, 322, 145, 390, 168, FBL)
    _poly(c, [(328, 160), (334, 168), (334, H-48), (328, H-48)], FBD)

    _rect(c, 80, 70, 335, 230, FBD)
    _rect(c, 80, 58, 335, 80, FBL)
    _rect(c, 86, 80, 328, 222, FB)
    _rect(c, 86, 80, 328, 96, FBL)
    _poly(c, [(328, 80), (334, 80), (334, 222), (328, 222)], FBX)

    _rect(c, 86, 222, 328, H - 56, FB)
    _rect(c, 86, 222, 328, 237, FBL)

    return c


def _draw_coffee_table() -> np.ndarray:
    W, H = 580, 220
    c = _canvas(W, H)
    TOP  = (195, 155, 95)
    TOPL = (218, 180, 120)
    SIDE = (142, 108, 58)
    DARK = (98, 70, 35)
    LEG  = (112, 80, 42)

    for lx in [70, 155, 395, 480]:
        _poly(c, [(lx, 128), (lx+22, 128), (lx+22, H-6), (lx, H-6)], LEG)
        _poly(c, [(lx+22, 128), (lx+28, 134), (lx+28, H-6), (lx+22, H-6)], DARK)

    _poly(c, [(60, 153), (520, 153), (520, 172), (60, 172)], SIDE)
    _poly(c, [(60, 153), (520, 153), (526, 160), (66, 160)], TOPL)

    ty = 40
    _poly(c, [(40, ty), (540, ty), (546, ty+10), (46, ty+10)], TOPL)
    _poly(c, [(40, ty+10), (540, ty+10), (540, 122), (40, 122)], TOP)
    _poly(c, [(540, ty), (546, ty+10), (546, 122), (540, 122)], DARK)
    for gx in range(60, 540, 55):
        _line(c, gx, ty, gx, ty + 8, TOPL, 1)

    return c


def _draw_dining_table() -> np.ndarray:
    W, H = 580, 380
    c = _canvas(W, H)
    TOP  = (185, 148, 88)
    TOPL = (210, 172, 110)
    SIDE = (135, 102, 52)
    DARK = (92, 65, 30)
    LEG  = (102, 72, 35)

    for lx in [80, 470]:
        _rect(c, lx, 162, lx + 22, H - 5, LEG)
        _poly(c, [(lx+22, 162), (lx+28, 168), (lx+28, H-5), (lx+22, H-5)], DARK)
    for lx in [95, 458]:
        _rect(c, lx, 92, lx + 16, 175, DARK)

    _rect(c, 90, 262, 490, 282, SIDE)
    _rect(c, 52, 138, 526, 175, SIDE)
    _poly(c, [(52, 175), (526, 175), (532, 185), (58, 185)], DARK)

    ty = 50
    _poly(c, [(42, ty), (536, ty), (542, ty+12), (48, ty+12)], TOPL)
    _poly(c, [(42, ty+12), (536, ty+12), (536, 148), (42, 148)], TOP)
    _poly(c, [(536, ty), (542, ty+12), (542, 148), (536, 148)], DARK)
    for gx in range(65, 535, 60):
        _line(c, gx, ty, gx, ty + 10, TOPL, 1)

    return c


def _draw_dining_chair() -> np.ndarray:
    W, H = 300, 480
    c = _canvas(W, H)
    FB  = (175, 142, 95)
    FBL = (205, 172, 122)
    FBD = (125, 95, 55)
    LEG = (88, 58, 26)

    for lx in [70, 194]:
        _rect(c, lx, 182, lx + 12, H - 5, LEG)
        _rect(c, lx - 2, 298, lx + 14, H - 5, LEG)

    for lx in [70, 194]:
        _rect(c, lx, 52, lx + 14, 192, LEG)

    for ry in [74, 128, 182]:
        _rect(c, 70, ry, 210, ry + 12, FBL if ry == 74 else FBD)
    for sx in [98, 128, 158, 188]:
        _rect(c, sx, 78, sx + 8, 182, FBD)

    _rect(c, 58, 268, 235, 308, FB)
    _rect(c, 58, 268, 235, 282, FBL)
    _poly(c, [(58, 308), (235, 308), (235, 324), (58, 324)], FBD)
    _rect(c, 70, 302, 225, 322, FBD)

    return c


def _draw_bookshelf() -> np.ndarray:
    W, H = 380, 600
    c = _canvas(W, H)
    WOOD = (152, 115, 62)
    WDLH = (178, 140, 86)
    WDDK = (105, 75, 36)
    WDXD = (72, 50, 22)

    _rect(c, 25, 25, 62, H - 8, WOOD)
    _poly(c, [(25, 25), (29, 20), (29, H-8), (25, H-8)], WDLH)
    _rect(c, 312, 25, 348, H - 8, WOOD)
    _poly(c, [(312, 25), (348, 25), (348, H-8), (312, H-8)], WDDK)

    _rect(c, 25, 25, 348, 56, WOOD)
    _poly(c, [(25, 25), (348, 25), (352, 20), (29, 20)], WDLH)
    _rect(c, 25, H - 40, 348, H - 8, WOOD)
    _rect(c, 62, 56, 312, H - 40, (132, 100, 52))

    for sy in [153, 253, 352, 442, 528]:
        _rect(c, 62, sy, 312, sy + 17, WOOD)
        _poly(c, [(62, sy), (312, sy), (312, sy-3), (62, sy-3)], WDLH)

    book_palette = [
        [(178, 78, 48), (208, 128, 58), (78, 118, 158), (158, 178, 78), (198, 78, 98)],
        [(78, 98, 178), (198, 158, 58), (158, 58, 78), (98, 178, 118), (78, 78, 158)],
        [(158, 98, 58), (78, 158, 98), (198, 118, 78), (98, 78, 198), (178, 78, 58)],
        [(98, 158, 78), (198, 78, 58), (78, 98, 178), (158, 78, 118), (78, 158, 158)],
    ]
    shelf_tops = [56, 153, 253, 352]
    shelf_bots = [153, 253, 352, 442]
    for i, (sy1, sy2) in enumerate(zip(shelf_tops, shelf_bots)):
        bx = 68
        for col in book_palette[i % len(book_palette)]:
            bh = int((sy2 - sy1 - 17) * 0.75)
            bw = 28
            _rect(c, bx, sy2 - 17 - bh, bx + bw - 2, sy2 - 17, col)
            _rect(c, bx, sy2 - 17 - bh, bx + 3, sy2 - 17,
                  tuple(min(v + 28, 255) for v in col))
            bx += bw
            if bx + bw > 305:
                break

    return c


def _draw_floor_lamp() -> np.ndarray:
    W, H = 180, 580
    c = _canvas(W, H)
    SHADE  = (222, 202, 152)
    SHADED = (172, 152, 100)
    POLE   = (122, 112, 102)
    POLEL  = (162, 152, 142)
    BASE   = (98, 90, 82)
    BASED  = (68, 60, 52)

    _ellipse(c, W//2, H - 30, 60, 18, BASED)
    _ellipse(c, W//2, H - 35, 54, 14, BASE)
    _ellipse(c, W//2, H - 38, 48, 10, (138, 128, 118))

    _rect(c, W//2 - 6, 152, W//2 + 6, H - 50, POLE)
    _rect(c, W//2 - 6, 152, W//2 - 2, H - 50, POLEL)

    _poly(c, [(W//2-14, 143), (W//2+14, 143),
               (W//2+64, 238), (W//2-64, 238)], SHADED)
    _poly(c, [(W//2-11, 146), (W//2+11, 146),
               (W//2+60, 235), (W//2-60, 235)], SHADE)

    _ellipse(c, W//2, 143, 16, 6, BASE)
    _ellipse(c, W//2, 238, 64, 20, SHADED)
    _ellipse(c, W//2, 238, 61, 17, SHADE)
    _ellipse(c, W//2, 220, 54, 14, (245, 222, 168), a=155)

    return c


def _draw_bed() -> np.ndarray:
    W, H = 660, 520
    c = _canvas(W, H)
    FRAME  = (162, 122, 70)
    FRAMEL = (192, 152, 92)
    FRAMED = (112, 80, 38)
    MATT   = (225, 218, 210)
    MATTD  = (188, 180, 170)
    PILLOW = (242, 237, 230)
    PILLSH = (208, 200, 190)
    SHEET  = (232, 228, 218)

    _rect(c, 42, 30, 618, 205, FRAME)
    _poly(c, [(42, 30), (618, 30), (618, 24), (42, 24)], FRAMEL)
    _poly(c, [(618, 30), (625, 36), (625, 205), (618, 205)], FRAMED)
    _rect(c, 60, 48, 600, 195, (142, 105, 55))
    _rect(c, 65, 53, 595, 190, FRAMED)
    _rect(c, 60, 115, 600, 128, FRAMEL)

    _rect(c, 42, H - 85, 618, H - 40, FRAME)
    _poly(c, [(618, H-85), (625, H-80), (625, H-40), (618, H-40)], FRAMED)
    _rect(c, 42, 200, 68, H - 82, FRAMED)
    _rect(c, 592, 200, 618, H - 82, FRAMED)
    _poly(c, [(618, 200), (625, 205), (625, H-80), (618, H-82)], FRAMED)

    for lx in [42, 592]:
        _rect(c, lx, H - 82, lx + 26, H - 4, FRAMED)

    _rect(c, 58, 195, 600, H - 90, MATT)
    _poly(c, [(58, 195), (600, 195), (600, 205), (58, 205)], (240, 235, 228))

    _rect(c, 58, 268, 600, H - 90, SHEET)
    _poly(c, [(58, 268), (600, 268), (600, 278), (58, 278)], (246, 242, 232))
    for wx in range(100, 580, 80):
        _line(c, wx, 285, wx + 20, H - 92, MATTD, 1)

    for px, pw in [(80, 220), (340, 220)]:
        _rect(c, px, 205, px + pw, 268, PILLOW)
        _poly(c, [(px, 268), (px+pw, 268), (px+pw, 276), (px, 276)], PILLSH)
        _rect(c, px + 5, 210, px + pw - 5, 263, (246, 242, 234))
        _rect(c, px + 12, 212, px + pw - 12, 215, (228, 224, 216))
        _rect(c, px + 12, 262, px + pw - 12, 265, (228, 224, 216))

    return c


def _draw_nightstand() -> np.ndarray:
    W, H = 310, 340
    c = _canvas(W, H)
    WOOD  = (160, 120, 68)
    WOODL = (186, 146, 90)
    WOODD = (112, 82, 42)
    WOODX = (78, 54, 24)
    KNOB  = (188, 176, 160)

    _poly(c, [(28, 38), (282, 38), (290, 28), (36, 28)], WOODL)
    _poly(c, [(28, 38), (282, 38), (282, 62), (28, 62)], WOOD)
    _rect(c, 28, 62, 282, H - 38, WOOD)
    _poly(c, [(282, 62), (290, 56), (290, H-34), (282, H-38)], WOODD)

    _rect(c, 38, 72, 272, 165, WOODD)
    _rect(c, 42, 76, 268, 161, WOODL)
    _rect(c, 42, 76, 268, 90, WOOD)
    _ellipse(c, W // 2, 118, 14, 10, KNOB)
    _ellipse(c, W // 2, 115, 10, 7, (210, 196, 180))

    _rect(c, 38, 172, 272, H - 48, WOODD)
    _rect(c, 42, 176, 268, H - 52, WOODL)
    _rect(c, 42, 176, 268, 190, WOOD)
    _ellipse(c, W // 2, 230, 14, 10, KNOB)
    _ellipse(c, W // 2, 227, 10, 7, (210, 196, 180))

    for lx in [35, 263]:
        _rect(c, lx, H - 38, lx + 18, H - 4, WOODX)
    _rect(c, 28, H - 45, 282, H - 36, WOOD)

    return c


def _draw_desk() -> np.ndarray:
    W, H = 700, 440
    c = _canvas(W, H)
    TOP   = (176, 138, 80)
    TOPL  = (202, 162, 102)
    SIDE  = (130, 98, 50)
    DARK  = (90, 62, 28)
    LEG   = (106, 76, 36)
    DRAW  = (152, 116, 62)
    DRAWL = (176, 138, 83)
    METAL = (152, 145, 138)

    _rect(c, 32, 145, 78, H - 5, LEG)
    _poly(c, [(78, 145), (86, 150), (86, H-5), (78, H-5)], DARK)
    _rect(c, 602, 145, 648, H - 5, LEG)
    _poly(c, [(648, 145), (656, 150), (656, H-5), (648, H-5)], DARK)
    _rect(c, 78, H - 60, 602, H - 44, SIDE)

    _rect(c, 32, 148, 220, H - 5, DRAW)
    _poly(c, [(220, 148), (228, 152), (228, H-5), (220, H-5)], DARK)
    for dy in [180, 260, 340]:
        _rect(c, 40, dy - 1, 215, dy + 2, DARK)
        _rect(c, 40, dy + 2, 215, dy + 5, DRAWL)
    for dy in [215, 295, 370]:
        _ellipse(c, 128, dy, 12, 8, METAL)

    ty = 40
    _poly(c, [(20, ty), (668, ty), (676, ty+10), (28, ty+10)], TOPL)
    _poly(c, [(20, ty+10), (668, ty+10), (668, 158), (20, 158)], TOP)
    _poly(c, [(668, ty), (676, ty+10), (676, 158), (668, 158)], DARK)
    for gx in range(45, 660, 60):
        _line(c, gx, ty, gx, ty + 8, TOPL, 1)

    return c


def _draw_office_chair() -> np.ndarray:
    W, H = 420, 550
    c = _canvas(W, H)
    FAB   = (128, 128, 145)
    FABL  = (160, 160, 178)
    FABD  = (92, 92, 108)
    METAL = (122, 115, 108)
    METD  = (80, 74, 68)
    WHEEL = (62, 60, 55)

    cx, cy = W // 2, H - 45
    for angle_deg in range(0, 360, 72):
        rad = np.radians(angle_deg)
        ex = int(cx + 88 * np.cos(rad))
        ey = int(cy + 28 * np.sin(rad))
        _poly(c, [(cx-5, cy-4), (cx+5, cy+4), (ex+5, ey+5), (ex-5, ey-5)], METD)
        _ellipse(c, ex, ey, 11, 7, WHEEL)
    _circle(c, cx, cy, 17, METAL)

    _rect(c, W//2 - 12, H - 140, W//2 + 12, H - 55, METD)
    _rect(c, W//2 - 12, H - 140, W//2 - 6, H - 55, METAL)
    _rect(c, W//2 - 18, H - 168, W//2 + 18, H - 135, METD)

    seat_y = H - 180
    _ellipse(c, W//2, seat_y, 108, 29, FABD)
    _rect(c, 100, seat_y - 30, 318, seat_y + 6, FAB)
    _ellipse(c, W//2, seat_y - 28, 108, 21, FABL)

    _rect(c, 115, 85, 300, seat_y - 18, FAB)
    _rect(c, 115, 85, 300, 108, FABL)
    _poly(c, [(300, 85), (308, 90), (308, seat_y-18), (300, seat_y-18)], FABD)
    for by in range(210, 295, 20):
        _rect(c, 115, by, 300, by + 7, FABD)

    _rect(c, 75, 278, 115, 358, METD)
    _rect(c, 88, 258, 115, 283, FAB)
    _rect(c, 300, 278, 340, 358, METD)
    _rect(c, 300, 258, 318, 283, FAB)

    return c


def _draw_plant() -> np.ndarray:
    W, H = 320, 540
    c = _canvas(W, H)
    GR1  = (52, 132, 58)
    GR2  = (38, 102, 42)
    GR3  = (82, 162, 70)
    STEM = (62, 98, 38)
    TER  = (170, 86, 50)
    TERL = (198, 112, 70)
    TERD = (125, 60, 30)
    SOIL = (78, 60, 40)

    _poly(c, [(70, 378), (250, 378), (270, H-14), (50, H-14)], TER)
    _poly(c, [(62, 378), (258, 378), (262, 398), (58, 398)], TERL)
    _poly(c, [(250, 378), (270, H-14), (278, H-14), (258, 378)], TERD)
    _poly(c, [(58, 393), (262, 393), (262, 408), (58, 408)], TER)
    _ellipse(c, W//2, 380, 86, 21, SOIL)

    stems = [
        (160, 373, 155, 278, 5),
        (160, 298, 118, 192, 4),
        (160, 298, 198, 188, 4),
        (153, 278, 98, 152, 3),
        (153, 258, 172, 138, 3),
        (160, 298, 182, 218, 4),
        (162, 218, 208, 132, 3),
        (143, 248, 108, 172, 3),
    ]
    for x1, y1, x2, y2, th in stems:
        _line(c, x1, y1, x2, y2, STEM, th)

    leaves = [
        (98, 143, 44, 21, -20, GR1), (98, 143, 29, 17, 30, GR2),
        (172, 128, 46, 19, 15, GR3), (172, 128, 31, 15, -25, GR1),
        (208, 122, 40, 17, 10, GR2), (108, 162, 37, 15, -30, GR1),
        (182, 212, 34, 14, 20, GR3), (138, 192, 38, 16, -15, GR2),
        (152, 118, 36, 15, -5, GR1), (112, 178, 34, 14, 40, GR3),
        (198, 172, 38, 16, 25, GR1), (132, 158, 40, 17, 10, GR2),
        (162, 142, 35, 14, -10, GR3), (188, 138, 37, 15, 30, GR1),
    ]
    for lx, ly, rx, ry, ang, col in leaves:
        _ellipse(c, lx, ly, rx, ry, col, ang)
    for lx, ly, col in [(103, 146, GR3), (175, 131, GR1), (110, 165, GR3),
                         (190, 125, GR3), (142, 190, GR1)]:
        _ellipse(c, lx, ly, 17, 8, col)

    return c


def _draw_tv_stand() -> np.ndarray:
    W, H = 700, 310
    c = _canvas(W, H)
    WOOD  = (165, 132, 82)
    WOODL = (192, 158, 105)
    WOODD = (118, 88, 45)
    WOODX = (80, 56, 26)
    METAL = (145, 138, 130)

    for lx in [35, 620]:
        _rect(c, lx, H - 55, lx + 20, H - 5, WOODX)

    _rect(c, 25, 65, 670, H - 50, WOOD)
    _poly(c, [(670, 65), (680, 72), (680, H-50), (670, H-50)], WOODD)
    _poly(c, [(25, 65), (670, 65), (680, 58), (35, 58)], WOODL)

    _rect(c, 35, 78, 215, H - 58, WOODD)
    _rect(c, 40, 82, 210, H - 62, (162, 132, 80))
    _rect(c, 45, 87, 205, H - 67, (183, 193, 208))
    for x1, y1, x2, y2 in [(45, 87, 205, 87), (45, H-67, 205, H-67),
                             (45, 87, 45, H-67), (205, 87, 205, H-67)]:
        _line(c, x1, y1, x2, y2, (152, 162, 178), 2)

    _rect(c, 455, 78, 635, H - 58, WOODD)
    _rect(c, 460, 82, 630, H - 62, (162, 132, 80))
    _rect(c, 465, 87, 625, H - 67, (183, 193, 208))
    for x1, y1, x2, y2 in [(465, 87, 625, 87), (465, H-67, 625, H-67),
                             (465, 87, 465, H-67), (625, 87, 625, H-67)]:
        _line(c, x1, y1, x2, y2, (152, 162, 178), 2)

    _rect(c, 215, 78, 455, H - 58, (102, 75, 40))
    _rect(c, 218, 138, 452, 143, (118, 88, 48))

    for dx in [212, 452]:
        _rect(c, dx - 3, 68, dx + 3, H - 52, WOODL)

    for hx in [125, 545]:
        _ellipse(c, hx, (65 + H - 50) // 2, 14, 6, METAL)

    return c


def _draw_wardrobe() -> np.ndarray:
    W, H = 480, 660
    c = _canvas(W, H)
    WOOD  = (155, 120, 68)
    WOODL = (182, 145, 90)
    WOODD = (105, 75, 38)
    WOODX = (70, 48, 22)
    METAL = (162, 152, 142)
    METD  = (108, 100, 90)

    _rect(c, 22, 22, 458, H - 8, WOOD)
    _poly(c, [(458, 22), (468, 28), (468, H-8), (458, H-8)], WOODD)
    _poly(c, [(22, 22), (458, 22), (468, 16), (32, 16)], WOODL)
    _rect(c, 14, 14, 470, 30, WOODL)
    _poly(c, [(14, 30), (470, 30), (470, 38), (14, 38)], WOODD)
    _rect(c, 14, H - 22, 470, H - 8, WOODD)
    for fx in [35, 425]:
        _rect(c, fx, H - 8, fx + 28, H, WOODX)

    _rect(c, 28, 36, 244, H - 22, WOODD)
    _rect(c, 34, 42, 238, H - 28, (142, 108, 58))
    _rect(c, 40, 48, 232, 340, WOODX)
    _rect(c, 44, 52, 228, 336, (125, 93, 50))
    _rect(c, 40, 348, 232, H - 32, WOODX)
    _rect(c, 44, 352, 228, H - 36, (125, 93, 50))
    _rect(c, 220, 328, 232, 358, METAL)
    _ellipse(c, 226, 343, 6, 15, METD)

    _rect(c, 250, 36, 466, H - 22, WOODD)
    _rect(c, 256, 42, 460, H - 28, (142, 108, 58))
    _rect(c, 262, 48, 454, 340, WOODX)
    _rect(c, 266, 52, 450, 336, (125, 93, 50))
    _rect(c, 262, 348, 454, H - 32, WOODX)
    _rect(c, 266, 352, 450, H - 36, (125, 93, 50))
    _rect(c, 252, 328, 264, 358, METAL)
    _ellipse(c, 258, 343, 6, 15, METD)

    _rect(c, 243, 28, 253, H - 18, WOODL)

    return c


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ITEMS: list[tuple[str, Any, str]] = [
    ("sofa.png",          _draw_sofa,         "Sofa"),
    ("armchair.png",      _draw_armchair,     "Armchair"),
    ("coffee_table.png",  _draw_coffee_table, "Coffee Table"),
    ("dining_table.png",  _draw_dining_table, "Dining Table"),
    ("dining_chair.png",  _draw_dining_chair, "Dining Chair"),
    ("bookshelf.png",     _draw_bookshelf,    "Bookshelf"),
    ("floor_lamp.png",    _draw_floor_lamp,   "Floor Lamp"),
    ("bed.png",           _draw_bed,          "Bed"),
    ("nightstand.png",    _draw_nightstand,   "Nightstand"),
    ("desk.png",          _draw_desk,         "Desk"),
    ("office_chair.png",  _draw_office_chair, "Office Chair"),
    ("plant.png",         _draw_plant,        "Potted Plant"),
    ("tv_stand.png",      _draw_tv_stand,     "TV Stand"),
    ("wardrobe.png",      _draw_wardrobe,     "Wardrobe"),
]

from typing import Any  # noqa: E402 (needed for ITEMS type hint above)


def main() -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(ITEMS)} furniture PNGs to {CATALOG_DIR}/\n")
    for filename, fn, label in ITEMS:
        img = fn()
        path = CATALOG_DIR / filename
        bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(path), bgra)
        print(f"  {label:22s} → {filename:25s}  ({img.shape[1]}×{img.shape[0]})")
    print(f"\nDone — {len(ITEMS)} items generated.")


if __name__ == "__main__":
    main()
