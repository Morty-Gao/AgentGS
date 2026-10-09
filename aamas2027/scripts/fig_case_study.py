#!/usr/bin/env python3
"""Draw the AgentGS case-study figure and export SVG + PDF to figures/.

Layout units are draw.io pixels on a 1680 x 1530 page. The avatars and the five
tool illustrations are PNG files under scripts/assets/fig_case_study/; all other
shapes are drawn here.

Usage:
    python3 scripts/fig_case_study.py [--name fig_case_study]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _figlib import Box, Canvas, drawio_dash, export

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets" / "fig_case_study"
OUT_DIR = SCRIPT_DIR.parent / "figures"

PAGE_W, PAGE_H = 1680, 1530

TEXT = "#444950"
DATA_TEXT = "#526A7A"
PANEL = "#C1CBD2"
SKILL_A = ("#8DB9EE", "#F4F8FE", "#EAF3FF")   # stroke, planning fill, completion fill
SKILL_B = ("#A7CDB9", "#F3F9F5", "#EAF5ED")
SKILL_C = ("#DC9CB0", "#FEF5F7")
BRIDGE = "#C57991"
FAILURE = "#C38B9C"

# Capability colours shared by every stage.
CAP_FILL = {"w": "#BBDCF1", "u": "#F7DF9D", "m": "#EBCBDF", "v": "#CFBCE8", "z": "#B8DCC5"}

# Stage 3 columns: capability, capability name, tool icon, tool label.
STEPS = [
    ("w", "InputCheck", "tool-input-validation", "Input\nvalidation"),
    ("u", "RasterMean", "tool-raster-composite", "Raster\ncomposite"),
    ("m", "ValidMean", "tool-zonal-statistics", "Zonal\nstatistics"),
    ("v", "ScalarDiff", "tool-scalar-difference", "Scalar\ndifference"),
    ("z", "SignCheck", "tool-sign-comparison", "Sign\ncomparison"),
]
# Icon boxes inside each tool card (x offset from card, y, size); the artwork
# has uneven padding, so each is placed by eye.
TOOL_ICON = [(0.1, 892.0, 155.2), (5.8, 894.8, 148.5), (-18.3, 869.4, 196.4),
             (-10.7, 866.4, 181.5), (-12.5, 878.3, 179.8)]
DATA_TYPES = ["Raster\nlist", "Raster\nfile", "Scalar\nmean", "Scalar\ndifference"]


def dialogue_box(c: Canvas, x, y, w, h, fill, stroke, sw=1.8) -> None:
    """Speech box whose tail points down-left from the body's lower-left corner."""
    b = h - 25
    c.path(f"M70 0 L{w - 30} 0 C{w - 10} 0 {w} 10 {w} 30 L{w} {b - 30} "
           f"C{w} {b - 10} {w - 10} {b} {w - 30} {b} L75 {b} L0 {h} L40 {b - 30} L40 30 "
           f"C40 10 50 0 70 0 Z", x, y, fill=fill, stroke=stroke, sw=sw)


def capability(c: Canvas, key: str, x, y, d=40.0) -> Box:
    b = c.ellipse(x, y, d, d, fill=CAP_FILL[key], stroke=TEXT, sw=1.8)
    c.text(x, y, d, d, key, 25, TEXT, italic=True)
    return b


def magnifier(c: Canvas, x, y) -> None:
    c.ellipse(x, y, 50, 50, fill="#E5F3F8", stroke=TEXT, sw=2.4)
    c.poly([(x + 40, y + 41), (x + 63, y + 67)], stroke=TEXT, sw=7, cap="round")
    c.path("M0 12 C0 4 6 0 13 0", x + 12, y + 9, stroke="#FFFFFF", sw=3, cap="round")


def tick(c: Canvas, x, y, s, color="#539E88", sw=3.0) -> None:
    """Check mark in a 16 x 13.6 frame scaled by s."""
    c.poly([(x, y + 8 * s), (x + 6 * s, y + 13.6 * s), (x + 16 * s, y)], stroke=color, sw=sw,
           join="round", cap="round")


def raster_stack(c: Canvas, x, y, s=1.0) -> None:
    """Three stacked raster layers; (x, y) is the top layer's frame, s the scale."""
    for k, fill in enumerate(["#EAE1F4", "#DDEEE5", "#DCECF7"]):
        oy = y + (2 - k) * 11 * s
        c.poly([(x, oy + 17 * s), (x + 46 * s, oy), (x + 77 * s, oy + 23 * s),
                (x + 31 * s, oy + 41 * s)], closed=True, fill=fill, stroke=TEXT, sw=1.8,
               join="round")
    for gx, gy in [(18, 11), (33, 5)]:
        c.poly([(x + gx * s, y + gy * s), (x + (gx + 31) * s, y + (gy + 22) * s)],
               stroke="#A8BBC9", sw=1, cap="round")


def brain(c: Canvas, x, y) -> None:
    c.path("M43 17 C31 0 15 9 15 21 C1 22 0 39 10 46 C9 60 28 66 39 58 C43 61 47 61 50 58 "
           "C65 67 78 56 76 46 C88 38 84 22 72 21 C70 7 54 3 43 17 Z", x, y,
           fill="#E5D9EF", stroke="#81708F", sw=1.7, join="round")
    fold: dict[str, Any] = dict(stroke="#94829F", sw=1.5, cap="round")
    c.path("M5 0 C0 13 10 26 5 39", x + 38, y + 18, **fold)
    c.path("M2 3 C19 0 18 17 7 17 C0 20 4 29 13 30", x + 15, y + 21, **fold)
    c.path("M16 1 C0 0 0 16 10 16 C18 20 14 27 4 29", x + 53, y + 23, **fold)


def build() -> str:
    c = Canvas(PAGE_W, PAGE_H, ASSET_DIR)
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#FFFFFF")

    dialogue_box(c, 180, 30, 1460, 160, "#F6FAFE", "#91AABD")
    for x, y, w, h in [(40, 215, 615, 422), (695, 215, 945, 422), (40, 683, 1600, 563)]:
        c.rect(x, y, w, h, fill="#FFFFFF", stroke=PANEL, sw=1.5, r=17.5)
    for x in (598, 1221):
        c.fat_arrow(x, 407, 178, 60, sw=2)
    c.rect(525, 618, 593, 80, fill="#FFFFFF", stroke="#697780", sw=1.8, r=14)

    # Skill enclosures: dashed while planning, solid once completed.
    dashed = drawio_dash("7 5", 2.2, False)
    c.rect(261.6, 338.5, 231.8, 68, fill=SKILL_A[1], stroke=SKILL_A[0], sw=2.2, r=34,
           dash=dashed, rot=18.88)
    c.rect(300.9, 521.5, 234.2, 68, fill=SKILL_B[1], stroke=SKILL_B[0], sw=2.2, r=34,
           dash=dashed, rot=9.35)
    c.rect(777.3, 428.5, 300.4, 74, fill=SKILL_C[1], stroke=SKILL_C[0], sw=2.2, r=37,
           dash=dashed, rot=-46.79)
    c.rect(796.4, 325, 245.1, 66, fill=SKILL_A[2], stroke=SKILL_A[0], sw=2.2, r=33, rot=16.21)
    c.rect(816.2, 527.5, 257.6, 66, fill=SKILL_B[2], stroke=SKILL_B[0], sw=2.2, r=33, rot=7.5)
    c.rect(1422, 315, 83, 298, fill="#F2F8FD", stroke="#94BFD9", sw=1.8, r=35)
    c.rect(931, 1113, 206, 47, fill="#FBF6F8", stroke="#CCA8B5", sw=1.5, r=20)
    dialogue_box(c, 180, 1290, 1460, 205, "#F4FAF7", "#8FB4A1")

    # Node positions (drawn after the edges so arrows tuck under them).
    stage1 = {"w": (280, 326), "u": (435, 379), "v": (316, 522), "z": (480, 549)}
    stage2 = {"w": (813, 313), "u": (985, 363), "m": (915, 442), "v": (830, 528), "z": (1020, 553)}
    chain = {k: (1445, 324 + 60 * i) for i, k in enumerate("wumvz")}
    context = [(142, 345), (100, 423), (143, 505)]
    detour = c.box(300, 449, 30, 30, "ellipse")
    cap_x = [259 + 260 * i for i in range(5)]

    nodes = {f"a{k}": c.box(x, y, 40, 40, "ellipse") for k, (x, y) in stage1.items()}
    nodes |= {f"b{k}": c.box(x, y, 40, 40, "ellipse") for k, (x, y) in stage2.items()}
    nodes |= {f"c{k}": c.box(x, y, 36, 36, "ellipse") for k, (x, y) in chain.items()}
    ctx = [c.box(x, y, 28, 28, "ellipse") for x, y in context]
    caps = [c.box(x, 807, 42, 42, "ellipse") for x in cap_x]
    tools = [c.box(x - 59, 899, 160, 184) for x in cap_x]
    gate = c.box(1298, 1115, 44, 44, "ellipse")
    error = c.box(931, 1113, 206, 47)

    def arrow(color=TEXT, sw=2.0, dashed=False) -> dict[str, Any]:
        return dict(color=color, sw=sw, end="classic", size=8, src_gap=3, tgt_gap=4,
                    dash=drawio_dash("6 5", sw, False) if dashed else None)

    c.connect(nodes["aw"], nodes["au"], **arrow())
    c.connect(nodes["av"], nodes["az"], **arrow())
    c.connect(ctx[0], ctx[1], **arrow("#9CA6AE", 1.6))
    c.connect(ctx[1], ctx[2], **arrow("#9CA6AE", 1.6))
    c.connect(nodes["au"], detour, **arrow("#A5AFB6", 1.5))
    c.connect(detour, nodes["av"], **arrow("#A5AFB6", 1.5))
    c.connect(nodes["bw"], nodes["bu"], **arrow())
    c.connect(nodes["bv"], nodes["bz"], **arrow())
    c.connect(nodes["bu"], nodes["bm"], **arrow(BRIDGE, dashed=True))
    c.connect(nodes["bm"], nodes["bv"], **arrow(BRIDGE, dashed=True))
    for a, b in zip("wumv", "umvz"):
        c.connect(nodes[f"c{a}"], nodes[f"c{b}"], **arrow(sw=1.8))
    for cap, tool in zip(caps, tools):
        c.connect(cap, tool, **arrow(sw=1.8))
    for a, b in zip(tools, tools[1:]):
        c.connect(a, b, **arrow())
    head: dict[str, Any] = dict(color=TEXT, end="classic", size=8)
    c.line([(181, 1129), (199, 1067), tools[0].perimeter((199, 1067), 4)], sw=2, **head)
    c.connect(tools[4], gate, **arrow(sw=1.8))
    c.connect(gate, error, **arrow(FAILURE, 1.8, dashed=True))
    c.line([gate.perimeter((1550, 1137), 3), (1550, 1137), (1550, 1006)], sw=1.8, **head)
    c.line([(1416.3, 1149), (1415, 1185)], sw=1.8, **head)
    c.line([(925, 1137), (826, 1137)], color=FAILURE, sw=1.8, end="classic", size=8,
           dash=drawio_dash("6 5", 1.8, False))

    # Nodes.
    for k, (x, y) in stage1.items():
        capability(c, k, x, y)
    for x, y in context:
        c.ellipse(x, y, 28, 28, fill="#F4F6F8", stroke="#A5AFB6", sw=1.5)
    c.ellipse(300, 449, 30, 30, fill="#FFFFFF", stroke="#A5AFB6", sw=1.5)
    for k, (x, y) in stage2.items():
        capability(c, k, x, y)
    c.ellipse(1288, 372, 32, 32, fill="#F1F8F4", stroke="#A4CDBA", sw=1.4)
    for k, (x, y) in chain.items():
        capability(c, k, x, y, 36)
    for (key, *_), x, tool in zip(STEPS, cap_x, tools):
        capability(c, key, x, 807, 42)
        c.rect(tool.x, tool.y, tool.w, tool.h, fill="#FFFFFF", stroke="#9AABB7", sw=1.6, r=10)
    c.ellipse(1298, 1115, 44, 44, fill="#FFFFFF", stroke="#758590", sw=1.8)
    c.ellipse(1203, 1121, 26, 26, fill="#FFFFFF", stroke="#CF8EA2", sw=1.4)
    c.ellipse(1405, 1124, 26, 26, fill="#EAF5EE", stroke="#81B6A0", sw=1.4)
    c.path("M10 32 C-3 15 14 -3 30 9 C38 15 37 26 28 32", 776, 1116, stroke="#667680", sw=2.3)
    c.path("M28 21 L27 33 L39 32", 776, 1116, stroke="#667680", sw=2.3)

    # Glyphs and illustrations.
    magnifier(c, 523, 330)
    magnifier(c, 1142, 370)
    tick(c, 1296, 380, 1.0)
    for (_, _, icon, _), x, (dx, y, s) in zip(STEPS, cap_x, TOOL_ICON):
        c.icon(icon, x - 59 + dx, y, s, s)
    raster_stack(c, 105, 1106)
    c.poly([(1211, 1129), (1221, 1139)], stroke="#C16F88", sw=2.2, cap="round")
    c.poly([(1211, 1139), (1221, 1129)], stroke="#C16F88", sw=2.2, cap="round")
    raster_stack(c, 1510, 924.3, 1.08)
    tick(c, 1412, 1131, 0.75)
    brain(c, 1372, 1174)
    c.icon("avatar-user", 35.3, 27, 139.8, 127.8)
    c.icon("avatar-agent", 44, 1347, 139.3, 134.1)

    # Labels.
    c.text(700, 632, 261, 54, "Agent", 50, TEXT)
    c.text(64, 235, 555, 38, "Stage 1. Skill Planning", 32, TEXT, align="left", bold=True)
    c.text(715, 232, 640, 38, "Stage 2. Missing Capability Completion", 32, TEXT,
           align="left", bold=True)
    c.text(64, 703, 254, 40, "Stage 3. Execution", 32, TEXT, bold=True)
    for x, y, w, h, s, color in [(313, 287, 135, 32, "Skill A", TEXT),
                                 (366, 586, 123, 31, "Skill B", TEXT),
                                 (890, 289, 122, 31, "Skill A", TEXT),
                                 (1055, 565, 124, 31, "Skill B", TEXT),
                                 (972, 477, 129, 32, "Skill C", "#BD6A85"),
                                 (1370, 239, 183, 67, "Capability\nchain", "#517F9F")]:
        c.text(x, y, w, h, s, 25, color)
    c.text(66, 807, 160, 39, "Capability", 34, TEXT, align="left")
    c.text(83, 955, 115, 39, "Tool", 34, TEXT, align="left")
    for (_, cap_name, _, tool_label), x in zip(STEPS, cap_x):
        c.text(x - 91, 760, 224, 39, cap_name, 34, TEXT)
        c.text(x - 64, 1033, 170, 45, tool_label, 23, TEXT)
    for i, label in enumerate(DATA_TYPES):
        c.text(353 + 260 * i, 934, 112, 46, label, 25, DATA_TEXT)
    c.text(76, 1175, 132, 29, "Raster list", 21, DATA_TEXT)
    c.text(1306, 1120, 28, 34, "?", 26, TEXT)
    c.text(931, 1120, 206, 33, "Error Check", 34, TEXT)
    c.text(1256, 1191, 127, 34, "memory", 30, TEXT)
    c.text(255, 55, 1330, 90,
           "Create and return a pixel-wise mean composite of these rasters.\n"
           "Report its valid-pixel mean, difference from the reference, and comparison result.",
           40, TEXT, align="left")
    c.text(255, 1313, 1325, 124,
           "I return the mean composite raster with valid-pixel mean μ.\n"
           "The difference from reference r is d = μ − r.\n"
           "The mean is above / equal to / below r when d > 0 / d = 0 / d < 0.",
           40, TEXT, align="left")
    c.text(34, 165, 145, 30, "User", 35, TEXT)
    c.text(40, 1485, 145, 30, "Agent", 35, TEXT)
    return c.svg()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fig_case_study", help="output file stem in figures/")
    export(build(), OUT_DIR, parser.parse_args().name)


if __name__ == "__main__":
    main()
