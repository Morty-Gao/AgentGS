#!/usr/bin/env python3
"""Draw the AgentGS main figure (panels a-c) and export SVG + PDF to figures/.

Layout units are pixels on a 3000 x 2062.5 (16:11) page (scaled to \\textwidth in the
paper, so 40 px text prints at roughly 6.7 pt). Each panel is drawn in its own
local frame. Only the query-user illustration and the capability icons are
files under scripts/assets/fig_main/; every other shape is drawn here.

Colour semantics are shared by all panels: Skill A blue, Skill B green, the
bridging Skill C orange, failure/re-planning red, Skill write-back purple.

Usage:
    python3 scripts/fig_main.py [--name fig_main]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from _figlib import SERIF, Canvas, drawio_dash, export
from _theme import (
    BLUE,
    BLUE_SOFT,
    FAINT,
    GREEN,
    GREEN_SOFT,
    HAND,
    INK,
    LINE,
    MUTED,
    NODE,
    ORANGE,
    ORANGE_SOFT,
    PAPER,
    PURPLE,
    RED,
    SW,
    capsule,
    document,
    flow,
    legend,
    lightbulb,
    panel_frame,
    pill,
    skill_graph,
    tool_graph,
    toolbox,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets" / "fig_main"
OUT_DIR = SCRIPT_DIR.parent / "figures"

PAGE_W, PAGE_H = 3000, 2062.5  # 16:11

def port_symbol(c: Canvas, cx, cy, length=60, sw=2.5) -> None:
    """Legend glyph for a data transfer: output port arc plus arrow."""
    x0, x1 = cx - length / 2, cx + length / 2
    c.port_arc(c.box(x0 - 12, cy, 0, 0), c.box(x1, cy, 0, 0), r=12, span=80, color=MUTED, sw=sw)
    c.line([(x0 + 4, cy), (x1, cy)], color=MUTED, sw=sw, end="classic", size=8)


def update_glyph(c: Canvas, x, y) -> None:
    """Skill-library update: the lens inspects the completed Skill C, and the
    arrow inserts it as a new entry into the empty (dashed) slot."""
    c.text(x, y, 110, 36, "update", 32, MUTED, align="left", font=HAND)
    c.text(x + 140, y, 110, 36, "skills", 32, MUTED, align="left", font=HAND)
    lx, ly, r = x + 42, y + 78, 28
    c.poly([(lx - 20, ly + 20), (lx - 40, ly + 42)], stroke=MUTED, sw=6, cap="round")
    c.ellipse(lx - r, ly - r, 2 * r, 2 * r, fill="#FFFFFF", stroke=MUTED, sw=4.5)
    c.rect(lx - 17, ly - 8, 34, 16, fill=ORANGE_SOFT, stroke=ORANGE, sw=2.5, r=8, dash="4 3")
    c.path("M0 24 C20 2 44 0 66 6", lx + 32, ly - 40, stroke=MUTED, sw=3.5, cap="round")
    c.line([(lx + 92, ly - 35), (lx + 101, ly - 31)], color=MUTED, sw=3.5, end="classic", size=7)
    bx, by = x + 150, y + 42
    c.rect(bx, by, 80, 104, fill="#FFFFFF", stroke=MUTED, sw=3, r=6)
    c.rect(bx + 9, by + 9, 62, 24, stroke=MUTED, sw=2.2, r=4, dash="5 4")
    for k in (1, 2):
        c.rect(bx + 9, by + 9 + 31 * k, 62, 24, fill="#FFFFFF", stroke=MUTED, sw=2.2, r=4)
    c.rect(bx - 22, by + 4, 50, 22, fill=ORANGE_SOFT, stroke=ORANGE, sw=2.5, r=4)


def label_along(c: Canvas, p0, p1, text, offset, color, size=38, below=False, t=0.5) -> None:
    """Text rotated to the segment p0 -> p1, centred on it and pushed `offset`
    to its upper (or lower) side (panel-local coordinates)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d = math.hypot(dx, dy)
    nx, ny = dy / d, -dx / d
    if (ny > 0) != below:
        nx, ny = -nx, -ny
    cx, cy = p0[0] + dx * t + nx * offset, p0[1] + dy * t + ny * offset
    c.text(cx - 100, cy - 25, 200, 50, text, size, color, rot=math.degrees(math.atan2(dy, dx)))


# ---------------------------------------------------------------------------
# Panel a: retrieval, planning, completion, grounding, execution, judging.
# ---------------------------------------------------------------------------
A_FRAME = (24, 24, 2952, 820)


def panel_a(c: Canvas) -> None:
    panel_frame(c, *A_FRAME, "a")
    c.origin(A_FRAME[0], A_FRAME[1])
    label = 44
    y_wb = 40      # write-back loop
    y_no = 112     # "no applicable Skill" route
    y_top = 162    # SkillSet / GeoSkillKG icons
    ym = 418       # main flow axis
    y_name = 284   # names above GeoToolKG / Result / Answer
    card = (244, 580)
    yc = 668       # completed chain
    y_loop = 778   # re-plan loop

    # Inputs and Skill retrieval.
    toolbox(c, 160, y_top)
    c.text(40, y_top + 82, 240, 54, "SkillSet", label, INK, font=HAND)
    skill_graph(c, 480, y_top)
    c.text(360, y_top + 82, 240, 54, "GeoSkillKG", label, INK, font=HAND)
    c.line([(255, y_top), (355, y_top)], **flow())

    c.icon("a-query-user", 103, ym - 62, 95, 124.72)
    c.speech_bubble(196, ym - 88, 66, 44, r=16, tail=(14, 26, 6, 10), color=INK, sw=3,
                    fill="#FFFFFF")
    for dx in (214, 229, 244):
        c.ellipse(dx - 4, ym - 70, 8, 8, fill=INK)
    c.text(40, ym + 76, 220, 54, "Query", label, INK, font=HAND)
    c.line([(240, ym), (365, ym)], **flow())

    pill(c, 480, ym, 220, 84, "Retrieve")
    c.line([(455, ym - 42), (455, y_top + 158)], **flow())
    c.line([(505, y_top + 158), (505, ym - 42)], **flow())

    pill(c, 800, ym, 200, 84, "Planner")
    c.line([(585, ym + 20), (702, ym + 20)], **flow())
    c.text(600, ym + 30, 90, 46, "Yes", 38, INK, bg="#FFFFFF")

    # No applicable Skill: plan directly over GeoToolKG.
    c.line([(585, ym - 20), (630, ym - 20), (630, y_no), (1980, y_no), (1980, y_name - 32)],
           **flow(MUTED, radius=10, dash=drawio_dash("4 3", SW, False)))
    c.text(645, (y_no + ym) / 2 - 23, 80, 46, "No", 38, MUTED, align="left")

    # Missing Capability Completion: Skill C bridges two selected fragments.
    c.line([(900, ym), (972, ym)], **flow())
    top, bottom = card
    c.rect(975, top, 665, bottom - top, fill=PAPER, stroke=FAINT, sw=2.5, r=24)
    c.text(975, top + 8, 665, 46, "Missing Capability Completion", 38, MUTED)
    ya, ycc, yb = top + 110, top + 184, top + 258
    selected = {"a1": (1130, ya - 14, BLUE), "a2": (1245, ya, BLUE), "a3": (1360, ya + 14, BLUE),
                "c": (1245, ycc, ORANGE),
                "b1": (1130, yb - 14, GREEN), "b2": (1245, yb, GREEN), "b3": (1360, yb + 14, GREEN)}
    capsule(c, 1245, ya, 312, 64, 7, BLUE, BLUE_SOFT)
    capsule(c, 1245, yb, 312, 64, 7, GREEN, GREEN_SOFT)
    nodes = {k: c.box(x - 20, y - 20, 40, 40, "ellipse") for k, (x, y, _) in selected.items()}
    for s, t in [("a1", "a2"), ("a2", "a3"), ("b1", "b2"), ("b2", "b3")]:
        c.connect(nodes[s], nodes[t], color=INK, sw=3, end="block", size=10, src_gap=2, tgt_gap=2)
    for s, t in [("a3", "c"), ("c", "b1")]:
        c.connect(nodes[s], nodes[t], color=ORANGE, sw=3.2, end="block", size=10, src_gap=2,
                  tgt_gap=2, dash=drawio_dash("3 2", 3.2, False))
    for x, y, fill in selected.values():
        c.ellipse(x - 20, y - 20, 40, 40, fill=fill, stroke=INK, sw=2.5)
    c.text(1430, ya - 22, 190, 44, "Skill A", 38, BLUE, align="left")
    c.text(1274, ycc - 4, 190, 44, "Skill C", 36, ORANGE, align="left")
    c.text(1430, yb - 22, 190, 44, "Skill B", 38, GREEN, align="left")

    # Completed capability chain C1..C7; C4 is supplied by Skill C.
    c.line([(1300, bottom), (1300, yc - 44)], **flow())
    xs = [1300 + (i - 3) * 112 for i in range(7)]
    for lo, hi, stroke, fill in [(0, 2, BLUE, BLUE_SOFT), (4, 6, GREEN, GREEN_SOFT),
                                 (2, 4, ORANGE, ORANGE_SOFT)]:
        x0, x1 = xs[lo] - 46, xs[hi] + 46
        c.rect(x0, yc - 42, x1 - x0, 84, fill=fill, stroke=stroke, sw=3, r=42)
    chain = [c.box(x - 30, yc - 30, 60, 60, "ellipse") for x in xs]
    for i in range(6):
        bridged = i in (2, 3)
        c.connect(chain[i], chain[i + 1], color=ORANGE if bridged else INK, sw=3, end="block",
                  size=9, src_gap=1, tgt_gap=1)
    for i, x in enumerate(xs):
        if i == 3:
            c.ellipse(x - 30, yc - 30, 60, 60, fill=ORANGE_SOFT, stroke=ORANGE, sw=4)
        else:
            c.ellipse(x - 30, yc - 30, 60, 60, fill="#FFFFFF", stroke=INK, sw=2.5)
        c.text(x - 30, yc - 30, 60, 60, f"C{i + 1}", 32, INK, font=HAND)
    for lo, hi, name, color in [(0, 2, "Skill A", BLUE), (2, 4, "Skill C", ORANGE),
                                (4, 6, "Skill B", GREEN)]:
        c.text((xs[lo] + xs[hi]) / 2 - 120, yc + 48, 240, 44, name, 38, color)

    # Tool grounding and execution.
    c.line([(1690, yc), (1980, yc), (1980, ym + 92)], **flow(radius=10))
    c.text(2005, (yc + ym) / 2 + 6, 260, 96, "Tool\nGrounding", 40, INK, align="left")
    tool_graph(c, 1980, ym, 0.85)
    c.text(1840, y_name - 27, 280, 54, "GeoToolKG", label, INK, font=HAND)
    c.line([(2070, ym), (2163, ym)], **flow())
    c.text(2070, ym - 56, 100, 46, "LLM", 40, INK)
    c.text(2140, ym - 225, 280, 54, "Tool Chain", label, INK, font=HAND)
    for i, y in enumerate([ym - 120, ym, ym + 120]):
        pill(c, 2280, y, 210, 72, f"Tool {i + 1}", 38)
    c.line([(2280, ym - 84), (2280, ym - 38)], **flow(sw=3))
    c.line([(2280, ym + 36), (2280, ym + 82)], **flow(sw=3))
    c.line([(2385, ym), (2556, ym)], **flow())
    c.text(2392, ym + 14, 160, 46, "Executor", 40, INK)

    # Result and judgement.
    c.text(2520, y_name - 27, 200, 54, "Result", label, INK, font=HAND)
    c.ellipse(2562, ym - 58, 116, 116, fill="#FFFFFF", stroke=INK, sw=3.5)
    document(c, 2596, ym - 30, 48, 60, 14, BLUE_SOFT, BLUE)
    c.text(2596, ym - 20, 48, 50, "?", 34, BLUE, bold=True, font=HAND)
    yj = ym + 104  # fork point
    c.line([(2620, ym + 58), (2620, yj)], color=INK, sw=SW)
    c.line([(2620, yj), (2616, yj + 38), (2570, yj + 62)], curved=True, **flow())
    c.line([(2620, yj), (2624, yj + 38), (2670, yj + 62)], curved=True, **flow())
    c.text(2652, yj - 30, 140, 46, "Judge", 40, INK, align="left")
    yd = yj + 94  # judgement badges
    c.ellipse(2520, yd - 30, 60, 60, fill="#FFFFFF", stroke=RED, sw=4)
    c.poly([(2538, yd - 12), (2562, yd + 12)], stroke=RED, sw=5.5, cap="round")
    c.poly([(2562, yd - 12), (2538, yd + 12)], stroke=RED, sw=5.5, cap="round")
    c.ellipse(2658, yd - 32, 64, 64, fill=GREEN)
    c.poly([(2674, yd + 1), (2685, yd + 13), (2707, yd - 13)], stroke="#FFFFFF", sw=6.5,
           join="round", cap="round")

    # Failure: re-plan; success: answer and write the composition back.
    red = flow(RED, dash=drawio_dash("8 4 2 4", SW, False))
    c.line([(2550, yd + 32), (2550, y_loop), (800, y_loop), (800, ym + 44)], radius=10, **red)
    c.text(2170, y_loop - 24, 200, 48, "Re-plan", 38, RED, bg="#FFFFFF")
    c.line([(2724, yd), (2846, yd), (2846, ym + 62)], **flow(radius=10))
    lightbulb(c, 2817, ym - 45)
    c.text(2746, y_name - 27, 200, 54, "Answer", label, INK, font=HAND)
    c.line([(2846, y_name - 32), (2846, y_wb), (160, y_wb), (160, y_top - 70)],
           **flow(PURPLE, radius=12, dash=drawio_dash("4 2", SW, False)))
    c.text(1200, y_wb - 26, 560, 52, "New Skill (after completion)", 42, PURPLE, bg="#FFFFFF")


# ---------------------------------------------------------------------------
# Panel b: GeoSkillKG over GeoToolKG and one Skill's tool implementations.
# Graph and example coordinates come from the original drawing and are
# compressed vertically by yg / ye; shapes keep their size.
# ---------------------------------------------------------------------------
B_FRAME = (24, 868, 1191.68, 1170.5)
CAP_D = (30.2, 27.25)
TOOL_D = (34.23, 30.88)
LOWER_SHIFT = -38.89  # the tool layer tucks under the skill layer
PLANE_K = 1.0


def yg(y: float) -> float:
    return 175 + (y - 180.76) * PLANE_K


def ye(y: float) -> float:
    return 880 + (y - 916) * 0.9


def panel_b(c: Canvas) -> None:
    panel_frame(c, *B_FRAME, "b")
    c.origin(48, B_FRAME[1] + 52)
    ds = LOWER_SHIFT

    c.speech_bubble(52, -30, 1078, 156, r=24, tail=(28, 70, 17, 28), color=LINE, sw=2.5,
                    fill=PAPER)
    for y, s in [(-18, "Reproject this elevation map to a local metric CRS,"),
                 (28, "crop it to the study-area extent, and calculate"),
                 (74, "slope in degrees.")]:
        c.text(84, y, 1030, 46, s, 42, INK, align="left")

    # Two stacked layers; the semi-opaque skill layer hides the tool layer's top edge.
    for (x, y, w, h), opacity in [((89.47, yg(514.25 + ds), 736.53, 320.92 * PLANE_K), None),
                                  ((163.57, yg(180.76), 736.53, 320.92 * PLANE_K), 0.88)]:
        d = c.rounded_polygon(c.parallelogram(x, y, w, h), 6)
        c.body.append(f'<path d="{d}" {c.paint(PAPER, LINE, 2.5, fill_opacity=opacity)}/>')
    c.text(-6, 280, 240, 44, "GeoSkillKG", 38, MUTED, align="left", font=HAND)
    c.text(-6, 330, 200, 76, "Skill A\nSlope analysis", 32, BLUE, align="left")
    c.text(-6, 540, 240, 44, "GeoToolKG", 38, MUTED, align="left", font=HAND)

    def place(x, top, h, f):
        return x, f(top + h / 2) - h / 2

    skill_a = {"C1", "C2", "C3", "S1", "S2", "S3"}
    graph_caps = {"C1": (292.97, 358.78), "C2": (414.79, 395.11), "C3": (554.73, 395.11),
                  "C4": (668.49, 321.2), "C5": (622.18, 414.3), "C6": (371.49, 208.91),
                  "C7": (549.69, 205.28), "C8": (582.91, 303.37), "C9": (701.71, 242.52),
                  "C10": (787.29, 277.94)}
    example_caps = {"S1": (374.95, 909.79), "S2": (671.95, 921.59), "S3": (946.79, 909.79)}
    caps = {k: place(x, y, CAP_D[1], yg) for k, (x, y) in graph_caps.items()}
    caps |= {k: place(x, y, CAP_D[1], ye) for k, (x, y) in example_caps.items()}
    graph_tools = {"T1": (241.07, 574.37), "T2": (217.51, 716.6), "T3": (393.09, 631.59),
                   "T4": (534.04, 573.46), "T5": (535.05, 685.18), "T6": (655.86, 631.59),
                   "T7": (706.19, 549.44), "T8": (324.63, 775.11), "T9": (451.48, 725.15),
                   "T10": (446.45, 795.01), "T11": (598.47, 770.48), "T12": (660.89, 703.72)}
    example_tools = {"U1": (403.14, 1009.7), "U2": (332.67, 1113.25), "U3": (637.72, 1051.48),
                     "U4": (908.54, 1009.7), "U5": (914.58, 1113.25)}
    mapped = {"T1", "T2", "T3", "T4", "T5", "U1", "U2", "U3", "U4", "U5"}
    tools = {k: place(x, y + ds, TOOL_D[1], yg) for k, (x, y) in graph_tools.items()}
    tools |= {k: place(x, y, TOOL_D[1], ye) for k, (x, y) in example_tools.items()}
    box = {k: c.box(x, y, *CAP_D, "ellipse") for k, (x, y) in caps.items()}
    box.update({k: c.box(x, y, *TOOL_D) for k, (x, y) in tools.items()})

    c.tube([(caps[k][0] + CAP_D[0] / 2, caps[k][1] + CAP_D[1] / 2) for k in ("C1", "C2", "C3")],
           half=28, extend=8, color=BLUE, sw=3)

    def lower_waypoints(src, tgt, pts):
        # Waypoints follow the tool layer in proportion to how far down they sit.
        y0, y1 = graph_caps[src][1], graph_tools[tgt][1]
        return [(x, yg(y + ds * min(1.0, max(0.0, (y - y0) / (y1 - y0))))) for x, y in pts]

    mapping: dict[str, Any] = dict(color=LINE, sw=2, dash="6 5", src_gap=5, tgt_gap=5)
    for s, t, wps in [("C1", "T1", [(265.78, 507.74)]),
                      ("C1", "T2", [(330.22, 535.9), (311.09, 649.44)]),
                      ("C2", "T3", []), ("C3", "T4", []),
                      ("C3", "T5", [(615.13, 566.78), (606.07, 674.87), (581.91, 716.65)])]:
        c.connect(box[s], box[t], lower_waypoints(s, t, wps), curved=bool(wps), **mapping)
    for s, t, wps in [("S1", "U1", []), ("S1", "U2", [(322.6, 1016.97), (324.61, 1070.56)]),
                      ("S2", "U3", []), ("S3", "U4", [])]:
        c.connect(box[s], box[t], [(x, ye(y)) for x, y in wps], curved=bool(wps), **mapping)
    c.line([(x, ye(y)) for x, y in [(976.73, 927.77), (1012.69, 969.75), (1009.67, 1049.68),
                                    (975.44, 1100.54), (956.31, 1108.96)]],
           color=LINE, sw=2, dash="6 5", curved=True)

    arrow: dict[str, Any] = dict(color=MUTED, sw=2.2, end="classic", size=8, src_gap=5,
                                 tgt_gap=5)
    for s, t in [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C1", "C6"),
                 ("C6", "C7"), ("C7", "C8"), ("C4", "C10"), ("C10", "C9"), ("C9", "C8"),
                 ("T1", "T3"), ("T2", "T3"), ("T3", "T4"), ("T3", "T5"), ("T4", "T6"),
                 ("T5", "T6"), ("T6", "T7"), ("T8", "T9"), ("T10", "T11"), ("T11", "T12"),
                 ("S1", "S2"), ("S2", "S3"), ("U1", "U3"), ("U2", "U3"), ("U3", "U4"),
                 ("U3", "U5")]:
        c.connect(box[s], box[t], **arrow)

    for k, (x, y) in caps.items():
        c.ellipse(x, y, *CAP_D, fill=BLUE if k in skill_a else NODE, stroke=INK, sw=1.5)
    for k, (x, y) in tools.items():
        if k in mapped:
            c.rect(x, y, *TOOL_D, fill=BLUE_SOFT, stroke=BLUE, sw=3, r=3)
        else:
            c.rect(x, y, *TOOL_D, fill="#FFFFFF", stroke=MUTED, sw=2.5, r=3)

    legend(c, 835, 318, 312, [
        (lambda x, y: c.rect(x - 30, y - 13, 60, 26, stroke=BLUE, sw=3, r=13), "Skill"),
        (lambda x, y: port_symbol(c, x, y), "Data transfer"),
        (lambda x, y: c.ellipse(x - 15, y - 13.6, *CAP_D, fill=NODE, stroke=INK, sw=1.5),
         "Capability"),
        (lambda x, y: c.rect(x - 17, y - 15.4, *TOOL_D, fill="#FFFFFF", stroke=MUTED, sw=2.5, r=3),
         "Tool"),
        (lambda x, y: c.line([(x - 32, y), (x + 32, y)], color=LINE, sw=2, dash="6 5"),
         "Mapping"),
    ], row_h=66, size=36)

    # Example: Skill A's capabilities and their candidate tools.
    c.line([(29.63, 812), (1015.25, 812)], color=FAINT, sw=2)
    c.text(40, ye(995) - 50, 260, 100, "Skill A:\nSlope analysis", 38, INK, align="left")
    for k, label in [("S1", "Reproject DEM"), ("S2", "Crop DEM"), ("S3", "Slope")]:
        x, y = caps[k]
        c.text(x - 110, y - 44, 250, 38, label, 34, INK, bg="#FFFFFF")
    for label, x, y in [("GDAL Warp", 320.28, 988), ("Rasterio Warp", 262, 1088),
                        ("GDAL Crop", 549.12, 1025), ("GDAL Slope", 776, 982),
                        ("GRASS Slope", 770, 1085)]:
        c.text(x, ye(y) - 22, 211.42, 34, label, 32, MUTED, bg="#FFFFFF")


# ---------------------------------------------------------------------------
# Panel c: cross-Skill completion through a bridging Skill.
# ---------------------------------------------------------------------------
C_FRAME = (1239.68, 868, 1736.32, 1170.5)
CAPABILITY_ICON = {
    "C0": "c-icon-load-elevation", "C1": "c-icon-fill-pits", "C2": "c-icon-d8-directions",
    "C9": "c-icon-flow-accumulation", "C3": "c-icon-counts-to-area",
    "C4": "c-icon-extract-channels", "C5": "c-icon-drainage-map",
}


def yc(y: float) -> float:
    return 168 + (y - 145) * 0.85


def cap_node(c: Canvas, x, y, d, kind="plain") -> None:
    """Capability marker: plain dot, source ring, or target bullseye."""
    if kind == "source":
        c.ellipse(x, y, d, d, fill="#FFFFFF", stroke=INK, sw=4.5)
    elif kind == "target":
        c.ellipse(x, y, d, d, fill="#FFFFFF", stroke=INK, sw=3)
        c.ellipse(x + d * 0.28, y + d * 0.28, d * 0.44, d * 0.44, fill=INK)
    else:
        c.ellipse(x, y, d, d, fill=NODE, stroke=INK, sw=1.2)


def panel_c(c: Canvas) -> None:
    panel_frame(c, *C_FRAME, "c")
    c.origin(1263.68, C_FRAME[1] + 52)

    c.speech_bubble(96, -30, 1196, 156, r=24, tail=(24, 56, 6, 28), color=LINE, sw=2.5,
                    fill=PAPER)
    for y, s in [(-18, "Fill artificial pits in this elevation map. Map channels with"),
                 (28, "\u22651 km\u00b2 upstream area; report slope, flow-direction and"),
                 (74, "coverage distributions, plus the channel-cell count.")]:
        c.text(124, y, 1150, 46, s, 42, INK, align="left")

    # Original (x, top, diameter); y is compressed through yc around the centre.
    raw = {"C0": (152.82, 362.22, 31.66), "C1": (419.07, 254.81, 28.64),
           "C2": (584.89, 363.61, 28.64), "C3": (898.62, 256.41, 28.64),
           "C4": (897.68, 511.62, 28.64), "C5": (1169.39, 340.62, 31.66),
           "C6": (311.67, 510.82, 28.64), "C7": (584.89, 538.64, 28.64),
           "C8": (1012.81, 622.4, 22.61), "C9": (741.75, 310.01, 28.64)}
    caps = {k: (x, yc(y + d / 2) - d / 2, d) for k, (x, y, d) in raw.items()}
    box = {k: c.box(x, y, d, d, "ellipse") for k, (x, y, d) in caps.items()}

    def centre(k):
        x, y, d = caps[k]
        return x + d / 2, y + d / 2

    c.tube([centre(k) for k in ("C0", "C1", "C2")], half=36, color=BLUE, sw=4)
    c.tube([centre(k) for k in ("C3", "C4", "C5")], half=38, color=GREEN, sw=4)
    c.tube([centre(k) for k in ("C2", "C9", "C3")], half=34, color=ORANGE, sw=3.5, dash="9 6")
    label_along(c, centre("C0"), centre("C1"), "Skill A", 62, BLUE)
    label_along(c, centre("C2"), centre("C9"), "Skill C", 64, ORANGE)
    label_along(c, centre("C4"), centre("C5"), "Skill B", 72, GREEN, below=True, t=0.72)

    edges = [("C0", "C1"), ("C1", "C2"), ("C2", "C9"), ("C9", "C3"), ("C3", "C4"), ("C4", "C5"),
             ("C0", "C6"), ("C6", "C7"), ("C2", "C7"), ("C4", "C7"), ("C4", "C8")]
    bridge = {("C2", "C9"), ("C9", "C3")}
    for s, t in edges:
        c.connect(box[s], box[t], src_gap=13.5, tgt_gap=7.5, color=ORANGE if (s, t) in bridge
                  else MUTED, sw=2.8, end="classic", size=7.5)
    for s, t in edges:
        c.port_arc(box[s], box[t], r=caps[s][2] / 2 + 14, color=MUTED, sw=2.8)
    for k, (x, y, d) in caps.items():
        cap_node(c, x, y, d, {"C0": "source", "C5": "target"}.get(k, "plain"))

    # Capability icons beside their nodes (original centres, compressed).
    for k, cx, cy, s in [("C0", 80, 377.7, 64), ("C1", 432.7, 180.1, 64),
                         ("C2", 663.4, 450, 64), ("C3", 1022.2, 268.4, 64),
                         ("C4", 856.3, 621.8, 64), ("C5", 1278.4, 362.4, 64),
                         ("C9", 776, 418, 58)]:
        c.icon(CAPABILITY_ICON[k], cx - s / 2, yc(cy) - s / 2, s, s, tint=INK)

    update_glyph(c, 1336, -28)

    legend(c, 1326, 158, 372, [
        (lambda x, y: port_symbol(c, x, y), "Data transfer"),
        (lambda x, y: cap_node(c, x - 14, y - 14, 28), "Capability"),
        (lambda x, y: cap_node(c, x - 16, y - 16, 32, "source"), "Source capability"),
        (lambda x, y: cap_node(c, x - 16, y - 16, 32, "target"), "Target capability"),
        (lambda x, y: c.rect(x - 32, y - 14, 64, 28, stroke=BLUE, sw=3.5, r=14), "Retrieved Skill"),
        (lambda x, y: c.rect(x - 32, y - 14, 64, 28, stroke=ORANGE, sw=3.5, r=14, dash="7 5"),
         "Bridging Skill"),
    ], row_h=62, size=36)

    # Linear workflows of the three Skills.
    y0, pitch, icon = 650, 160, 70
    rows = {"A": (y0, ["C0", "C1", "C2"], BLUE, MUTED,
                  ["Load elevation data", "Fill artificial pits", "Compute D8 directions"],
                  ["Elevation grid", "Filled elevation grid"]),
            "C": (y0 + pitch, ["C2", "C9", "C3"], ORANGE, ORANGE,
                  ["Compute D8 directions", "Accumulate flow", "Convert counts to area"],
                  ["D8 direction grid", "Upstream cell counts"]),
            "B": (y0 + 2 * pitch, ["C3", "C4", "C5"], GREEN, MUTED,
                  ["Convert counts to area", "Extract channels", "Render drainage map"],
                  ["Contributing-area grid", "Channel mask"])}
    c.rect(0, rows["C"][0] - 26, 1658, 152, fill=ORANGE_SOFT, stroke=ORANGE, sw=3, r=76,
           dash="9 6", fill_opacity=0.35)
    col_x = [343.6, 893.8, 1444]
    for row, (y, keys, name_color, edge_color, names, transfers) in rows.items():
        c.text(26, y + 13, 180, 44, f"Skill {row}", 40, name_color, align="left", bold=True)
        icons = [c.icon(CAPABILITY_ICON[k], x - icon / 2, y, icon, icon, tint=INK)
                 for x, k in zip(col_x, keys)]
        for x, s in zip(col_x, names):
            c.text(x - 200, y + 76, 400, 40, s, 36, INK)
        for i in range(2):
            c.connect(icons[i], icons[i + 1], src_gap=18, tgt_gap=18, color=edge_color, sw=2.8,
                      end="classic", size=7.5)
            c.text(467.31 + 550.21 * i, y - 6, 309.02, 36, transfers[i], 32,
                   ORANGE if row == "C" else MUTED, italic=True)


def build() -> str:
    c = Canvas(PAGE_W, PAGE_H, ASSET_DIR)
    c.font = SERIF
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#FFFFFF")
    panel_a(c)
    panel_b(c)
    panel_c(c)
    return c.svg()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fig_main", help="output file stem in figures/")
    export(build(), OUT_DIR, parser.parse_args().name)


if __name__ == "__main__":
    main()
