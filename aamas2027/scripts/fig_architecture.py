#!/usr/bin/env python3
"""Draw the AgentGS system architecture figure and export SVG + PDF to figures/.

Layout units are draw.io pixels on an 1800 x 900 page. All illustrations are
simple enough to be drawn here; no asset files are needed.

Usage:
    python3 scripts/fig_architecture.py [--name fig_architecture]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _figlib import Canvas, drawio_dash, export

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "figures"

PAGE_W, PAGE_H = 1800, 900

INK = "#111111"
LINE = "#29353E"
FRAME = "#34414A"
DIVIDER = "#75818A"

BLUE, BLUE_DOT = "#D5EEF9", "#96D0EF"
GREEN, GREEN_DOT = "#D8F1E5", "#A9DBCA"
PURPLE, PURPLE_DOT = "#E8DDF6", "#CDB8EA"
CYAN, CYAN_DOT = "#CEF0F5", "#97DADB"
YELLOW, YELLOW_DOT = "#FFF0BC", "#FFE296"
RED = "#F3978D"
PAPER = "#F9FBFD"


def bar(c: Canvas, x, y, w, color) -> None:
    """Placeholder text line inside an illustration."""
    c.rect(x, y, w, 7, fill=color, r=3.5)


def document(c: Canvas, x, y, w, h, fold, fill, fold_fill, sw=2.3) -> None:
    """Sheet with a folded top-right corner."""
    r = 5
    c.path(f"M{r} 0 L{w - fold} 0 L{w} {fold} L{w} {h - r} Q{w} {h} {w - r} {h} "
           f"L{r} {h} Q0 {h} 0 {h - r} L0 {r} Q0 0 {r} 0 Z",
           x, y, fill=fill, stroke=LINE, sw=sw, join="round", cap="round")
    c.path(f"M0 0 L{fold} {fold} L{r} {fold} Q0 {fold} 0 {fold - r} Z",
           x + w - fold, y, fill=fold_fill, stroke=LINE, sw=sw, join="round", cap="round")


def user_interface(c: Canvas, x, y) -> None:
    """Browser window with a short chat exchange."""
    c.origin(x, y)
    c.rect(0, 0, 284, 222, fill="#FFFFFF", stroke=LINE, sw=2.3, r=14)
    c.path("M14 0 L270 0 C279 0 284 5 284 14 L284 35 L0 35 L0 14 C0 5 5 0 14 0 Z",
           0, 0, fill=BLUE, stroke=LINE, sw=2.3, join="round")
    for cx, fill in [(17.7, RED), (42.7, YELLOW), (67.7, GREEN_DOT)]:
        c.ellipse(cx, 11.2, 12.6, 12.6, fill=fill, stroke=LINE, sw=2)
    # Incoming message (tail bottom-left), outgoing reply (tail bottom-right).
    c.path("M24 0 L142 0 C154 0 161 7 161 18 L161 32 C161 43 154 50 142 50 L23 50 L0 54 "
           "L9 41 L9 18 C9 7 15 0 24 0 Z", 8, 54, fill=BLUE, stroke=LINE, sw=2.3, join="round")
    bar(c, 34, 70, 113, "#8CA6B8")
    bar(c, 34, 86, 75, "#8CA6B8")
    c.path("M17 0 L162 0 C172 0 179 7 179 17 L179 35 L188 53 L168 49 L17 49 C7 49 0 43 0 32 "
           "L0 17 C0 7 7 0 17 0 Z", 80, 119, fill=GREEN, stroke=LINE, sw=2.3, join="round")
    bar(c, 101, 134, 141, "#91ACA8")
    bar(c, 101, 150, 91, "#91ACA8")
    bar(c, 17, 187, 234, "#CDD4DB")
    bar(c, 17, 201, 121, "#CDD4DB")
    c.origin(0, 0)


def agent_orchestration(c: Canvas, x, y) -> None:
    """Capability plan: a row of three steps wrapping to a row of two."""
    c.origin(x, y)
    steps = [(0, 0, 107, BLUE, BLUE_DOT), (132, 0, 107, GREEN, GREEN_DOT),
             (264, 0, 107, PURPLE, PURPLE_DOT), (54, 124, 118, CYAN, CYAN_DOT),
             (214, 124, 118, YELLOW, YELLOW_DOT)]
    for sx, sy, w, fill, dot in steps:
        c.rect(sx, sy, w, 50, fill=fill, stroke=LINE, sw=2.3, r=25)
        c.ellipse(sx + 17.3, sy + 14.3, 21.4, 21.4, fill=dot, stroke=LINE, sw=2.3)
        c.poly([(sx + 49, sy + 25), (sx + w - 17, sy + 25)], stroke=LINE, sw=2.2)
    arrow = dict(color=LINE, sw=2.3, end="block", size=9)
    c.line([(107, 25), (132, 25)], **arrow)
    c.line([(239, 25), (264, 25)], **arrow)
    c.line([(317.5, 50), (317.5, 88), (113, 88), (113, 124)], radius=4, **arrow)
    c.line([(172, 149), (214, 149)], **arrow)
    c.origin(0, 0)


def knowledge_resources(c: Canvas, x, y) -> None:
    """Skill definition sheets next to a small tool knowledge graph."""
    c.origin(x, y)
    c.rect(0, 4, 108, 156, fill=BLUE, stroke=LINE, sw=2.3, r=10)
    document(c, 20, 22, 125, 162, 36.25, PAPER, BLUE)
    tree = {"root": (80, 66), "left": (54, 105), "right": (109, 105)}
    for leaf in ("left", "right"):
        c.poly([tree["root"], tree[leaf]], stroke=LINE, sw=2)
    for key, fill in [("root", BLUE_DOT), ("left", GREEN_DOT), ("right", PURPLE_DOT)]:
        cx, cy = tree[key]
        c.ellipse(cx - 12.3, cy - 12.3, 24.6, 24.6, fill=fill, stroke=LINE, sw=2.3)
    bar(c, 39, 137, 87, "#97A3AD")
    bar(c, 39, 158, 68, "#97A3AD")

    nodes = {"top": (241, 0, BLUE_DOT), "left": (164, 49, PURPLE),
             "right": (319, 49, YELLOW), "center": (241, 84, GREEN_DOT),
             "lower_left": (177, 142, GREEN), "lower_right": (308, 142, "#D9DCF5")}
    centre = {k: (nx + 24, ny + 24) for k, (nx, ny, _) in nodes.items()}
    for a, b in [("top", "left"), ("top", "right"), ("top", "center"), ("left", "center"),
                 ("right", "center"), ("center", "lower_left"), ("center", "lower_right"),
                 ("lower_left", "lower_right")]:
        c.poly([centre[a], centre[b]], stroke=LINE, sw=2.2)
    for nx, ny, fill in nodes.values():
        c.ellipse(nx, ny, 48, 48, fill=fill, stroke=LINE, sw=2.3)
    c.origin(0, 0)


def tool_execution(c: Canvas, x, y) -> None:
    """Input document -> toolbox -> verified result document."""
    c.origin(x, y)
    document(c, 0, 0, 77, 111, 22.33, PAPER, "#EDF3F7")
    bar(c, 15, 43, 45, "#939FA8")
    bar(c, 15, 64, 45, "#939FA8")

    c.rect(143, 15, 64, 37, fill=YELLOW, stroke=LINE, sw=2.3, r=9)
    c.rect(153, 24, 44, 29, fill="#FFFFFF", stroke=LINE, sw=2.3, r=4)
    c.rect(121, 60, 110, 51, fill=CYAN, stroke=LINE, sw=2.3, r=6)
    c.rect(119, 39, 114, 31, fill=CYAN, stroke=LINE, sw=2.3, r=6)
    c.rect(168, 54, 17, 28, fill=RED, stroke=LINE, sw=2, r=1)

    document(c, 288, 0, 77, 120, 22.33, GREEN, "#EBF8F0")
    c.poly([(310, 54), (321, 66), (344, 37)], stroke="#4FAD93", sw=8, join="round", cap="round")
    bar(c, 305, 80, 49, "#8CAAA1")
    bar(c, 305, 97, 38, "#8CAAA1")

    arrow = dict(color=LINE, sw=2.2, end="block", size=9)
    c.line([(85, 63), (111, 63)], **arrow)
    c.line([(242, 63), (279, 63)], **arrow)
    c.origin(0, 0)


def build() -> str:
    c = Canvas(PAGE_W, PAGE_H)
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#FFFFFF")
    c.rect(32, 130, 1736, 400, fill="#FFFFFF", stroke=FRAME, sw=2, r=18)
    c.rect(32, 620, 1736, 250, fill="#FFFFFF", stroke=FRAME, sw=2, r=18)

    # Module boundaries: slanted dashed lines broken by the interaction arrows.
    dash = drawio_dash("5 4", 2, False)
    for top, arrow_xy, bottom in [((425, 130), (421, 281), (411, 530)),
                                  ((823.6, 132), (828, 284), (834.2, 531.2)),
                                  ((1288.9, 129.2), (1291, 284), (1297, 530))]:
        b = c.box(*arrow_xy, 100, 60)
        c.line([top, b.perimeter(top)], color=DIVIDER, sw=2, dash=dash)
        c.line([b.perimeter(bottom), bottom], color=DIVIDER, sw=2, dash=dash)
        c.fat_arrow(*arrow_xy, 100, 60, double=True)

    user_interface(c, 95, 177)
    agent_orchestration(c, 477, 215)
    knowledge_resources(c, 934, 203)
    tool_execution(c, 1396, 245)

    c.text(110, 28, 1580, 66, "AgentGS System Architecture", 52, INK, bold=True)
    for x, y, w, h, label in [(45, 437, 382, 66, "User Interface"),
                              (464, 437, 406, 66, "Agent Orchestration"),
                              (908, 431, 410, 83, "Knowledge and\nCapability Resources"),
                              (1340, 437, 409, 66, "Tool Execution Service")]:
        c.text(x, y, w, h, label, 33, INK)

    # Function lists; each entry is (bullet y, text y, text height, label).
    columns = [
        (68, 103, 325, [(668, 650, 42, "Task Dialogue"), (718, 700, 42, "Process Inspection"),
                        (768, 750, 42, "Graph Browsing")]),
        (474, 508, 365, [(660, 642, 76, "Capability Planning and\nCompletion"),
                         (736, 718, 42, "Tool Selection and Binding"),
                         (778, 760, 76, "Execution Feedback and\nRecovery")]),
        (940, 976, 344, [(668, 650, 42, "Skill Definitions"),
                         (718, 700, 42, "Capability–Tool Mapping"),
                         (768, 750, 42, "Tool Knowledge Graph")]),
        (1376, 1410, 328, [(668, 650, 42, "Unified Invocation"), (718, 700, 42, "Tool Execution"),
                           (768, 750, 42, "Result Return")]),
    ]
    for bx, tx, tw, items in columns:
        for by, ty, th, label in items:
            c.ellipse(bx, by, 7, 7, fill=INK)
            c.text(tx, ty, tw, th, label, 28, INK, align="left")

    for x in (237, 666, 1113, 1549):
        c.line([(x, 530), (x, 607)], color=LINE, sw=2.1, end="block", size=12)
    return c.svg()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fig_architecture", help="output file stem in figures/")
    export(build(), OUT_DIR, parser.parse_args().name)


if __name__ == "__main__":
    main()
