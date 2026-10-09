#!/usr/bin/env python3
"""Draw the AgentGS planning pipeline (Section 3 column figure) and export SVG + PDF.

The page is 1430 px wide so that, scaled to \\columnwidth, text prints at the same
size as in fig_main (scaled to \\textwidth from 3000 px). Stage pills run down
the left; to their right a thumbnail shows the plan after each stage, following
the skill-guided path in agentgs.agent.engine: Skill retrieval, goal selection
and predecessor expansion compose the initial plan; support review and
shared-capability bridge search complete it; the completed plan is grounded
over GeoToolKG and executed.

Usage:
    python3 scripts/fig_pipeline.py [--name fig_pipeline]
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    RED,
    SW,
    flow,
    lightbulb,
    pill,
    skill_graph,
    tool_graph,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets" / "fig_main"
OUT_DIR = SCRIPT_DIR.parent / "figures"

PAGE_W, PAGE_H = 1430, 1484

PILL_X, PILL_W, PILL_H = 230, 320, 80
XS = [560 + i * 130 for i in range(7)]  # capability slots of the completed plan
NODE_D = 38
CAPSULE_H = 66
# (first slot, last slot, stroke, fill) of each Skill fragment.
SKILL_A = (0, 2, BLUE, BLUE_SOFT)
SKILL_B = (4, 6, GREEN, GREEN_SOFT)
SKILL_C = (2, 4, ORANGE, ORANGE_SOFT)
SLOT_COLOR = [BLUE, BLUE, BLUE, ORANGE, GREEN, GREEN, GREEN]


# -- plan thumbnails --------------------------------------------------------
def skill_capsule(c: Canvas, skill, cy, dash=None, fill=True) -> None:
    lo, hi, stroke, soft = skill
    x0, x1 = XS[lo] - 46, XS[hi] + 46
    c.rect(x0, cy - CAPSULE_H / 2, x1 - x0, CAPSULE_H, fill=soft if fill else "none",
           stroke=stroke, sw=3, r=CAPSULE_H / 2, dash=dash)


def node(c: Canvas, i, cy, color, kind="plain") -> None:
    x, r = XS[i], NODE_D / 2
    if kind == "goal":
        c.ellipse(x - r, cy - r, NODE_D, NODE_D, fill="#FFFFFF", stroke=color, sw=4)
        c.ellipse(x - 8, cy - 8, 16, 16, fill=color)
    elif kind == "faint":
        c.ellipse(x - 9, cy - 9, 18, 18, fill=FAINT)
    else:
        c.ellipse(x - r, cy - r, NODE_D, NODE_D, fill=color, stroke=INK, sw=2.5)


def link(c: Canvas, i, j, cy, color=INK, dash=None) -> None:
    r = NODE_D / 2 + 2
    c.line([(XS[i] + r, cy), (XS[j] - r, cy)], color=color, sw=3, end="block", size=10,
           dash=dash)


def fragments(c: Canvas, cy) -> None:
    """Selected fragments of Skills A and B with their predecessors expanded."""
    skill_capsule(c, SKILL_A, cy)
    skill_capsule(c, SKILL_B, cy)
    for lo, hi, color, _ in (SKILL_A, SKILL_B):
        for i in range(lo, hi):
            link(c, i, i + 1, cy)
        for i in range(lo, hi + 1):
            node(c, i, cy, color)


def unresolved(c: Canvas, cy) -> None:
    """Open input of Skill B's first operation: no fragment supplies it yet."""
    x0, x1 = XS[3] - 22, XS[4] - NODE_D / 2 - 2
    c.line([(x0, cy), (x1, cy)], color=RED, sw=3, end="block", size=10,
           dash=drawio_dash("3 2", 3, False))
    c.text(x0 - 38, cy - 24, 40, 48, "?", 40, RED, font=HAND, bold=True)


def retrieved(c: Canvas, cy) -> None:
    skill_graph(c, XS[0] - 10, cy - 18, 0.5)
    c.text(XS[0] - 110, cy + 34, 200, 44, "GeoSkillKG", 32, INK, font=HAND)
    c.line([(XS[0] + 66, cy - 18), (XS[1] + 40, cy - 18)], **flow(sw=3))
    for k, (name, stroke, soft, dot) in enumerate([("Skill A", BLUE, BLUE_SOFT, BLUE),
                                                   ("Skill B", GREEN, GREEN_SOFT, GREEN),
                                                   ("Skill D", LINE, PAPER, FAINT)]):
        cx = XS[2] + 60 + k * 225
        edge = LINE if dot == FAINT else INK
        c.rect(cx - 95, cy - 46, 190, 56, fill=soft, stroke=stroke, sw=3, r=28)
        c.poly([(cx - 52, cy - 18), (cx + 52, cy - 18)], stroke=edge, sw=2.5)
        for dx in (-52, 0, 52):
            c.ellipse(cx + dx - 13, cy - 31, 26, 26, fill=dot, stroke=edge, sw=2)
        c.text(cx - 100, cy + 14, 200, 44, name, 34, MUTED if dot == FAINT else stroke)


def goals(c: Canvas, cy) -> None:
    dash = drawio_dash("4 3", 3, False)
    skill_capsule(c, SKILL_A, cy, dash=dash, fill=False)
    skill_capsule(c, SKILL_B, cy, dash=dash, fill=False)
    for i in (0, 1, 4, 5):
        node(c, i, cy, FAINT, "faint")
    node(c, 2, cy, BLUE, "goal")
    node(c, 6, cy, GREEN, "goal")


def expanded(c: Canvas, cy) -> None:
    fragments(c, cy)
    unresolved(c, cy)


def support(c: Canvas, cy) -> None:
    """Supporting operations proposed from compatible Skills, reviewed for relevance."""
    fragments(c, cy + 14)
    unresolved(c, cy + 14)
    top = cy - 54
    # Accepted proposal feeds Skill A's second operation.
    c.ellipse(XS[0] + 32, top - 16, 32, 32, fill=NODE, stroke=INK, sw=2.5)
    c.line([(XS[0] + 62, top + 4), (XS[1] - 18, cy - 6)], color=INK, sw=2.5, end="block", size=8)
    c.poly([(XS[0] + 72, top - 22), (XS[0] + 80, top - 12), (XS[0] + 96, top - 32)],
           stroke=GREEN, sw=5, join="round", cap="round")
    # Rejected proposal stays outside the plan.
    x = XS[5] + 40
    c.rect(x - 16, top - 16, 32, 32, fill="#FFFFFF", stroke=LINE, sw=2.5, r=16, dash="4 3")
    c.poly([(x + 28, top - 10), (x + 44, top + 6)], stroke=RED, sw=5, cap="round")
    c.poly([(x + 44, top - 10), (x + 28, top + 6)], stroke=RED, sw=5, cap="round")


def bridge(c: Canvas, cy) -> None:
    fragments(c, cy)
    lo, hi, stroke, _ = SKILL_C
    x0, x1 = XS[lo] - 40, XS[hi] + 40
    c.rect(x0, cy - CAPSULE_H / 2 - 8, x1 - x0, CAPSULE_H + 16, stroke=stroke, sw=3.5,
           r=CAPSULE_H / 2 + 8, dash="9 6")
    dash = drawio_dash("3 2", 3, False)
    link(c, 2, 3, cy, ORANGE, dash)
    link(c, 3, 4, cy, ORANGE, dash)
    node(c, 3, cy, ORANGE)
    c.text(XS[3] - 100, cy - 84, 200, 40, "Skill C", 34, ORANGE)


def completed(c: Canvas, cy) -> None:
    """Completed capability chain, drawn like the chain in fig_main panel a."""
    y = cy - 14
    for skill in (SKILL_A, SKILL_B, SKILL_C):
        skill_capsule(c, skill, y)
    for i in range(6):
        link(c, i, i + 1, y, ORANGE if i in (2, 3) else INK)
    for i, color in enumerate(SLOT_COLOR):
        node(c, i, y, color)
    for (lo, hi, color, _), name in [(SKILL_A, "Skill A"), (SKILL_C, "Skill C"),
                                     (SKILL_B, "Skill B")]:
        c.text((XS[lo] + XS[hi]) / 2 - 100, cy + 22, 200, 44, name, 34, color)


# -- legend -------------------------------------------------------------------
def legend_grid(c: Canvas, x, y, w, rows, cols=2, row_h=54, size=32) -> None:
    """Boxed legend laid out column by column."""
    pad, sym_w = 16, 60
    n = -(-len(rows) // cols)
    c.rect(x, y, w, n * row_h + 2 * pad, fill="#FFFFFF", stroke=FAINT, sw=2, r=14)
    col_w = (w - 2 * pad) / cols
    for k, (draw, label) in enumerate(rows):
        cx = x + pad + (k // n) * col_w
        cy = y + pad + row_h * (k % n + 0.5)
        draw(cx + sym_w / 2, cy)
        c.text(cx + sym_w + 10, cy - row_h / 2, col_w - sym_w - 10, row_h, label, size, INK,
               align="left")


def build() -> str:
    c = Canvas(PAGE_W, PAGE_H, ASSET_DIR)
    c.font = SERIF
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#FFFFFF")

    # Query.
    c.icon("a-query-user", PILL_X - 48, 14, 95, 124.72)
    c.speech_bubble(330, 18, 1080, 116, r=24, tail=(30, 66, -34, 30), color=LINE, sw=2.5,
                    fill=PAPER)
    for y, s in [(30, "Is the mean of the cloud-free composite above"),
                 (76, "the supplied reference value?")]:
        c.text(362, y, 1030, 46, s, 40, INK, align="left")

    # One card per subsection.
    for top, bottom, title in [(196, 696, "Skill-guided Plan Composition"),
                               (746, 1246, "Automatic Plan Completion")]:
        c.rect(20, top, PAGE_W - 40, bottom - top, fill=PAPER, stroke=FAINT, sw=2.5, r=24)
        c.text(400, top + 8, PAGE_W - 420, 46, title, 38, MUTED)

    rows = [(320, "Retrieve", retrieved), (468, "Select Goals", goals),
            (614, "Expand", expanded), (868, "Review Support", support),
            (1016, "Search Bridges", bridge), (1164, "Insert Branch", completed)]
    for cy, label, thumb in rows:
        pill(c, PILL_X, cy, PILL_W, PILL_H, label, 40)
        thumb(c, cy)
    c.line([(PILL_X, 140), (PILL_X, rows[0][0] - PILL_H / 2)], **flow())
    for (y0, _, _), (y1, _, _) in zip(rows, rows[1:]):
        c.line([(PILL_X, y0 + PILL_H / 2), (PILL_X, y1 - PILL_H / 2)], **flow())

    # Arrow annotations.
    c.text(PILL_X - 104, 380, 90, 46, "Yes", 36, INK, align="right")
    for cy in (468, 868):
        c.text(PILL_X + 16, cy - 104, 120, 46, "LLM", 36, INK, align="left")
    c.text(PILL_X + 16, 698, 260, 46, "initial plan", 36, MUTED, align="left", italic=True)

    # Grounding over GeoToolKG, execution, answer.
    yt = 1360
    c.line([(PILL_X, rows[-1][0] + PILL_H / 2), (PILL_X, yt - 64)], **flow())
    c.text(PILL_X + 16, 1250, 360, 46, "Tool Grounding (LLM)", 36, INK, align="left")
    tool_graph(c, PILL_X, yt, 0.62)
    c.text(PILL_X - 140, yt + 54, 280, 50, "GeoToolKG", 38, INK, font=HAND)
    c.line([(PILL_X + 70, yt), (PILL_X + 250, yt)], **flow())
    c.text(PILL_X + 80, yt + 6, 160, 46, "Executor", 36, INK)
    lightbulb(c, PILL_X + 270, yt - 46, 1.1, 1.05)
    c.text(PILL_X + 210, yt + 54, 170, 50, "Answer", 38, INK, font=HAND)

    # No applicable Skill: plan directly over GeoToolKG.
    c.line([(PILL_X - PILL_W / 2, 320), (44, 320), (44, yt), (PILL_X - 82, yt)],
           **flow(MUTED, radius=10, dash=drawio_dash("4 3", SW, False)))
    c.text(54, 364, 80, 46, "No", 36, MUTED, align="left")

    def goal_symbol(x, y):
        c.ellipse(x - 19, y - 19, 38, 38, fill="#FFFFFF", stroke=INK, sw=4)
        c.ellipse(x - 8, y - 8, 16, 16, fill=INK)

    def support_symbol(x, y):
        c.ellipse(x - 26, y - 13, 26, 26, fill=NODE, stroke=INK, sw=2.5)
        c.poly([(x + 6, y), (x + 13, y + 8), (x + 27, y - 9)], stroke=GREEN, sw=5,
               join="round", cap="round")

    legend_grid(c, 760, 1276, 650, [
        (lambda x, y: c.rect(x - 30, y - 15, 60, 30, fill=BLUE_SOFT, stroke=BLUE, sw=3, r=15),
         "Skill fragment"),
        (goal_symbol, "Goal operation"),
        (lambda x, y: c.ellipse(x - 17, y - 17, 34, 34, fill=NODE, stroke=INK, sw=2.5),
         "Capability"),
        (support_symbol, "Support candidate"),
        (lambda x, y: c.line([(x - 30, y), (x + 26, y)], color=RED, sw=3, end="block", size=10,
                             dash=drawio_dash("3 2", 3, False)), "Unresolved input"),
        (lambda x, y: c.rect(x - 30, y - 15, 60, 30, stroke=ORANGE, sw=3.5, r=15, dash="7 5"),
         "Bridging Skill"),
    ])
    return c.svg()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fig_pipeline", help="output file stem in figures/")
    export(build(), OUT_DIR, parser.parse_args().name)


if __name__ == "__main__":
    main()
