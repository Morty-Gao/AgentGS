#!/usr/bin/env python3
"""Draw the GeoToolEval figure (composition + per-domain FAS) and export SVG + PDF.

Panel a shows what the benchmark contains: the ten research themes sized by
record count (inner ring) and each theme's split by reference chain length
(outer ring). Panel b compares agents and LLMs across the same ten themes.
Layout units are pixels on a 3000 x 1050 page scaled to \\textwidth, matching
the print scale of fig_main.

Usage:
    python3 scripts/fig_geotooleval.py [--name fig_geotooleval]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _figlib import SERIF, Canvas, export, num
from _theme import BLUE, FAINT, GREEN, HAND, INK, LINE, MUTED, PURPLE, RED, panel_frame

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "figures"

PAGE_W, PAGE_H = 3000, 1050
A_FRAME = (24, 24, 1180, 1002)
B_FRAME = (1228, 24, 1748, 1002)
CY = 500  # shared centre height of the donut and the radar

# Themes in the order of tables/benchmark.tex: short name, colour, and record
# counts with 1, 2-3 and >=4 reference calls.
THEMES = [
    ("Deep Earth", "#8C6D46", (85, 15, 0)),
    ("GeoQuestions", "#5B7DB1", (40, 0, 0)),
    ("Critical Minerals", "#B0707E", (44, 14, 0)),
    ("Climate Change", "#D08C3E", (31, 16, 1)),
    ("Energy Resources", "#D9B23F", (100, 0, 0)),
    ("Ocean Carbon", "#3C8DA0", (100, 0, 0)),
    ("Deep-Time Biosphere", "#6E9F4E", (12, 7, 5)),
    ("CCS", "#7E7BB5", (12, 5, 7)),
    ("Remote Sensing", "#4C6A8A", (0, 7, 42)),
    ("Natural Hazards", "#C0504D", (0, 55, 5)),
]
CHAIN = [("1 call", "#DCE3EA"), ("2–3 calls", "#93A8BD"), ("≥4 calls", "#3E5C76")]
TEETH = (34, 68, 102)  # outer-ring depth per chain-length class
MIN_TOOTH = 3.0  # degrees

# PLACEHOLDER per-theme mean FAS (x100), in THEMES order; replace with the
# evaluated results before submission. Drawing order is back to front; agent
# frameworks (solid) run on AGENT_BACKBONE, Bash-only agents are dashed.
AGENT_BACKBONE = "DeepSeek V4.1 Flash"
METHODS = [
    ("GeoGPT", "#C99A2E", True, [5, 10, 10, 35, 15, 30, 5, 10, 35, 55]),
    ("DeepSeek V4.1 Flash", PURPLE, True, [70, 10, 30, 20, 40, 72, 5, 5, 65, 72]),
    ("EarthAgent", INK, False, [5, 3, 3, 5, 3, 3, 5, 15, 85, 5]),
    ("GeoAgent", GREEN, False, [88, 62, 70, 55, 95, 97, 10, 30, 65, 92]),
    ("AgentGS (Tool Only)", BLUE, False, [92, 58, 40, 50, 96, 96, 40, 20, 18, 85]),
    ("AgentGS", RED, False, [95, 75, 78, 62, 98, 99, 70, 65, 80, 98]),
]


def polar(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def sector(c: Canvas, cx, cy, r0, r1, a0, a1, fill, stroke="#FFFFFF", sw=3.0) -> None:
    """Annular sector between radii r0 < r1 from angle a0 to a1 (degrees, clockwise)."""
    gx, gy = c.p(cx, cy)
    p = [polar(gx, gy, r, a) for r, a in [(r1, a0), (r1, a1), (r0, a1), (r0, a0)]]
    large = 1 if a1 - a0 > 180 else 0
    d = (f"M{num(p[0][0])},{num(p[0][1])} A{num(r1)},{num(r1)} 0 {large} 1 "
         f"{num(p[1][0])},{num(p[1][1])} L{num(p[2][0])},{num(p[2][1])} "
         f"A{num(r0)},{num(r0)} 0 {large} 0 {num(p[3][0])},{num(p[3][1])} Z")
    c.body.append(f'<path d="{d}" {c.paint(fill, stroke, sw)}/>')


def theme_badge(c: Canvas, x, y, k, r=20) -> None:
    c.ellipse(x - r, y - r, 2 * r, 2 * r, fill=THEMES[k][1])
    c.text(x - r, y - r, 2 * r, 2 * r, str(k + 1), 24, "#FFFFFF", font=HAND, bold=True)


# ---------------------------------------------------------------------------
# Panel a: benchmark composition.
# ---------------------------------------------------------------------------
def panel_a(c: Canvas) -> None:
    panel_frame(c, *A_FRAME, "a")
    c.origin(A_FRAME[0], A_FRAME[1])
    cx, cy = 380, CY
    r0, r1 = 130, 230
    base = r1 + 6
    total = sum(sum(n) for _, _, n in THEMES)
    angle = -90.0
    for k, (_, color, counts) in enumerate(THEMES):
        span = 360 * sum(counts) / total
        sector(c, cx, cy, r0, r1, angle, angle + span, color)
        # Outer teeth: one per chain-length class, longer for longer chains.
        # A one-record tooth would be a hairline, so teeth get a minimum width
        # and are drawn shortest first; a widened tooth overlaps its neighbour.
        a, teeth = angle, []
        for (_, shade), tooth, n in zip(CHAIN, TEETH, counts):
            if n:
                w = 360 * n / total
                pad = max(0.0, MIN_TOOTH - w) / 2
                teeth.append((tooth, a - pad, a + w + pad, shade))
            a += 360 * n / total
        for tooth, a0, a1, shade in sorted(teeth):
            sector(c, cx, cy, base, base + tooth, a0, a1, shade, sw=2.5)
        mx, my = polar(cx, cy, (r0 + r1) / 2, angle + span / 2)
        c.text(mx - 30, my - 24, 60, 48, str(k + 1), 34, "#FFFFFF", font=HAND, bold=True)
        angle += span
    c.text(cx - 120, cy - 64, 240, 80, str(total), 72, INK, bold=True)
    c.text(cx - 120, cy + 8, 240, 50, "records", 36, MUTED)

    # Theme list: numbered badges stacked vertically, centred on the donut.
    x, pitch = 750, 64
    y = cy - 4.5 * pitch
    c.text(x - 2, y - 76, 380, 50, "Research theme", 34, INK, align="left", bold=True)
    for k, (name, _, counts) in enumerate(THEMES):
        yy = y + k * pitch
        theme_badge(c, x + 20, yy, k)
        c.text(x + 54, yy - 25, 300, 50, name, 31, INK, align="left")
        c.text(x + 290, yy - 25, 90, 50, str(sum(counts)), 31, MUTED, align="right")

    # Chain-length key: tooth icons in one row beneath both.
    y = 932
    c.text(40, y - 25, 300, 50, "Reference chain", 34, INK, align="left", bold=True)
    for k, ((label, shade), tooth) in enumerate(zip(CHAIN, TEETH)):
        x = 330 + k * 280
        h = 12 + tooth * 0.3
        c.rect(x, y + 16 - h, 34, h, fill=shade)
        n = sum(t[2][k] for t in THEMES)
        c.text(x + 48, y - 25, 220, 50, f"{label} ({n})", 32, INK, align="left")


# ---------------------------------------------------------------------------
# Panel b: per-theme FAS radar.
# ---------------------------------------------------------------------------
def panel_b(c: Canvas) -> None:
    panel_frame(c, *B_FRAME, "b")
    c.origin(B_FRAME[0], B_FRAME[1])
    cx, cy, R = 660, CY, 340
    n = len(THEMES)
    angles = [-90 + 360 * k / n for k in range(n)]

    for level in (20, 40, 60, 80, 100):
        ring = [polar(cx, cy, R * level / 100, a) for a in angles]
        c.poly(ring, closed=True, stroke=FAINT if level < 100 else LINE, sw=2 if level < 100 else 2.5)
    for a in angles:
        c.poly([(cx, cy), polar(cx, cy, R, a)], stroke=FAINT, sw=2)

    for name, color, dashed, values in METHODS:
        pts = [polar(cx, cy, R * v / 100, a) for v, a in zip(values, angles)]
        main = name == "AgentGS"
        c.poly(pts, closed=True, stroke=color, sw=5.5 if main else 3.2,
               dash="10 7" if dashed else None, join="round")
        for x, y in pts:
            r = 7.5 if main else 5.5
            c.ellipse(x - r, y - r, 2 * r, 2 * r, fill=color)

    # Ring values sit above the series so lines never hide them.
    for level in (20, 40, 60, 80, 100):
        x, y = polar(cx, cy, R * level / 100, angles[0] + 18)
        c.rect(x - 2, y - 34, 52, 34, fill="#FFFFFF", r=6)
        c.text(x, y - 34, 48, 34, str(level), 28, MUTED)

    # Axis labels: numbered badge at the tip, name outside.
    for k, a in enumerate(angles):
        bx, by = polar(cx, cy, R + 42, a)
        theme_badge(c, bx, by, k)
        name, ca = THEMES[k][0], math.cos(math.radians(a))
        if abs(ca) < 0.2:
            dy = -64 if math.sin(math.radians(a)) < 0 else 22
            c.text(bx - 200, by + dy, 400, 44, name, 32, INK)
        elif ca > 0:
            c.text(bx + 28, by - 22, 320, 44, name, 32, INK, align="left")
        else:
            c.text(bx - 348, by - 22, 320, 44, name, 32, INK, align="right")

    # Legend, grouped into tool-using agent frameworks and Bash-only agents.
    x, y = 1330, CY - 290
    c.text(x, y - 25, 400, 50, "Mean FAS (×100)", 34, INK, align="left", bold=True)
    groups = [(f"Agents ({AGENT_BACKBONE})", [m for m in reversed(METHODS) if not m[2]]),
              ("Bash-only", [m for m in reversed(METHODS) if m[2]])]
    y += 76
    for title, members in groups:
        c.text(x, y - 25, 400, 50, title, 30, MUTED, align="left", italic=True)
        y += 60
        for name, color, dashed, _ in members:
            main = name == "AgentGS"
            c.poly([(x, y), (x + 70, y)], stroke=color, sw=5.5 if main else 3.2,
                   dash="10 7" if dashed else None)
            c.ellipse(x + 28, y - 7, 14, 14, fill=color)
            c.text(x + 88, y - 25, 300, 50, name, 32, INK, align="left", bold=main)
            y += 64
        y += 24


def build() -> str:
    c = Canvas(PAGE_W, PAGE_H)
    c.font = SERIF
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#FFFFFF")
    panel_a(c)
    panel_b(c)
    return c.svg()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fig_geotooleval", help="output file stem in figures/")
    export(build(), OUT_DIR, parser.parse_args().name)


if __name__ == "__main__":
    main()
