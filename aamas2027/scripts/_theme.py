"""Visual language shared by the AgentGS method figures (fig_main, fig_pipeline).

Palette, fonts, flow arrows, component pills, Skill capsules and the recurring
glyphs (SkillSet, GeoSkillKG, GeoToolKG, documents, answer bulb) live here so
every figure draws them identically.
"""

from __future__ import annotations

from typing import Any, Callable

from _figlib import Box, Canvas

# Component names and text inside glyphs use a hand-drawn face; everything
# descriptive (queries, legends, annotations) stays in the serif body face.
HAND = "'Comic Sans MS', 'Comic Neue', 'Chalkboard SE', cursive"

# Palette.
INK = "#1F2933"
MUTED = "#52606D"
LINE = "#9AA5B1"
FAINT = "#CBD2D9"
PAPER = "#F5F7FA"
MODULE = "#EEF3F8"
NODE = "#4A5864"
BLUE, BLUE_SOFT = "#3B6EA8", "#DCE7F3"
GREEN, GREEN_SOFT = "#3E8E64", "#DCEFE4"
ORANGE, ORANGE_SOFT = "#E07B39", "#FBE3D1"
RED, RED_SOFT = "#C8473D", "#F8DEDB"
PURPLE = "#7A5BA6"
YELLOW = "#F4CF62"

SW = 3.5  # main-flow stroke width


def flow(color: str = INK, sw: float = SW, **kw) -> dict[str, Any]:
    return dict(color=color, sw=sw, end="block", size=12, **kw)


def panel_frame(c: Canvas, x, y, w, h, letter: str) -> None:
    c.origin(0, 0)
    c.rect(x, y, w, h, fill="#FFFFFF", stroke=FAINT, sw=2, r=18)
    c.text(x + 20, y + 12, 60, 56, letter, 48, INK, align="left", bold=True)


def pill(c: Canvas, cx, cy, w, h, label, size=44, fill=MODULE, stroke=INK) -> Box:
    b = c.rect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, sw=3, r=h / 2)
    c.text(cx - w / 2, cy - h / 2, w, h, label, size, INK, font=HAND)
    return b


def capsule(c: Canvas, cx, cy, length, h, rot, stroke, fill, sw=3.0, dash=None, opacity=None):
    c.rect(cx - length / 2, cy - h / 2, length, h, fill=fill, stroke=stroke, sw=sw, r=h / 2,
           rot=rot, dash=dash, fill_opacity=opacity)


def legend(c: Canvas, x, y, w, rows: list[tuple[Callable[[float, float], None], str]],
           row_h=66.0, size=34) -> None:
    """Boxed legend: fixed symbol column on the left, left-aligned labels."""
    pad, gap, sym_w = 18.0, 12.0, 66.0
    c.rect(x, y, w, len(rows) * row_h + 2 * pad, fill="#FFFFFF", stroke=FAINT, sw=2, r=14)
    for i, (draw, label) in enumerate(rows):
        cy = y + pad + row_h * (i + 0.5)
        draw(x + gap + sym_w / 2, cy)
        c.text(x + 2 * gap + sym_w, cy - row_h / 2, w - sym_w - 3 * gap, row_h, label, size,
               INK, align="left")


# -- shared glyphs ---------------------------------------------------------------
def toolbox(c: Canvas, cx, cy) -> None:
    c.rect(cx - 36, cy - 66, 72, 48, stroke=INK, sw=6, r=12)
    c.rect(cx - 78, cy - 32, 156, 96, fill=BLUE_SOFT, stroke=INK, sw=3.5, r=10)
    c.poly([(cx - 78, cy + 2), (cx + 78, cy + 2)], stroke=INK, sw=3)
    c.rect(cx - 13, cy - 12, 26, 30, fill=ORANGE, stroke=INK, sw=2.5, r=4)


def skill_graph(c: Canvas, cx, cy, s=1.0) -> None:
    """GeoSkillKG: three Skills crossing at a shared capability, scaled by s."""
    for rot, stroke, fill in [(0, BLUE, BLUE_SOFT), (45, GREEN, GREEN_SOFT),
                              (-45, ORANGE, ORANGE_SOFT)]:
        capsule(c, cx, cy, 216 * s, 44 * s, rot, stroke, fill, opacity=0.75)
    ends = [(-82, 0, BLUE), (82, 0, BLUE), (-58, -58, ORANGE), (58, 58, ORANGE),
            (-58, 58, GREEN), (58, -58, GREEN)]
    r = 15 * s
    for dx, dy, fill in ends:
        c.ellipse(cx + dx * s - r, cy + dy * s - r, 2 * r, 2 * r, fill=fill, stroke=INK, sw=2)
    c.ellipse(cx - 17 * s, cy - 17 * s, 34 * s, 34 * s, fill=INK, stroke=INK, sw=2)


def tool_graph(c: Canvas, cx, cy, s=1.0) -> None:
    """GeoToolKG: a small tool network, scaled by s."""
    pts = [(0, -68), (-62, -22), (10, 14), (68, -36), (82, 30), (40, 78), (-50, 66)]
    fills = [BLUE_SOFT, YELLOW, GREEN_SOFT, GREEN_SOFT, "#E5DCF0", BLUE_SOFT, "#E5DCF0"]
    p = [(cx + dx * s, cy + dy * s) for dx, dy in pts]
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6),
                 (3, 4), (4, 5), (5, 6)]:
        c.poly([p[a], p[b]], stroke=MUTED, sw=2.5)
    r = 19 * s
    for (x, y), fill in zip(p, fills):
        c.ellipse(x - r, y - r, 2 * r, 2 * r, fill=fill, stroke=INK, sw=2.5)


def document(c: Canvas, x, y, w, h, fold, fill, stroke, sw=3.0) -> None:
    c.poly([(x, y), (x + w - fold, y), (x + w, y + fold), (x + w, y + h), (x, y + h)],
           closed=True, fill=fill, stroke=stroke, sw=sw, join="round")
    c.poly([(x + w - fold, y), (x + w - fold, y + fold), (x + w, y + fold)], fill=stroke,
           stroke=stroke, sw=sw, join="round")


def lightbulb(c: Canvas, x: float, y: float, sx: float = 1.418, sy: float = 1.319) -> None:
    """Answer bulb with rays, drawn in a 41 x 53 glyph frame placed at (x, y)."""
    c.path("M8 53 L8 45 C6 40 0 37 0 27 C0 0 41 0 41 27 C41 37 34 42 32 46 L32 53 Z",
           x, y, sx, sy, fill=YELLOW, stroke=INK, sw=3.5)
    c.path("M9 62.3 L31 62.3", x, y, sx, sy, stroke=INK, sw=4.2, cap="round")
    c.path("M15 67.7 C16 72.7 25 72.7 27 67.7", x, y, sx, sy, stroke=INK, sw=4, cap="round")
    for ray in ("M20 -9 L20 -1", "M-7 1 L-1 7", "M41 7 L47 1", "M-17 28 L-8 28", "M49 28 L58 28"):
        c.path(ray, x, y, sx, sy, stroke=INK, sw=3.5, cap="round")
