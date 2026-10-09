"""Shared SVG drawing helpers for the paper figures.

Coordinates follow draw.io conventions (pixels, y down) so layouts can be read
straight off a draw.io page. `Canvas` collects SVG elements; `export` writes the
SVG and converts it to PDF.
"""

from __future__ import annotations

import base64
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

SERIF = "'Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', serif"
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"

INK = "#202222"
GRAY = "#555E60"


def num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def pts_str(points) -> str:
    return " ".join(f"{num(x)},{num(y)}" for x, y in points)


def drawio_dash(pattern: str | None, sw: float, fixed: bool) -> str:
    """draw.io scales dash patterns by stroke width unless fixDash=1."""
    values = [float(v) for v in (pattern or "3 3").split()]
    if not fixed:
        values = [v * sw for v in values]
    return " ".join(num(v) for v in values)


class Box:
    def __init__(self, x: float, y: float, w: float, h: float, kind: str = "rect"):
        self.x, self.y, self.w, self.h, self.kind = x, y, w, h, kind

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def perimeter(self, toward, spacing: float = 0.0) -> tuple[float, float]:
        """Point where the ray from the centre toward `toward` leaves the shape
        grown by `spacing` (draw.io perimeter spacing)."""
        cx, cy = self.center
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        a, b = self.w / 2 + spacing, self.h / 2 + spacing
        if self.kind == "ellipse":
            t = 1 / math.hypot(dx / a, dy / b)
        else:
            t = min(a / abs(dx) if dx else math.inf, b / abs(dy) if dy else math.inf)
        return cx + dx * t, cy + dy * t


class Canvas:
    def __init__(self, width: float, height: float, asset_dir: Path | None = None):
        self.width, self.height = width, height
        self.asset_dir = asset_dir
        self.font = SERIF  # default family for text() when no font is given
        self.defs: list[str] = []
        self.body: list[str] = []
        self.ox = 0.0
        self.oy = 0.0
        self._next_id = 0
        self._icon_cache: dict[Path, str] = {}
        self._tint_cache: dict[str, str] = {}

    # -- helpers -----------------------------------------------------------
    def origin(self, x: float, y: float) -> None:
        self.ox, self.oy = x, y

    def uid(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}{self._next_id}"

    def p(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x, self.oy + y

    def box(self, x, y, w, h, kind="rect") -> Box:
        return Box(self.ox + x, self.oy + y, w, h, kind)

    def gradient(self, c1: str, c2: str, direction: str) -> str:
        # draw.io: fillColor at the start, gradientColor toward `direction`.
        x1, y1, x2, y2 = {
            "south": (0, 0, 0, 1),
            "north": (0, 1, 0, 0),
            "east": (0, 0, 1, 0),
            "west": (1, 0, 0, 0),
        }[direction]
        gid = self.uid("g")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
            f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
            "</linearGradient>"
        )
        return f"url(#{gid})"

    @staticmethod
    def paint(fill="none", stroke=None, sw=1.0, dash=None, fill_opacity=None,
              join="miter", cap="butt") -> str:
        attrs = [f'fill="{fill}"']
        if fill_opacity is not None:
            attrs.append(f'fill-opacity="{num(fill_opacity)}"')
        if stroke:
            attrs += [f'stroke="{stroke}"', f'stroke-width="{num(sw)}"',
                      f'stroke-linejoin="{join}"', f'stroke-linecap="{cap}"']
            if dash:
                attrs.append(f'stroke-dasharray="{dash}"')
        else:
            attrs.append('stroke="none"')
        return " ".join(attrs)

    # -- shapes ------------------------------------------------------------
    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.0, r=0.0, dash=None,
             rot=0.0, fill_opacity=None, grad=None) -> Box:
        b = self.box(x, y, w, h)
        if grad:
            fill = self.gradient(fill, *grad)
        r = min(r, w / 2, h / 2)
        rr = f' rx="{num(r)}"' if r else ""
        tf = f' transform="rotate({num(rot)} {num(b.center[0])} {num(b.center[1])})"' if rot else ""
        self.body.append(
            f'<rect x="{num(b.x)}" y="{num(b.y)}" width="{num(w)}" height="{num(h)}"{rr}{tf} '
            f"{self.paint(fill, stroke, sw, dash, fill_opacity)}/>"
        )
        return b

    def ellipse(self, x, y, w, h, fill="none", stroke=None, sw=1.0) -> Box:
        b = self.box(x, y, w, h, "ellipse")
        cx, cy = b.center
        self.body.append(
            f'<ellipse cx="{num(cx)}" cy="{num(cy)}" rx="{num(w / 2)}" ry="{num(h / 2)}" '
            f"{self.paint(fill, stroke, sw)}/>"
        )
        return b

    def text(self, x, y, w, h, s, size, color=INK, align="center", font=None,
             bold=False, italic=False, rot=0.0, bg=None) -> None:
        """draw.io plain-text label: lines of 1.2 * size, vertically centred."""
        b = self.box(x, y, w, h)
        cx, cy = b.center
        lines = s.split("\n")
        lh = size * 1.2
        first = cy - len(lines) * lh / 2 + size - 1
        if align == "left":
            tx, anchor = b.x, "start"
        elif align == "right":
            tx, anchor = b.x + w, "end"
        else:
            tx, anchor = cx, "middle"
        tf = f' transform="rotate({num(rot)} {num(cx)} {num(cy)})"' if rot else ""
        if bg:
            tw = max(len(line) for line in lines) * size * 0.47
            bx = {"start": tx, "middle": tx - tw / 2, "end": tx - tw}[anchor]
            self.body.append(
                f'<rect x="{num(bx - 1)}" y="{num(cy - len(lines) * lh / 2)}" '
                f'width="{num(tw + 2)}" height="{num(len(lines) * lh)}" fill="{bg}"{tf}/>'
            )
        weight = (' font-weight="bold"' if bold else "") + (' font-style="italic"' if italic else "")
        spans = "".join(
            f'<tspan x="{num(tx)}" y="{num(first + i * lh)}">{escape(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        self.body.append(
            f'<text font-family="{font or self.font}" font-size="{num(size)}"{weight} fill="{color}" '
            f'text-anchor="{anchor}"{tf}>{spans}</text>'
        )

    def rounded_polygon(self, corners, r) -> str:
        """Closed path with each corner cut by a quadratic arc of radius r."""
        n = len(corners)
        segs = []
        for i in range(n):
            prev, cur, nxt = corners[i - 1], corners[i], corners[(i + 1) % n]
            a = _toward(cur, prev, r)
            c = _toward(cur, nxt, r)
            segs.append(("M" if i == 0 else "L") + f"{num(a[0])},{num(a[1])}")
            segs.append(f"Q{num(cur[0])},{num(cur[1])} {num(c[0])},{num(c[1])}")
        return " ".join(segs) + " Z"

    def parallelogram(self, x, y, w, h, size=0.23, r=6.0) -> list[tuple[float, float]]:
        b = self.box(x, y, w, h)
        dx = w * size
        return [(b.x, b.y + h), (b.x + dx, b.y), (b.x + w, b.y), (b.x + w - dx, b.y + h)]

    # -- edges -------------------------------------------------------------
    def edge(self, points, color=INK, sw=1.0, dash=None, end=None, size=6.0,
             curved=False, radius=0.0) -> None:
        """Polyline / curve with an optional draw.io `classic` or `block` head.
        `points` are absolute coordinates."""
        pts = [tuple(p) for p in points]
        head = None
        if end:
            pts[-1], head = _arrow_head(pts[-2], pts[-1], size, sw, end)
        if curved and len(pts) > 2:
            d = [f"M{num(pts[0][0])},{num(pts[0][1])}"]
            for i in range(1, len(pts) - 2):
                mx = (pts[i][0] + pts[i + 1][0]) / 2
                my = (pts[i][1] + pts[i + 1][1]) / 2
                d.append(f"Q{num(pts[i][0])},{num(pts[i][1])} {num(mx)},{num(my)}")
            d.append(f"Q{num(pts[-2][0])},{num(pts[-2][1])} {num(pts[-1][0])},{num(pts[-1][1])}")
            path = " ".join(d)
        elif radius and len(pts) > 2:
            d = [f"M{num(pts[0][0])},{num(pts[0][1])}"]
            for i in range(1, len(pts) - 1):
                a = _toward(pts[i], pts[i - 1], radius)
                c = _toward(pts[i], pts[i + 1], radius)
                d.append(f"L{num(a[0])},{num(a[1])} Q{num(pts[i][0])},{num(pts[i][1])} {num(c[0])},{num(c[1])}")
            d.append(f"L{num(pts[-1][0])},{num(pts[-1][1])}")
            path = " ".join(d)
        else:
            path = "M" + " L".join(f"{num(x)},{num(y)}" for x, y in pts)
        join = "round" if radius else "miter"
        self.body.append(f'<path d="{path}" {self.paint("none", color, sw, dash, join=join)}/>')
        if head:
            self.body.append(
                f'<polygon points="{pts_str(head)}" {self.paint(color, color, sw)}/>'
            )

    def line(self, points, **kw) -> None:
        """Edge between explicit panel-local points."""
        self.edge([self.p(*pt) for pt in points], **kw)

    def connect(self, src: Box, tgt: Box, waypoints=(), src_gap=0.0, tgt_gap=0.0, **kw) -> None:
        """Edge between two shapes, clipped at their perimeters like draw.io."""
        mids = [self.p(*pt) for pt in waypoints]
        start = src.perimeter(mids[0] if mids else tgt.center, src_gap)
        finish = tgt.perimeter(mids[-1] if mids else src.center, tgt_gap)
        self.edge([start, *mids, finish], **kw)

    # -- glyphs ------------------------------------------------------------
    def poly(self, points, closed=False, fill="none", stroke=INK, sw=1.0, dash=None,
             join="miter", cap="butt") -> None:
        """Polyline / polygon through panel-local points."""
        d = "M" + " L".join(f"{num(x)},{num(y)}" for x, y in (self.p(*q) for q in points))
        self.body.append(f'<path d="{d}{" Z" if closed else ""}" '
                         f"{self.paint(fill, stroke, sw, dash, join=join, cap=cap)}/>")

    def path(self, d: str, x: float, y: float, sx: float = 1.0, sy: float = 1.0,
             fill="none", stroke=INK, sw=1.0, join="miter", cap="butt") -> None:
        """Absolute-command path `d` drawn in a glyph frame scaled by (sx, sy) at (x, y)."""
        gx, gy = self.p(x, y)
        self.body.append(f'<path d="{_map_path(d, gx, gy, sx, sy)}" '
                         f"{self.paint(fill, stroke, sw, join=join, cap=cap)}/>")

    def port_arc(self, src: Box, tgt: Box, r: float, span: float = 38.0,
                 color=GRAY, sw=2.86) -> None:
        """Output port: an arc around `src` facing the edge toward `tgt`."""
        cx, cy = src.center
        tx, ty = tgt.center
        mid = math.atan2(ty - cy, tx - cx)
        a0, a1 = mid - math.radians(span) / 2, mid + math.radians(span) / 2
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        self.body.append(
            f'<path d="M{num(p0[0])},{num(p0[1])} A{num(r)},{num(r)} 0 0 1 {num(p1[0])},{num(p1[1])}" '
            f"{self.paint('none', color, sw, join='round', cap='round')}/>"
        )

    def tube(self, route, half: float, extend: float = 10.0, color=GRAY, sw=2.0,
             dash=None) -> None:
        """Skill outline: a capsule of half-width `half` around a polyline through
        capability centres (panel-local), extended by `extend` past both ends."""
        pts = [self.p(*q) for q in route]
        pts[0] = _toward(pts[0], pts[1], -extend)
        pts[-1] = _toward(pts[-1], pts[-2], -extend)
        normals = []
        for a, b in zip(pts, pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            d = math.hypot(dx, dy)
            normals.append((-dy / d, dx / d))

        def side(sign):
            out = [_shift(pts[0], normals[0], sign * half)]
            for k in range(1, len(pts) - 1):
                a = _shift(pts[k], normals[k - 1], sign * half)
                b = _shift(pts[k], normals[k], sign * half)
                out.append(_intersect(a, _sub(pts[k], pts[k - 1]), b, _sub(pts[k + 1], pts[k])))
            out.append(_shift(pts[-1], normals[-1], sign * half))
            return out

        left, right = side(1), side(-1)[::-1]
        cap = f"A{num(half)},{num(half)} 0 0 0"
        d = (f"M{num(left[0][0])},{num(left[0][1])} {_rounded(left, half)} "
             f"{cap} {num(right[0][0])},{num(right[0][1])} {_rounded(right, half)} "
             f"{cap} {num(left[0][0])},{num(left[0][1])} Z")
        self.body.append(f'<path d="{d}" {self.paint("none", color, sw, dash, join="round")}/>')

    def speech_bubble(self, x, y, w, h, r, tail, color=GRAY, sw=2.7, fill="none") -> None:
        """Rounded body (x, y, w, h) with a tail on the bottom edge.
        tail = (base_x0, base_x1, tip_x, tip_drop), offsets from the body's left edge."""
        X, Y = self.p(x, y)
        x0, x1, tx, drop = tail
        d = (f"M{num(X + r)},{num(Y)} H{num(X + w - r)} A{num(r)},{num(r)} 0 0 1 {num(X + w)},{num(Y + r)} "
             f"V{num(Y + h - r)} A{num(r)},{num(r)} 0 0 1 {num(X + w - r)},{num(Y + h)} "
             f"H{num(X + x1)} L{num(X + tx)},{num(Y + h + drop)} L{num(X + x0)},{num(Y + h)} "
             f"H{num(X + r)} A{num(r)},{num(r)} 0 0 1 {num(X)},{num(Y + h - r)} "
             f"V{num(Y + r)} A{num(r)},{num(r)} 0 0 1 {num(X + r)},{num(Y)} Z")
        self.body.append(f'<path d="{d}" {self.paint(fill, color, sw, join="round", cap="round")}/>')

    def fat_arrow(self, x, y, w, h, double=False, shaft=0.3, head=0.2, fill="#FFFFFF",
                  stroke="#000000", sw=1.0) -> Box:
        """draw.io singleArrow / doubleArrow: head length head*w, shaft thickness shaft*h."""
        dx, dy = w * head, h * shaft
        top, bot, mid = (h - dy) / 2, (h + dy) / 2, h / 2
        right = [(w - dx, top), (w - dx, 0), (w, mid), (w - dx, h), (w - dx, bot)]
        left = [(dx, bot), (dx, h), (0, mid), (dx, 0), (dx, top)] if double else [(0, bot), (0, top)]
        self.poly([(x + px, y + py) for px, py in right + left], closed=True, fill=fill,
                  stroke=stroke, sw=sw)
        return self.box(x, y, w, h)

    # -- icons -------------------------------------------------------------
    def icon(self, name: str, x, y, w, h, tint: str | None = None) -> Box:
        """Place an SVG or PNG from the asset directory, stretched to w x h.
        Each file is embedded once as a <symbol>; SVGs stay vector. `tint`
        repaints every opaque pixel in one colour (for line-art icons)."""
        path = next(self.asset_dir.glob(f"{name}.*"))
        if path not in self._icon_cache:
            sid = self.uid("icon")
            if path.suffix == ".svg":
                text = path.read_text(encoding="utf-8")
                m = re.search(r"<svg\b([^>]*)>(.*)</svg>", text, re.S)
                assert m, f"{path} is not an SVG document"
                view = re.search(r'viewBox="([^"]+)"', m.group(1))
                assert view, f"{path} needs a viewBox"
                inner, viewbox = m.group(2), view.group(1)
            else:
                raw = path.read_bytes()
                iw, ih = struct.unpack(">II", raw[16:24])
                viewbox = f"0 0 {iw} {ih}"
                data = base64.b64encode(raw).decode()
                inner = (f'<image width="{iw}" height="{ih}" '
                         f'xlink:href="data:image/png;base64,{data}"/>')
            self.defs.append(f'<symbol id="{sid}" viewBox="{viewbox}" '
                             f'preserveAspectRatio="none">{inner}</symbol>')
            self._icon_cache[path] = sid
        b = self.box(x, y, w, h)
        self.body.append(
            f'<use xlink:href="#{self._icon_cache[path]}"{self._tint(tint)} '
            f'x="{num(b.x)}" y="{num(b.y)}" '
            f'width="{num(w)}" height="{num(h)}"/>'
        )
        return b

    def _tint(self, color: str | None) -> str:
        if not color:
            return ""
        if color not in self._tint_cache:
            fid = self.uid("tint")
            self.defs.append(
                f'<filter id="{fid}" x="0" y="0" width="1" height="1" '
                f'color-interpolation-filters="sRGB"><feFlood flood-color="{color}"/>'
                f'<feComposite in2="SourceAlpha" operator="in"/></filter>'
            )
            self._tint_cache[color] = fid
        return f' filter="url(#{self._tint_cache[color]})"'

    def svg(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{num(self.width)}" height="{num(self.height)}" '
            f'viewBox="0 0 {num(self.width)} {num(self.height)}">\n'
            f"<defs>{''.join(self.defs)}</defs>\n" + "\n".join(self.body) + "\n</svg>\n"
        )


def _toward(a, b, dist):
    """Point `dist` from a toward b (capped at half the segment; negative goes away)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    dist = min(dist, d / 2)
    return a[0] + dx / d * dist, a[1] + dy / d * dist


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _shift(p, n, k):
    return p[0] + n[0] * k, p[1] + n[1] * k


def _intersect(p, d, q, e):
    """Intersection of lines p + t*d and q + s*e (q when parallel)."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return q
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p[0] + d[0] * t, p[1] + d[1] * t


def _rounded(points, r):
    """Path commands from points[0] (already current) through points[1:],
    with interior corners rounded by quadratic arcs of radius r."""
    cmds = []
    for i in range(1, len(points) - 1):
        a = _toward(points[i], points[i - 1], r)
        c = _toward(points[i], points[i + 1], r)
        cmds.append(f"L{num(a[0])},{num(a[1])} Q{num(points[i][0])},{num(points[i][1])} "
                    f"{num(c[0])},{num(c[1])}")
    cmds.append(f"L{num(points[-1][0])},{num(points[-1][1])}")
    return " ".join(cmds)


def _arrow_head(prev, end, size, sw, kind):
    """draw.io mxMarker arrow: returns the shortened line end and the head polygon."""
    dx, dy = end[0] - prev[0], end[1] - prev[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    ox, oy = ux * sw * 1.118, uy * sw * 1.118
    vx, vy = ux * (size + sw), uy * (size + sw)
    tip = (end[0] - ox, end[1] - oy)
    notch = 3 / 4 if kind == "classic" else 1
    line_end = (end[0] - vx * notch - ox, end[1] - vy * notch - oy)
    head = [tip, (tip[0] - vx - vy / 2, tip[1] - vy + vx / 2)]
    if kind == "classic":
        head.append((tip[0] - vx * 3 / 4, tip[1] - vy * 3 / 4))
    head.append((tip[0] + vy / 2 - vx, tip[1] - vy - vx / 2))
    return line_end, head


def _map_path(d: str, gx, gy, sx, sy) -> str:
    """Scale an absolute-command path by (sx, sy) and move it to (gx, gy)."""
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    out: list[str] = []
    i, cmd = 0, "M"
    arity = {"M": 2, "L": 2, "T": 2, "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            if cmd not in arity:
                raise ValueError(f"unsupported path command {cmd!r}")
            out.append(cmd)
            i += 1
            if cmd == "Z":
                continue
        vals = [float(v) for v in tokens[i:i + arity[cmd]]]
        i += arity[cmd]
        if cmd == "A":
            rx, ry, rot, laf, sf, x, y = vals
            out.append(f"{num(rx * sx)} {num(ry * sy)} {num(rot)} {int(laf)} {int(sf)} "
                       f"{num(gx + x * sx)},{num(gy + y * sy)}")
        else:
            out.append(" ".join(f"{num(gx + vals[k] * sx)},{num(gy + vals[k + 1] * sy)}"
                                for k in range(0, len(vals), 2)))
    return " ".join(out)


def to_pdf(svg_path: Path, pdf_path: Path) -> None:
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)]
    elif shutil.which("inkscape"):
        cmd = ["inkscape", str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"]
    else:
        sys.exit("PDF export needs rsvg-convert or inkscape on PATH")
    subprocess.run(cmd, check=True)


def export(svg: str, out_dir: Path, name: str) -> None:
    """Write <name>.svg and <name>.pdf into out_dir."""
    svg_path, pdf_path = out_dir / f"{name}.svg", out_dir / f"{name}.pdf"
    svg_path.write_text(svg, encoding="utf-8")
    to_pdf(svg_path, pdf_path)
    print(f"wrote {svg_path} and {pdf_path}")
