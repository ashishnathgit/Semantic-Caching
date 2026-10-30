"""Small SVG pictures shared by the Gradio demo and the notebook.

angle_fan_svg   the query vector, the closest cached vectors at their angles, and
                the hit cone of half-angle arccos(tau)
spectrum_svg    the singular values sigma_1 >= sigma_2 >= ...

Both return a self-contained <svg> string with its own white background, so they
read the same in light and dark notebook or Gradio themes.
"""

from __future__ import annotations

import html
import math

PAPER, INK, INK_SOFT, INK_FAINT, RULE = "#ffffff", "#1e2a3e", "#4d586a", "#7f8896", "#d8dce2"
QUERY, HIT, HIT_WASH = "#3a4fd8", "#2f6f5e", "#e3efe9"


def _angle(score: float) -> float:
    return math.degrees(math.acos(min(1.0, max(0.0, score))))


def angle_fan_svg(matches: list[dict], tau: float, exact: bool = False, width: int = 400) -> str:
    """``matches`` are dicts with "query" and "score" (as returned by the pipeline)."""
    W, H, ox, oy, R = 400, 262, 200, 228, 178

    def polar(deg: float, radius: float) -> tuple[float, float]:
        rad = math.radians(deg)
        return ox + radius * math.sin(rad), oy - radius * math.cos(rad)

    cone = _angle(tau)
    x1, y1 = polar(-cone, R)
    x2, y2 = polar(cone, R)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{width}" '
        f'role="img" aria-label="Angles between the query and the closest cached questions" '
        f'style="font-family: system-ui, sans-serif; max-width: 100%; height: auto">',
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="{PAPER}"/>',
        f'<path d="M {ox - R} {oy} A {R} {R} 0 0 1 {ox + R} {oy}" fill="none" stroke="{RULE}"/>',
        f'<line x1="{ox - R - 6}" y1="{oy}" x2="{ox + R + 6}" y2="{oy}" stroke="{RULE}"/>',
    ]
    for t in range(-90, 91, 10):
        a, b = polar(t, R), polar(t, R - (9 if t % 30 == 0 else 5))
        parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{INK_FAINT}"/>')
        if t % 30 == 0 and abs(t) < 90:
            lx, ly = polar(t, R - 20)
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10" fill="{INK_FAINT}" '
                         f'text-anchor="middle" dominant-baseline="central">{abs(t)}°</text>')
    parts.append(f'<path d="M {ox} {oy} L {x1:.1f} {y1:.1f} A {R} {R} 0 0 1 {x2:.1f} {y2:.1f} Z" '
                 f'fill="{HIT_WASH}" stroke="{HIT}"/>')
    lx, ly = polar(cone, R + 16)
    parts.append(f'<text x="{lx + 4:.1f}" y="{ly:.1f}" font-size="11.5" fill="{HIT}" '
                 f'dominant-baseline="central">{cone:.1f}°</text>')

    thetas = [_angle(m["score"]) * (1 if i % 2 == 0 else -1) for i, m in enumerate(matches)]
    radii = [R + 13.0] * len(thetas)
    for side in (1, -1):
        idx = sorted((i for i, t in enumerate(thetas) if (t >= 0) == (side == 1)), key=lambda i: abs(thetas[i]))
        step = 0
        for prev, cur in zip(idx, idx[1:]):
            step = step + 1 if abs(abs(thetas[cur]) - abs(thetas[prev])) < 8 else 0
            radii[cur] = R - 14 * step - 4 if step else R + 13
    for i, (m, theta) in enumerate(zip(matches, thetas)):
        inside = (exact and i == 0) or m["score"] >= tau
        color = HIT if inside else INK_SOFT
        ex, ey = polar(theta, R)
        tx, ty = polar(theta, radii[i])
        tip = html.escape(f'{m["query"]} (cos {m["score"]:.3f}, {abs(theta):.1f}°)')
        parts.append(f'<g><title>{tip}</title>'
                     f'<line x1="{ox}" y1="{oy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="2" '
                     f'stroke-linecap="round"/>'
                     f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.5" fill="{color}" stroke="{PAPER}" stroke-width="2"/>'
                     f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="12.5" font-style="italic" font-family="Georgia, serif" '
                     f'fill="{INK}" text-anchor="middle" dominant-baseline="central">c<tspan font-size="9" dy="3">{i + 1}</tspan></text></g>')
    parts += [
        f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - R - 4}" stroke="{QUERY}" stroke-width="3" stroke-linecap="round"/>',
        f'<path d="M {ox - 6} {oy - R + 8} L {ox} {oy - R - 6} L {ox + 6} {oy - R + 8}" fill="none" '
        f'stroke="{QUERY}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<text x="{ox + 10}" y="{oy - R - 12}" font-size="15" font-style="italic" font-family="Georgia, serif" '
        f'fill="{QUERY}">q</text>',
        f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{INK}"/>',
        "</svg>",
    ]
    return "".join(parts)


def spectrum_svg(singular_values, width: int = 420) -> str:
    values = [float(v) for v in singular_values]
    W, H, left, right, top, bottom = 360, 150, 34, 10, 10, 24
    if not values:
        return ""
    top_value = values[0] or 1.0
    n = len(values)

    def x(i: int) -> float:
        return left + i / max(1, n - 1) * (W - left - right)

    def y(v: float) -> float:
        return top + (1 - v / top_value) * (H - top - bottom)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{width}" role="img" '
             f'aria-label="Singular values" style="font-family: system-ui, sans-serif; max-width: 100%; height: auto">',
             f'<rect width="{W}" height="{H}" rx="8" fill="{PAPER}"/>']
    for frac in (0.0, 0.5, 1.0):
        v = frac * top_value
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{RULE}"/>')
        parts.append(f'<text x="{left - 6}" y="{y(v):.1f}" font-size="10" fill="{INK_FAINT}" text-anchor="end" '
                     f'dominant-baseline="central">{v:.1f}</text>')
    parts += [f'<text x="{left}" y="{H - 6}" font-size="10" fill="{INK_FAINT}">1</text>',
              f'<text x="{W - right}" y="{H - 6}" font-size="10" fill="{INK_FAINT}" text-anchor="end">{n}</text>',
              f'<polyline points="{points}" fill="none" stroke="{QUERY}" stroke-width="2" '
              f'stroke-linejoin="round" stroke-linecap="round"/>', "</svg>"]
    return "".join(parts)
