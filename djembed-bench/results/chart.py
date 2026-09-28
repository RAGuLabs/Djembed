"""Draws Djembed's and TEI's requests per second from the results files next to this script, as a light and a dark SVG.

    python3 djembed-bench/results/chart.py

A grid of small multiples: one row per precision (results file in ROWS), one column per workload, each panel with its
own y axis from zero, since the workloads differ by two orders of magnitude. Writes throughput-light.svg and
throughput-dark.svg. Standard library only.
"""
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
ROWS = [("strict fp32", "2026-09-29-rtx3090ti-fp32.md"), ("fp16", "2026-09-29-rtx3090ti-fp16.md")]
WORKLOADS = ["query", "ingest", "rerank"]
CONCURRENCY = [1, 8, 32, 128]

# Emphasis form: Djembed in the accent hue, TEI as the gray reference. Each also has its own marker shape.
THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "baseline": "#c3c2b7", "djembed": "#2a78d6", "tei": "#898781"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "baseline": "#383835", "djembed": "#3987e5", "tei": "#898781"},
}

WIDTH, HEIGHT = 960, 560
PANEL_W, PANEL_H = 200, 150       # plot area of one panel; ~50px after it for the gap label
COL_X = [104, 404, 704]           # left edge of each column's plot area
ROW_Y = [120, 356]                # top edge of each row's plot area
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def results(path):
    """{workload: {target: [req/s per concurrency]}} from a results file's per-workload tables."""
    text = (HERE / path).read_text()
    data = {}
    for workload in WORKLOADS:
        section = text.split(f"## {workload} (")[1].split("\n## ")[0]
        rows = re.findall(r"\| (djembed|tei) \| (\d+) \| ([\d.]+) \|", section)
        by_target = {}
        for target, concurrency, rps in rows:
            by_target.setdefault(target, {})[int(concurrency)] = float(rps)
        data[workload] = {t: [by_target[t][c] for c in CONCURRENCY] for t in by_target}
    return data


def nice_ticks(maximum):
    """Round tick values from zero to just above maximum, about four intervals."""
    raw = maximum / 4
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    count = math.ceil(maximum / step)
    return [round(i * step, 6) for i in range(count + 1)]


def fmt(value):
    return f"{value:,.0f}" if value >= 100 else (f"{value:g}" if value == int(value) else f"{value:.1f}")


def marker(shape, cx, cy, color, surface):
    ring = f'stroke="{surface}" stroke-width="2"'
    if shape == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" {ring}/>'
    return f'<rect x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" rx="1.5" fill="{color}" {ring}/>'


def panel(out, t, left, top, series, title):
    ticks = nice_ticks(max(max(v) for v in series.values()))
    top_value = ticks[-1]
    y = lambda v: top + PANEL_H - v / top_value * PANEL_H
    x = lambda i: left + i * PANEL_W / (len(CONCURRENCY) - 1)

    out.append(f'<text x="{left}" y="{top - 14}" {FONT} font-size="13" font-weight="600" fill="{t["primary"]}">'
               f'{title}</text>')
    for tick in ticks:
        gy = y(tick)
        out.append(f'<line x1="{left - 6}" x2="{left + PANEL_W + 6}" y1="{gy:.1f}" y2="{gy:.1f}" '
                   f'stroke="{t["baseline"] if tick == 0 else t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{left - 12}" y="{gy + 4:.1f}" {FONT} font-size="11" text-anchor="end" '
                   f'fill="{t["muted"]}" style="font-variant-numeric: tabular-nums">{fmt(tick)}</text>')
    for i, c in enumerate(CONCURRENCY):
        out.append(f'<text x="{x(i):.1f}" y="{top + PANEL_H + 18}" {FONT} font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" style="font-variant-numeric: tabular-nums">{c}</text>')

    for target, shape in (("tei", "square"), ("djembed", "circle")):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series[target]))
        out.append(f'<polyline points="{points}" fill="none" stroke="{t[target]}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(series[target]):
            out.append(marker(shape, x(i), y(v), t[target], t["surface"]))

    # The gap at the highest concurrency, beside the two end points.
    djembed, tei = series["djembed"][-1], series["tei"][-1]
    gap = (djembed / tei - 1) * 100
    label_y = (y(djembed) + y(tei)) / 2 + 4
    out.append(f'<text x="{left + PANEL_W + 10}" y="{label_y:.1f}" {FONT} font-size="12" font-weight="600" '
               f'fill="{t["primary"]}" style="font-variant-numeric: tabular-nums">{gap:+.0f}%</text>')


def draw(theme):
    t = THEMES[theme]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
           f'role="img" aria-labelledby="title desc">',
           '<title id="title">Djembed and TEI requests per second</title>',
           '<desc id="desc">Requests per second of Djembed and TEI for query, ingest and rerank workloads at 1, 8, 32 and '
           '128 concurrent clients, strict fp32 and fp16, on an RTX 3090 Ti, with the gap at 128 clients.</desc>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="{t["surface"]}"/>',
           f'<text x="32" y="36" {FONT} font-size="17" font-weight="600" fill="{t["primary"]}">'
           'Requests per second, Djembed and TEI</text>',
           f'<text x="32" y="58" {FONT} font-size="13" fill="{t["secondary"]}">'
           'RTX 3090 Ti · bge-m3, bge-reranker-v2-m3 · by concurrent clients · gap at 128 clients</text>']

    lx = 740
    for target, name, shape in (("djembed", "Djembed", "circle"), ("tei", "TEI", "square")):
        out.append(marker(shape, lx, 31, t[target], t["surface"]))
        out.append(f'<text x="{lx + 11}" y="36" {FONT} font-size="13" fill="{t["secondary"]}">{name}</text>')
        lx += 96

    for row, (precision, path) in enumerate(ROWS):
        data = results(path)
        top = ROW_Y[row]
        out.append(f'<text x="32" y="{top + PANEL_H / 2:.1f}" {FONT} font-size="13" font-weight="600" '
                   f'fill="{t["primary"]}" transform="rotate(-90 32 {top + PANEL_H / 2:.1f})" text-anchor="middle">'
                   f'{precision}</text>')
        for col, workload in enumerate(WORKLOADS):
            panel(out, t, COL_X[col], top, data[workload], f"{workload}, req/s")

    out.append(f'<text x="{WIDTH / 2:.0f}" y="{HEIGHT - 14}" {FONT} font-size="12" text-anchor="middle" '
               f'fill="{t["muted"]}">concurrent clients</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for theme in THEMES:
        (HERE / f"throughput-{theme}.svg").write_text(draw(theme))
        print("wrote", HERE / f"throughput-{theme}.svg")
