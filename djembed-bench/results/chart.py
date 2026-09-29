"""Draws Djembed's requests per second against the other servers', from the results files next to this script.

    python3 djembed-bench/results/chart.py

Reads results files named <date>-<gpu>-<competitor>-<fp16|fp32>.md (the latest of each) and writes throughput-light.svg
and -dark.svg: a grid of small multiples, one row per precision, one column per workload, each panel with its own y
axis from zero, since the workloads differ by two orders of magnitude. Every competitor measured at a precision is a
line in that row; Djembed's line comes from the first competitor's file in COMPETITORS order. Standard library only.
"""
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
COMPETITORS = {"tei": "TEI", "infinity": "Infinity"}
PRECISIONS = [("fp16", "fp16"), ("strict fp32", "fp32")]
WORKLOADS = ["query", "ingest", "rerank"]
CONCURRENCY = [1, 8, 32, 128]

# Emphasis form: Djembed in the accent hue, the competitors in quieter hues. Each has its own marker shape.
THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "baseline": "#c3c2b7", "djembed": "#2a78d6", "tei": "#898781", "infinity": "#c98a2e"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "baseline": "#383835", "djembed": "#3987e5", "tei": "#898781", "infinity": "#d99a3e"},
}

WIDTH = 960
PANEL_W, PANEL_H = 200, 150       # plot area of one panel
COL_X = [104, 404, 704]           # left edge of each column's plot area
ROW_Y = [120, 356]                # top edge of each row's plot area
BOTTOM = 54                       # below the last row: tick labels and the axis title
SHAPES = {"djembed": "circle", "tei": "square", "infinity": "diamond"}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def results(path, competitor):
    """{workload: {"djembed" | competitor: [req/s per concurrency]}} from a results file's per-workload tables."""
    text = path.read_text()
    data = {}
    for workload in WORKLOADS:
        section = text.split(f"## {workload} (")[1].split("\n## ")[0]
        rows = re.findall(rf"\| (djembed|{competitor}) \| (\d+) \| ([\d.]+) \|", section)
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
    if shape == "diamond":
        return (f'<path d="M{cx:.1f} {cy - 5.5:.1f} L{cx + 5.5:.1f} {cy:.1f} L{cx:.1f} {cy + 5.5:.1f} '
                f'L{cx - 5.5:.1f} {cy:.1f} Z" fill="{color}" {ring}/>')
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

    for target in sorted(series, key=lambda name: name == "djembed"):  # Djembed drawn last, on top
        shape = SHAPES[target]
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series[target]))
        out.append(f'<polyline points="{points}" fill="none" stroke="{t[target]}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(series[target]):
            out.append(marker(shape, x(i), y(v), t[target], t["surface"]))


def draw(theme, rows):
    """rows: [(precision label, {competitor: results path})], one grid row each."""
    t = THEMES[theme]
    present = [c for c in COMPETITORS if any(c in files for _, files in rows)]
    names = " and ".join([", ".join(["Djembed"] + [COMPETITORS[c] for c in present[:-1]]), COMPETITORS[present[-1]]])
    height = ROW_Y[len(rows) - 1] + PANEL_H + BOTTOM
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
           f'role="img" aria-labelledby="title desc">',
           f'<title id="title">Requests per second, {names}</title>',
           f'<desc id="desc">Requests per second of {names} for query, ingest and rerank workloads at 1, 8, 32 and '
           f'128 concurrent clients, on an RTX 3090 Ti, by precision.</desc>',
           f'<rect width="{WIDTH}" height="{height}" rx="8" fill="{t["surface"]}"/>',
           f'<text x="32" y="36" {FONT} font-size="17" font-weight="600" fill="{t["primary"]}">'
           f'Requests per second, {names}</text>',
           f'<text x="32" y="58" {FONT} font-size="13" fill="{t["secondary"]}">'
           'RTX 3090 Ti · bge-m3, bge-reranker-v2-m3 · by concurrent clients</text>']

    lx = 680
    for target, label in [("djembed", "Djembed")] + [(c, COMPETITORS[c]) for c in present]:
        out.append(marker(SHAPES[target], lx, 31, t[target], t["surface"]))
        out.append(f'<text x="{lx + 11}" y="36" {FONT} font-size="13" fill="{t["secondary"]}">{label}</text>')
        lx += 96

    for row, (precision, files) in enumerate(rows):
        top = ROW_Y[row]
        out.append(f'<text x="32" y="{top + PANEL_H / 2:.1f}" {FONT} font-size="13" font-weight="600" '
                   f'fill="{t["primary"]}" transform="rotate(-90 32 {top + PANEL_H / 2:.1f})" text-anchor="middle">'
                   f'{precision}</text>')
        data = {c: results(path, c) for c, path in files.items()}
        for col, workload in enumerate(WORKLOADS):
            series = {"djembed": next(iter(data.values()))[workload]["djembed"]}
            series.update({c: data[c][workload][c] for c in data})
            panel(out, t, COL_X[col], top, series, f"{workload}, req/s")

    out.append(f'<text x="{WIDTH / 2:.0f}" y="{height - 14}" {FONT} font-size="12" text-anchor="middle" '
               f'fill="{t["muted"]}">concurrent clients</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def latest(competitor, precision):
    """The most recent results file for a competitor and precision; names start with an ISO date, so sort by name."""
    matches = sorted(HERE.glob(f"*-{competitor}-{precision}.md"))
    return matches[-1] if matches else None


if __name__ == "__main__":
    rows = []
    for label, precision in PRECISIONS:
        files = {c: latest(c, precision) for c in COMPETITORS}
        files = {c: path for c, path in files.items() if path is not None}
        if files:
            rows.append((label, files))
    for theme in THEMES:
        out = HERE / f"throughput-{theme}.svg"
        out.write_text(draw(theme, rows))
        print("wrote", out.name, "from", ", ".join(path.name for _, files in rows for path in files.values()))
