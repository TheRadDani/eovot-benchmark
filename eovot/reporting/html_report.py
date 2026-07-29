"""Self-contained HTML benchmark report generator for EOVOT.

Produces a single ``.html`` file from one or more
:class:`~eovot.benchmark.engine.BenchmarkResult` objects.  The file is
entirely self-contained — no external CSS, JS, or image assets — making it
ideal for sharing results via email or attaching to GitHub issues.

Embedded plots are generated as inline SVG from raw benchmark data, so the
report renders correctly even when ``matplotlib`` is not installed.

Generated sections:

1. **Header** — experiment title, generation date, dataset name.
2. **Summary table** — one row per tracker: mIoU, success AUC, precision AUC,
   FPS, peak memory, and (if available) mean energy per frame.
3. **Success curves** — SVG line chart, one curve per tracker.
4. **Efficiency scatter** — SVG scatter of FPS vs. mIoU; bubble area ∝ memory.
5. **Per-sequence details** — collapsible ``<details>`` block per tracker,
   containing a per-sequence IoU table sorted by mean IoU descending.

Usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.reporting.html_report import HTMLReportGenerator

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=10)
    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
    ]

    gen  = HTMLReportGenerator(output_dir="results/reports")
    path = gen.generate(results, filename="benchmark_report.html")
    print(f"Report saved to {path}")
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Colour palette — colour-blind-safe, works in both light and dark contexts.
_PALETTE = [
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#22c55e",  # green
    "#f97316",  # orange
    "#a855f7",  # purple
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#84cc16",  # lime
]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HTMLReportGenerator:
    """Generate self-contained HTML benchmark reports.

    Args:
        output_dir: Directory where HTML files are written.  Created
            automatically if it does not exist.  Default: ``"results/reports"``.
    """

    def __init__(self, output_dir: str = "results/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List["BenchmarkResult"],
        filename: str = "benchmark_report.html",
        title: str = "EOVOT Benchmark Report",
    ) -> Path:
        """Render a full benchmark report and write it to disk.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects (one per tracker).  Must contain at least one result.
            filename: Output filename relative to ``output_dir``.
            title: Report title shown in the page header and browser tab.

        Returns:
            :class:`pathlib.Path` of the written HTML file.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        dataset_name = results[0].dataset_name
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        body = "\n".join([
            _render_header(title, generated_at, dataset_name, len(results)),
            _render_summary_table(results),
            _render_success_curves(results),
            _render_efficiency_scatter(results),
            _render_per_sequence_details(results),
            _render_footer(),
        ])

        html_content = _PAGE_TEMPLATE.format(
            title=html.escape(title),
            body=body,
        )

        path = self.output_dir / filename
        path.write_text(html_content, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(
    title: str,
    generated_at: str,
    dataset_name: str,
    n_trackers: int,
) -> str:
    return (
        f'<header class="page-header">'
        f'<h1>{html.escape(title)}</h1>'
        f'<p class="meta">'
        f'Dataset: <strong>{html.escape(dataset_name)}</strong> &nbsp;·&nbsp; '
        f'{n_trackers} tracker{"s" if n_trackers != 1 else ""} &nbsp;·&nbsp; '
        f'Generated: {html.escape(generated_at)}'
        f'</p>'
        f'</header>'
    )


def _render_summary_table(results: List["BenchmarkResult"]) -> str:
    has_auc = any(r.mean_success_auc is not None for r in results)
    has_energy = any(r.mean_energy_per_frame_mj is not None for r in results)

    header_cols = ["Tracker", "Mean IoU"]
    if has_auc:
        header_cols += ["Success AUC", "Precision AUC"]
    header_cols += ["FPS", "Memory (MB)"]
    if has_energy:
        header_cols.append("mJ/frame")

    th = "".join(f"<th>{c}</th>" for c in header_cols)
    rows = []
    for i, r in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        dot = f'<span class="dot" style="background:{color}"></span>'
        cells = [
            f"<td>{dot}{html.escape(r.tracker_name)}</td>",
            f"<td>{r.mean_iou:.4f}</td>",
        ]
        if has_auc:
            sauc = r.mean_success_auc
            pauc = r.mean_precision_auc
            cells.append(f"<td>{sauc:.4f}</td>" if sauc is not None else "<td>—</td>")
            cells.append(f"<td>{pauc:.4f}</td>" if pauc is not None else "<td>—</td>")
        cells.append(f"<td>{r.mean_fps:.1f}</td>")
        cells.append(f"<td>{r.peak_memory_mb:.1f}</td>")
        if has_energy:
            e = r.mean_energy_per_frame_mj
            cells.append(f"<td>{e:.3f}</td>" if e is not None else "<td>—</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<section>'
        '<h2>Summary</h2>'
        '<div class="table-wrap">'
        '<table><thead><tr>' + th + '</tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody>'
        '</table>'
        '</div>'
        '</section>'
    )


def _render_success_curves(results: List["BenchmarkResult"]) -> str:
    svg = _success_curves_svg(results)
    return (
        '<section>'
        '<h2>Success Curves</h2>'
        '<p class="chart-note">Fraction of frames with IoU &gt; threshold. '
        'Area Under Curve (AUC) shown in legend.</p>'
        + svg +
        '</section>'
    )


def _render_efficiency_scatter(results: List["BenchmarkResult"]) -> str:
    svg = _efficiency_scatter_svg(results)
    return (
        '<section>'
        '<h2>Efficiency vs. Accuracy</h2>'
        '<p class="chart-note">X = mean FPS, Y = mean IoU. '
        'Bubble area ∝ peak memory footprint.</p>'
        + svg +
        '</section>'
    )


def _render_per_sequence_details(results: List["BenchmarkResult"]) -> str:
    blocks = []
    for i, r in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        sorted_seqs = sorted(r.sequence_results, key=lambda s: s.mean_iou, reverse=True)
        rows = []
        for sr in sorted_seqs:
            iou_cls = (
                "iou-good" if sr.mean_iou >= 0.5
                else "iou-mid" if sr.mean_iou >= 0.3
                else "iou-bad"
            )
            fps = sr.profiling.fps
            mem = sr.profiling.peak_memory_mb
            rows.append(
                f"<tr>"
                f'<td>{html.escape(sr.sequence_name)}</td>'
                f'<td class="{iou_cls}">{sr.mean_iou:.4f}</td>'
                f"<td>{fps:.1f}</td>"
                f"<td>{mem:.1f}</td>"
                f"</tr>"
            )

        table = (
            '<div class="table-wrap">'
            '<table>'
            '<thead><tr>'
            '<th>Sequence</th><th>Mean IoU</th><th>FPS</th><th>Memory (MB)</th>'
            '</tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody>'
            '</table>'
            '</div>'
        )
        tracker_label = html.escape(r.tracker_name)
        n = len(sorted_seqs)
        blocks.append(
            f'<details>'
            f'<summary style="border-left:4px solid {color};padding-left:.6em">'
            f'<strong>{tracker_label}</strong> — {n} sequence{"s" if n != 1 else ""}'
            f'</summary>'
            + table +
            f'</details>'
        )

    return (
        '<section>'
        '<h2>Per-Sequence Details</h2>'
        + "\n".join(blocks) +
        '</section>'
    )


def _render_footer() -> str:
    return (
        '<footer>'
        'Generated by <a href="https://github.com/theraddani/eovot-benchmark" '
        'rel="noopener">EOVOT Benchmark Suite</a>'
        '</footer>'
    )


# ---------------------------------------------------------------------------
# SVG chart generation
# ---------------------------------------------------------------------------

def _success_curves_svg(
    results: List["BenchmarkResult"],
    width: int = 680,
    height: int = 360,
) -> str:
    ML, MT, MR, MB = 58, 28, 28, 52
    pw = width - ML - MR
    ph = height - MT - MB

    thresholds = np.linspace(0.0, 1.0, 101)
    elems: List[str] = []

    # Background
    elems.append(
        f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" '
        f'fill="var(--chart-bg)" rx="4"/>'
    )

    # Gridlines (horizontal and vertical)
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gx = ML + v * pw
        gy = MT + (1.0 - v) * ph
        elems.append(
            f'<line x1="{gx:.1f}" y1="{MT}" x2="{gx:.1f}" y2="{MT+ph}" '
            f'stroke="var(--grid)" stroke-width="0.8" stroke-dasharray="4,4"/>'
        )
        elems.append(
            f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML+pw}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="0.8" stroke-dasharray="4,4"/>'
        )
        elems.append(
            f'<text x="{gx:.1f}" y="{MT+ph+16}" text-anchor="middle" '
            f'class="axis-label">{v:.1f}</text>'
        )
        elems.append(
            f'<text x="{ML-6}" y="{gy+4:.1f}" text-anchor="end" '
            f'class="axis-label">{v:.1f}</text>'
        )

    # Axes
    elems.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ph}" '
        f'stroke="var(--axis)" stroke-width="1.5"/>'
    )
    elems.append(
        f'<line x1="{ML}" y1="{MT+ph}" x2="{ML+pw}" y2="{MT+ph}" '
        f'stroke="var(--axis)" stroke-width="1.5"/>'
    )

    # Axis titles
    cx_label = ML + pw / 2
    elems.append(
        f'<text x="{cx_label:.1f}" y="{MT+ph+40}" text-anchor="middle" '
        f'class="axis-title">IoU Threshold</text>'
    )
    cy_label = MT + ph / 2
    elems.append(
        f'<text x="13" y="{cy_label:.1f}" text-anchor="middle" '
        f'class="axis-title" transform="rotate(-90 13 {cy_label:.1f})">Success Rate</text>'
    )

    # Legend position
    legend_x = ML + pw - 180
    legend_y = MT + 10

    # Curves + legend
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    for i, result in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        all_ious = np.concatenate([sr.ious for sr in result.sequence_results])
        rates = np.array([(all_ious > t).mean() for t in thresholds])
        auc = float(trapz(rates, thresholds))

        pts = " ".join(
            f"{ML + t * pw:.1f},{MT + (1.0 - r) * ph:.1f}"
            for t, r in zip(thresholds, rates)
        )
        elems.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round"/>'
        )

        # Legend entry
        ly = legend_y + i * 22
        elems.append(
            f'<line x1="{legend_x}" y1="{ly+6}" x2="{legend_x+18}" y2="{ly+6}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        label = html.escape(result.tracker_name)
        elems.append(
            f'<text x="{legend_x+22}" y="{ly+10}" class="legend-label">'
            f'{label} AUC={auc:.3f}</text>'
        )

    body = "\n".join(elems)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart-svg" role="img" '
        f'aria-label="Success curves">'
        f'{body}'
        f'</svg>'
    )


def _efficiency_scatter_svg(
    results: List["BenchmarkResult"],
    width: int = 680,
    height: int = 360,
) -> str:
    ML, MT, MR, MB = 62, 28, 40, 52
    pw = width - ML - MR
    ph = height - MT - MB

    fps_vals = [r.mean_fps for r in results]
    iou_vals = [r.mean_iou for r in results]
    mem_vals = [r.peak_memory_mb for r in results]

    fps_max = max(fps_vals) * 1.15 if fps_vals else 1.0
    fps_max = max(fps_max, 1.0)
    mem_max = max(mem_vals) if mem_vals else 1.0

    elems: List[str] = []

    # Background
    elems.append(
        f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" '
        f'fill="var(--chart-bg)" rx="4"/>'
    )

    # Horizontal gridlines at IoU 0.2, 0.4, 0.6, 0.8, 1.0
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = MT + (1.0 - v) * ph
        elems.append(
            f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML+pw}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="0.8" stroke-dasharray="4,4"/>'
        )
        elems.append(
            f'<text x="{ML-6}" y="{gy+4:.1f}" text-anchor="end" '
            f'class="axis-label">{v:.1f}</text>'
        )

    # Axes
    elems.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ph}" '
        f'stroke="var(--axis)" stroke-width="1.5"/>'
    )
    elems.append(
        f'<line x1="{ML}" y1="{MT+ph}" x2="{ML+pw}" y2="{MT+ph}" '
        f'stroke="var(--axis)" stroke-width="1.5"/>'
    )

    # Axis titles
    cx_label = ML + pw / 2
    elems.append(
        f'<text x="{cx_label:.1f}" y="{MT+ph+40}" text-anchor="middle" '
        f'class="axis-title">Mean FPS</text>'
    )
    cy_label = MT + ph / 2
    elems.append(
        f'<text x="13" y="{cy_label:.1f}" text-anchor="middle" '
        f'class="axis-title" transform="rotate(-90 13 {cy_label:.1f})">Mean IoU</text>'
    )

    # Bubbles (plotted back-to-front so smaller bubbles appear on top)
    indexed = sorted(enumerate(results), key=lambda x: -mem_vals[x[0]])
    for i, result in indexed:
        color = _PALETTE[i % len(_PALETTE)]
        fx = ML + (fps_vals[i] / fps_max) * pw
        fy = MT + (1.0 - max(0.0, min(1.0, iou_vals[i]))) * ph
        radius = max(10.0, math.sqrt(mem_vals[i] / max(mem_max, 1.0)) * 35.0)

        elems.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.65" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        label = html.escape(result.tracker_name)
        elems.append(
            f'<text x="{fx:.1f}" y="{fy - radius - 4:.1f}" '
            f'text-anchor="middle" class="bubble-label">{label}</text>'
        )

    # X-axis tick labels (FPS)
    n_ticks = 5
    for j in range(n_ticks + 1):
        val = fps_max * j / n_ticks
        gx = ML + (val / fps_max) * pw
        elems.append(
            f'<text x="{gx:.1f}" y="{MT+ph+16}" text-anchor="middle" '
            f'class="axis-label">{val:.0f}</text>'
        )

    # Legend note
    elems.append(
        f'<text x="{ML+pw}" y="{MT+ph+44}" text-anchor="end" '
        f'class="legend-label" fill="var(--muted)">bubble area ∝ memory</text>'
    )

    body = "\n".join(elems)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart-svg" role="img" '
        f'aria-label="Efficiency scatter plot">'
        f'{body}'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# HTML page template
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
/* ── Reset ────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  background: var(--bg);
  color: var(--fg);
  padding: 0 1rem 3rem;
  max-width: 900px;
  margin: 0 auto;
}}

/* ── CSS variables — light mode default ────────────────────── */
:root {{
  --bg:        #ffffff;
  --fg:        #111827;
  --subtle:    #6b7280;
  --muted:     #9ca3af;
  --border:    #e5e7eb;
  --accent:    #3b82f6;
  --good:      #16a34a;
  --mid:       #b45309;
  --bad:       #dc2626;
  --chart-bg:  #f9fafb;
  --grid:      #d1d5db;
  --axis:      #6b7280;
  --th-bg:     #f3f4f6;
  --row-alt:   #fafafa;
  --details-bg:#f9fafb;
}}

/* ── Dark mode ─────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:        #111827;
    --fg:        #f9fafb;
    --subtle:    #9ca3af;
    --muted:     #6b7280;
    --border:    #374151;
    --accent:    #60a5fa;
    --good:      #4ade80;
    --mid:       #fbbf24;
    --bad:       #f87171;
    --chart-bg:  #1f2937;
    --grid:      #374151;
    --axis:      #9ca3af;
    --th-bg:     #1f2937;
    --row-alt:   #1a2535;
    --details-bg:#1a2535;
  }}
}}

/* ── Layout ────────────────────────────────────────────────── */
.page-header {{
  padding: 2rem 0 1rem;
  border-bottom: 2px solid var(--accent);
  margin-bottom: 2rem;
}}
.page-header h1 {{ font-size: 1.8rem; color: var(--accent); }}
.meta {{ color: var(--subtle); margin-top: .4rem; font-size: .9rem; }}

section {{
  margin-top: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}}
section h2 {{
  font-size: 1.2rem;
  margin-bottom: .6rem;
  color: var(--fg);
}}
.chart-note {{ font-size: .85rem; color: var(--subtle); margin-bottom: .8rem; }}

/* ── Tables ────────────────────────────────────────────────── */
.table-wrap {{ overflow-x: auto; }}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: .9rem;
  min-width: 420px;
}}
th, td {{ padding: .45rem .75rem; border: 1px solid var(--border); text-align: right; }}
thead th {{ background: var(--th-bg); font-weight: 600; text-align: center; }}
td:first-child {{ text-align: left; white-space: nowrap; }}
th:first-child {{ text-align: left; }}
tr:nth-child(even) td {{ background: var(--row-alt); }}
.dot {{
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 50%;
  margin-right: .45rem;
  vertical-align: middle;
}}
.iou-good {{ color: var(--good); font-weight: 600; }}
.iou-mid  {{ color: var(--mid);  font-weight: 600; }}
.iou-bad  {{ color: var(--bad);  font-weight: 600; }}

/* ── SVG charts ────────────────────────────────────────────── */
.chart-svg {{ max-width: 100%; height: auto; display: block; }}
.axis-label {{ font-size: 11px; fill: var(--subtle); font-family: inherit; }}
.axis-title {{ font-size: 12px; fill: var(--subtle); font-family: inherit; font-weight: 500; }}
.legend-label {{ font-size: 11.5px; fill: var(--fg); font-family: inherit; }}
.bubble-label {{ font-size: 11px; fill: var(--fg); font-family: inherit; font-weight: 600; }}

/* ── Details / collapsible ─────────────────────────────────── */
details {{
  margin-top: .8rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--details-bg);
}}
details summary {{
  padding: .6rem .9rem;
  cursor: pointer;
  font-size: .95rem;
  list-style: none;
  user-select: none;
}}
details summary::-webkit-details-marker {{ display: none; }}
details summary::before {{
  content: "▶ ";
  font-size: .75rem;
  color: var(--subtle);
}}
details[open] summary::before {{ content: "▼ "; }}
details .table-wrap {{ padding: .5rem .9rem .9rem; }}

/* ── Footer ────────────────────────────────────────────────── */
footer {{
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: .82rem;
  color: var(--muted);
  text-align: center;
}}
footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
