"""Self-contained HTML dashboard reporter for EOVOT benchmark results.

Generates a single ``.html`` file that embeds all charts as inline SVG and
all styles as inline CSS.  No external CDN, JavaScript framework, or
matplotlib dependency is required at runtime — the output can be opened
directly in any modern browser, shared as an email attachment, or committed
to a repository.

Charts generated
----------------
* **Leaderboard table** — colour-coded comparison of every tracker across
  all key metrics (mIoU, Success AUC, Precision AUC, FPS, Latency, Memory).
* **Success curves** — one line per tracker showing the fraction of frames
  with IoU above a swept threshold (0 → 1).  Aggregated across all sequences.
* **Efficiency scatter** — FPS (x-axis) vs. mean IoU (y-axis) scatter plot
  with named annotations per tracker.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.reporting.html_reporter import HTMLReporter

    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine(verbose=False)

    results = [
        engine.run(MOSSETracker(), dataset, "Synthetic"),
        engine.run(KCFTracker(),   dataset, "Synthetic"),
    ]

    reporter = HTMLReporter(output_dir="results/")
    path = reporter.save_html(results, name="dashboard")
    print(f"Report written to {path}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Palette — distinct colours for up to 8 trackers
# ---------------------------------------------------------------------------
_PALETTE = [
    "#2563eb",  # blue
    "#16a34a",  # green
    "#dc2626",  # red
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#65a30d",  # lime
]


class HTMLReporter:
    """Generate a self-contained HTML benchmark dashboard.

    Args:
        output_dir: Directory where the HTML file is written.  Created
            automatically if it does not exist.  Default: ``"results/"``.
    """

    def __init__(self, output_dir: str = "results/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def save_html(
        self,
        results: List[BenchmarkResult],
        name: str = "dashboard",
    ) -> Path:
        """Build and write the HTML dashboard.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects, one per tracker / dataset combination.
            name: Output filename without extension.  Default: ``"dashboard"``.

        Returns:
            :class:`pathlib.Path` of the written HTML file.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        html = _build_html(results)
        path = self.output_dir / f"{name}.html"
        path.write_text(html, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def _build_html(results: List[BenchmarkResult]) -> str:
    dataset_name = results[0].dataset_name if results else "Unknown"
    leaderboard = _render_leaderboard(results)
    success_svg = _render_success_curves(results)
    scatter_svg = _render_efficiency_scatter(results)
    sequence_sections = _render_sequence_sections(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EOVOT Benchmark Dashboard — {_esc(dataset_name)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<header>
  <h1>EOVOT Benchmark Dashboard</h1>
  <p class="subtitle">Dataset: <strong>{_esc(dataset_name)}</strong> &nbsp;|&nbsp;
     {len(results)} tracker{"s" if len(results) != 1 else ""} evaluated</p>
</header>

<section>
  <h2>Leaderboard</h2>
  {leaderboard}
</section>

<section>
  <h2>Success Curves</h2>
  <p class="hint">Fraction of frames with IoU above threshold, aggregated across all sequences.</p>
  <div class="chart-wrap">{success_svg}</div>
</section>

<section>
  <h2>Efficiency Scatter (FPS vs. Mean IoU)</h2>
  <p class="hint">Top-right is optimal: high accuracy <em>and</em> high throughput.</p>
  <div class="chart-wrap">{scatter_svg}</div>
</section>

<section>
  <h2>Per-Sequence Breakdown</h2>
  {sequence_sections}
</section>

<footer>
  <p>Generated by <strong>EOVOT HTMLReporter</strong> &mdash; Edge-Optimized Visual Object Tracking Benchmark Suite</p>
</footer>

<script>
{_JS}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------

def _render_leaderboard(results: List[BenchmarkResult]) -> str:
    rows_html = ""
    for i, r in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        s = r.summary()
        miou = s.get("mean_iou", 0.0)
        fps = s.get("mean_fps", 0.0)
        mem = s.get("peak_memory_mb", 0.0)
        sauc = s.get("success_auc", "—")
        pauc = s.get("precision_auc", "—")
        lat = s.get("mean_latency_ms", "—")
        energy = s.get("mean_energy_per_frame_mj", "—")

        sauc_fmt = f"{sauc:.4f}" if isinstance(sauc, float) else sauc
        pauc_fmt = f"{pauc:.4f}" if isinstance(pauc, float) else pauc
        lat_fmt = f"{lat:.2f}" if isinstance(lat, float) else lat
        energy_fmt = f"{energy:.3f}" if isinstance(energy, float) else energy

        rows_html += f"""
    <tr>
      <td><span class="tracker-dot" style="background:{colour}"></span>{_esc(r.tracker_name)}</td>
      <td>{_bar_cell(miou, 0, 1, colour)}</td>
      <td>{sauc_fmt}</td>
      <td>{pauc_fmt}</td>
      <td class="num">{fps:.1f}</td>
      <td class="num">{lat_fmt}</td>
      <td class="num">{mem:.1f}</td>
      <td class="num">{energy_fmt}</td>
    </tr>"""

    return f"""<table class="leaderboard">
  <thead>
    <tr>
      <th>Tracker</th>
      <th>mIoU</th>
      <th>Success AUC</th>
      <th>Precision AUC</th>
      <th>FPS</th>
      <th>Latency (ms)</th>
      <th>Memory (MB)</th>
      <th>Energy (mJ/fr)</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>"""


def _bar_cell(value: float, lo: float, hi: float, colour: str) -> str:
    pct = max(0.0, min(100.0, 100.0 * (value - lo) / max(hi - lo, 1e-9)))
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar" style="width:{pct:.1f}%;background:{colour}"></div>'
        f'<span class="bar-label">{value:.4f}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Success curves SVG
# ---------------------------------------------------------------------------

def _render_success_curves(results: List[BenchmarkResult]) -> str:
    W, H = 600, 320
    PAD = {"l": 60, "r": 20, "t": 20, "b": 50}
    pw = W - PAD["l"] - PAD["r"]
    ph = H - PAD["t"] - PAD["b"]
    thresholds = np.linspace(0, 1, 41)

    # Grid lines
    grid = ""
    for y_val in np.linspace(0, 1, 6):
        y_px = PAD["t"] + ph * (1 - y_val)
        grid += f'<line x1="{PAD["l"]}" y1="{y_px:.1f}" x2="{PAD["l"]+pw}" y2="{y_px:.1f}" class="grid"/>'
        grid += f'<text x="{PAD["l"]-6}" y="{y_px+4:.1f}" class="axis-lbl" text-anchor="end">{y_val:.1f}</text>'
    for x_val in np.linspace(0, 1, 6):
        x_px = PAD["l"] + pw * x_val
        grid += f'<line x1="{x_px:.1f}" y1="{PAD["t"]}" x2="{x_px:.1f}" y2="{PAD["t"]+ph}" class="grid"/>'
        grid += f'<text x="{x_px:.1f}" y="{PAD["t"]+ph+16}" class="axis-lbl" text-anchor="middle">{x_val:.1f}</text>'

    # Curves
    curves = ""
    legend_items = ""
    for i, r in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        all_ious = np.concatenate([sr.ious for sr in r.sequence_results]) if r.sequence_results else np.array([])
        points = []
        for t in thresholds:
            rate = float(np.mean(all_ious >= t)) if len(all_ious) else 0.0
            x_px = PAD["l"] + pw * t
            y_px = PAD["t"] + ph * (1.0 - rate)
            points.append(f"{x_px:.1f},{y_px:.1f}")
        polyline = " ".join(points)
        rates = [float(np.mean(all_ious >= t)) for t in thresholds]
        auc = float(np.trapezoid(rates, thresholds)) if len(all_ious) else 0.0
        curves += f'<polyline points="{polyline}" stroke="{colour}" fill="none" stroke-width="2.5" stroke-linejoin="round"/>'
        ly = 28 + i * 22
        legend_items += (
            f'<rect x="{PAD["l"]+pw-110}" y="{ly-10}" width="14" height="3" fill="{colour}"/>'
            f'<text x="{PAD["l"]+pw-92}" y="{ly}" class="legend-lbl">{_esc(r.tracker_name)} (AUC={auc:.3f})</text>'
        )

    axes = (
        f'<text x="{PAD["l"]+pw//2}" y="{H-8}" class="axis-title" text-anchor="middle">IoU Threshold</text>'
        f'<text x="12" y="{PAD["t"]+ph//2}" class="axis-title" text-anchor="middle" '
        f'transform="rotate(-90,12,{PAD["t"]+ph//2})">Success Rate</text>'
    )

    return (
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
        f"{grid}{curves}{legend_items}{axes}"
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Efficiency scatter SVG
# ---------------------------------------------------------------------------

def _render_efficiency_scatter(results: List[BenchmarkResult]) -> str:
    W, H = 600, 320
    PAD = {"l": 65, "r": 30, "t": 20, "b": 50}
    pw = W - PAD["l"] - PAD["r"]
    ph = H - PAD["t"] - PAD["b"]

    fps_vals = [r.mean_fps for r in results]
    iou_vals = [r.mean_iou for r in results]
    fps_min, fps_max = min(fps_vals) * 0.9, max(fps_vals) * 1.1
    iou_min, iou_max = max(0.0, min(iou_vals) - 0.05), min(1.0, max(iou_vals) + 0.05)
    fps_range = max(fps_max - fps_min, 1.0)
    iou_range = max(iou_max - iou_min, 0.01)

    # Grid
    grid = ""
    for y_val in np.linspace(iou_min, iou_max, 5):
        y_px = PAD["t"] + ph * (1.0 - (y_val - iou_min) / iou_range)
        grid += f'<line x1="{PAD["l"]}" y1="{y_px:.1f}" x2="{PAD["l"]+pw}" y2="{y_px:.1f}" class="grid"/>'
        grid += f'<text x="{PAD["l"]-6}" y="{y_px+4:.1f}" class="axis-lbl" text-anchor="end">{y_val:.3f}</text>'
    for x_val in np.linspace(fps_min, fps_max, 5):
        x_px = PAD["l"] + pw * (x_val - fps_min) / fps_range
        grid += f'<line x1="{x_px:.1f}" y1="{PAD["t"]}" x2="{x_px:.1f}" y2="{PAD["t"]+ph}" class="grid"/>'
        grid += f'<text x="{x_px:.1f}" y="{PAD["t"]+ph+16}" class="axis-lbl" text-anchor="middle">{x_val:.0f}</text>'

    # Points
    dots = ""
    for i, r in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        x_px = PAD["l"] + pw * (r.mean_fps - fps_min) / fps_range
        y_px = PAD["t"] + ph * (1.0 - (r.mean_iou - iou_min) / iou_range)
        dots += f'<circle cx="{x_px:.1f}" cy="{y_px:.1f}" r="8" fill="{colour}" opacity="0.85"/>'
        dots += (
            f'<text x="{x_px:.1f}" y="{y_px-12:.1f}" class="dot-lbl" fill="{colour}" text-anchor="middle">'
            f'{_esc(r.tracker_name)}</text>'
        )

    axes = (
        f'<text x="{PAD["l"]+pw//2}" y="{H-8}" class="axis-title" text-anchor="middle">FPS (higher is better)</text>'
        f'<text x="12" y="{PAD["t"]+ph//2}" class="axis-title" text-anchor="middle" '
        f'transform="rotate(-90,12,{PAD["t"]+ph//2})">Mean IoU</text>'
    )

    return (
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
        f"{grid}{dots}{axes}"
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Per-sequence accordion sections
# ---------------------------------------------------------------------------

def _render_sequence_sections(results: List[BenchmarkResult]) -> str:
    out = ""
    for i, r in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        rows = ""
        for sr in r.sequence_results:
            sauc = f"{sr.success_auc:.4f}" if sr.success_auc is not None else "—"
            energy = (
                f"{sr.energy.energy_per_frame_mj:.3f}"
                if sr.energy is not None else "—"
            )
            rows += (
                f"<tr>"
                f"<td>{_esc(sr.sequence_name)}</td>"
                f"<td class='num'>{sr.mean_iou:.4f}</td>"
                f"<td class='num'>{sauc}</td>"
                f"<td class='num'>{sr.profiling.fps:.1f}</td>"
                f"<td class='num'>{sr.profiling.latency_mean_ms:.2f}</td>"
                f"<td class='num'>{sr.profiling.peak_memory_mb:.1f}</td>"
                f"<td class='num'>{energy}</td>"
                f"</tr>"
            )
        out += f"""
<details class="tracker-detail">
  <summary>
    <span class="tracker-dot" style="background:{colour}"></span>
    <strong>{_esc(r.tracker_name)}</strong>
    &nbsp;&mdash;&nbsp;{len(r.sequence_results)} sequences
  </summary>
  <table class="seq-table">
    <thead>
      <tr>
        <th>Sequence</th><th>mIoU</th><th>Success AUC</th>
        <th>FPS</th><th>Latency (ms)</th><th>Mem (MB)</th><th>Energy (mJ/fr)</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</details>"""
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML entity escaping."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Embedded CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; font-size: 14px;
       color: #1e293b; background: #f8fafc; line-height: 1.5; }
header { background: #0f172a; color: #f1f5f9; padding: 24px 40px; }
header h1 { font-size: 1.6rem; font-weight: 700; }
header .subtitle { color: #94a3b8; margin-top: 4px; }
section { padding: 32px 40px; border-bottom: 1px solid #e2e8f0; }
section h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 12px;
             color: #0f172a; }
.hint { color: #64748b; font-size: 13px; margin-bottom: 12px; }
footer { text-align: center; padding: 20px; color: #94a3b8; font-size: 12px; }

/* Leaderboard */
.leaderboard { width: 100%; border-collapse: collapse; margin-top: 8px; }
.leaderboard th { text-align: left; padding: 10px 12px; background: #1e293b;
                  color: #e2e8f0; font-weight: 600; font-size: 13px; }
.leaderboard td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0;
                  vertical-align: middle; }
.leaderboard tr:last-child td { border-bottom: none; }
.leaderboard tr:hover td { background: #f1f5f9; }
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Coloured dot */
.tracker-dot { display: inline-block; width: 10px; height: 10px;
               border-radius: 50%; margin-right: 7px; vertical-align: middle; }

/* Progress bar */
.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar { height: 10px; border-radius: 4px; min-width: 2px; }
.bar-label { font-size: 13px; white-space: nowrap; }

/* SVG charts */
.chart-wrap { overflow-x: auto; }
.chart-wrap svg { display: block; }
.grid { stroke: #e2e8f0; stroke-width: 1; }
.axis-lbl { font-size: 11px; fill: #64748b; font-family: system-ui, sans-serif; }
.axis-title { font-size: 12px; fill: #475569; font-family: system-ui, sans-serif;
              font-weight: 600; }
.legend-lbl { font-size: 11px; fill: #334155; font-family: system-ui, sans-serif;
              dominant-baseline: middle; }
.dot-lbl { font-size: 11px; font-family: system-ui, sans-serif; font-weight: 600; }

/* Accordion */
.tracker-detail { border: 1px solid #e2e8f0; border-radius: 8px;
                  margin-bottom: 10px; overflow: hidden; }
.tracker-detail summary { padding: 12px 16px; cursor: pointer;
                           list-style: none; background: #f8fafc;
                           display: flex; align-items: center; }
.tracker-detail summary::-webkit-details-marker { display: none; }
.tracker-detail[open] summary { background: #f1f5f9; border-bottom: 1px solid #e2e8f0; }
.seq-table { width: 100%; border-collapse: collapse; }
.seq-table th { padding: 8px 12px; background: #f1f5f9; font-size: 12px;
                color: #475569; text-align: left; font-weight: 600; }
.seq-table td { padding: 7px 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px; }
.seq-table tr:last-child td { border-bottom: none; }
"""

# ---------------------------------------------------------------------------
# Embedded JS (minimal: table sort)
# ---------------------------------------------------------------------------

_JS = """
// Make leaderboard table header cells sortable on click
document.querySelectorAll('.leaderboard th').forEach((th, col) => {
  th.style.cursor = 'pointer';
  th.title = 'Click to sort';
  th._asc = true;
  th.addEventListener('click', () => {
    const tbody = th.closest('table').querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const av = a.cells[col].innerText.trim();
      const bv = b.cells[col].innerText.trim();
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = isNaN(an) || isNaN(bn) ? av.localeCompare(bv) : an - bn;
      return th._asc ? cmp : -cmp;
    });
    th._asc = !th._asc;
    rows.forEach(r => tbody.appendChild(r));
  });
});
"""
