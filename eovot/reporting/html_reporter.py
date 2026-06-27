"""Self-contained HTML report generator for EOVOT benchmark results.

Generates a single, shareable ``.html`` file from one or more
:class:`~eovot.benchmark.engine.BenchmarkResult` objects.  The report
includes interactive charts (success curves, FPS comparison, efficiency–
accuracy scatter) and a sortable per-sequence breakdown table — all driven
by `Chart.js <https://www.chartjs.org/>`_ loaded from CDN.

No matplotlib or other visualisation library is required.

Usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.reporting.html_reporter import HTMLReporter

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=5)
    results = [engine.run(t, dataset) for t in [MOSSETracker(), KCFTracker()]]

    reporter = HTMLReporter(output_dir="results/")
    path = reporter.save(results, name="comparison")
    print(f"Report written to {path}")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Chart.js version pinned for reproducibility
_CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"

# Colour-blind-friendly palette (same as BenchmarkVisualizer)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HTMLReporter:
    """Generate a self-contained HTML benchmark report.

    The output is a single ``.html`` file that can be opened in any modern
    browser or shared with collaborators — no server or extra files needed.
    Charts require an internet connection to load Chart.js from CDN on first
    view; subsequent views work offline because browsers cache CDN resources.

    Args:
        output_dir: Directory where HTML files are written.  Created
            automatically if it does not exist.  Default: ``"results/"``.
        title: Page title shown in the browser tab and the report header.
            Default: ``"EOVOT Benchmark Report"``.
        max_sequences_in_table: Cap on per-sequence rows in the breakdown
            table to keep very long runs readable.  Sequences are sorted by
            mean IoU descending.  Default: ``50``.
    """

    def __init__(
        self,
        output_dir: str = "results/",
        title: str = "EOVOT Benchmark Report",
        max_sequences_in_table: int = 50,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.max_sequences_in_table = max_sequences_in_table

    def save(
        self,
        results: List["BenchmarkResult"],
        name: str = "report",
    ) -> Path:
        """Render and write the HTML report to disk.

        Args:
            results: One or more :class:`BenchmarkResult` objects.  Each
                represents one tracker evaluated on a dataset.
            name: Base filename without extension.

        Returns:
            :class:`pathlib.Path` of the written ``.html`` file.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        html = self._render(results)
        path = self.output_dir / f"{name}.html"
        path.write_text(html, encoding="utf-8")
        return path

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def _render(self, results: List["BenchmarkResult"]) -> str:
        dataset_name = results[0].dataset_name
        chart_data = self._build_chart_data(results)
        seq_rows = self._build_sequence_rows(results)
        summary_rows = self._build_summary_rows(results)

        return _HTML_TEMPLATE.format(
            title=self.title,
            chartjs_cdn=_CHARTJS_CDN,
            dataset_name=_esc(dataset_name),
            report_title=_esc(self.title),
            summary_rows=summary_rows,
            seq_rows=seq_rows,
            chart_data_json=json.dumps(chart_data, indent=2),
        )

    # ------------------------------------------------------------------ #
    # Data builders                                                        #
    # ------------------------------------------------------------------ #

    def _build_chart_data(self, results: List["BenchmarkResult"]) -> Dict[str, Any]:
        """Serialise all chart datasets into a single JSON blob."""
        iou_thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 51).tolist()]
        dist_thresholds = [round(t, 1) for t in np.linspace(0.0, 50.0, 51).tolist()]

        success_datasets: List[Dict] = []
        precision_datasets: List[Dict] = []
        fps_labels: List[str] = []
        fps_values: List[float] = []
        fps_colors: List[str] = []
        scatter_points: List[Dict] = []
        memory_labels: List[str] = []
        memory_values: List[float] = []

        for i, result in enumerate(results):
            color = _PALETTE[i % len(_PALETTE)]
            name = result.tracker_name

            # Success curve
            all_ious = np.concatenate([sr.ious for sr in result.sequence_results])
            success_rates = [
                round(float((all_ious >= t).mean()), 4) for t in np.linspace(0.0, 1.0, 51)
            ]
            sauc = result.mean_success_auc
            label = f"{name} [AUC={sauc:.3f}]" if sauc is not None else name
            success_datasets.append({
                "label": label,
                "data": success_rates,
                "borderColor": color,
                "backgroundColor": color + "22",
                "tension": 0.2,
                "pointRadius": 0,
                "borderWidth": 2,
            })

            # Precision curve
            has_dists = all(
                sr.center_distances is not None for sr in result.sequence_results
            )
            if has_dists:
                all_dists = np.concatenate(
                    [sr.center_distances for sr in result.sequence_results]
                )
            else:
                all_dists = (1.0 - all_ious) * 50.0

            prec_rates = [
                round(float((all_dists < t).mean()), 4) for t in np.linspace(0.0, 50.0, 51)
            ]
            precision_datasets.append({
                "label": name,
                "data": prec_rates,
                "borderColor": color,
                "backgroundColor": color + "22",
                "tension": 0.2,
                "pointRadius": 0,
                "borderWidth": 2,
            })

            # FPS bar
            fps_labels.append(name)
            fps_values.append(round(result.mean_fps, 1))
            fps_colors.append(color)

            # Efficiency scatter
            scatter_points.append({
                "x": round(result.mean_fps, 2),
                "y": round(result.mean_iou, 4),
                "r": max(6, min(20, result.peak_memory_mb / 50)),
                "label": name,
                "color": color,
            })

            # Memory bar
            memory_labels.append(name)
            memory_values.append(round(result.peak_memory_mb, 1))

        return {
            "iou_thresholds": iou_thresholds,
            "dist_thresholds": dist_thresholds,
            "success": success_datasets,
            "precision": precision_datasets,
            "fps": {
                "labels": fps_labels,
                "values": fps_values,
                "colors": fps_colors,
            },
            "scatter": scatter_points,
            "memory": {
                "labels": memory_labels,
                "values": memory_values,
                "colors": fps_colors,
            },
        }

    def _build_summary_rows(self, results: List["BenchmarkResult"]) -> str:
        rows: List[str] = []
        for result in results:
            s = result.summary()
            iou = s.get("mean_iou", 0.0)
            fps = s.get("mean_fps", 0.0)
            mem = s.get("peak_memory_mb", 0.0)
            sauc = s.get("success_auc", "—")
            pauc = s.get("precision_auc", "—")
            energy = s.get("total_energy_j", None)
            energy_str = f"{energy:.3f}" if energy is not None else "—"
            sauc_str = f"{sauc:.4f}" if isinstance(sauc, float) else sauc
            pauc_str = f"{pauc:.4f}" if isinstance(pauc, float) else pauc
            rows.append(
                f"<tr>"
                f"<td><strong>{_esc(result.tracker_name)}</strong></td>"
                f"<td>{_esc(result.dataset_name)}</td>"
                f"<td>{iou:.4f}</td>"
                f"<td>{sauc_str}</td>"
                f"<td>{pauc_str}</td>"
                f"<td>{fps:.1f}</td>"
                f"<td>{mem:.1f}</td>"
                f"<td>{energy_str}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _build_sequence_rows(self, results: List["BenchmarkResult"]) -> str:
        rows: List[str] = []
        for result in results:
            seq_results = sorted(
                result.sequence_results, key=lambda s: s.mean_iou, reverse=True
            )[: self.max_sequences_in_table]
            for sr in seq_results:
                iou_class = (
                    "iou-high" if sr.mean_iou >= 0.5
                    else "iou-mid" if sr.mean_iou >= 0.3
                    else "iou-low"
                )
                fps_val = round(sr.profiling.fps, 1)
                lat_val = round(sr.profiling.latency_mean_ms, 2)
                lat_p95 = round(sr.profiling.latency_p95_ms, 2)
                mem_val = round(sr.profiling.peak_memory_mb, 1)
                sauc_val = (
                    f"{sr.accuracy_metrics.success_auc:.4f}"
                    if sr.accuracy_metrics else "—"
                )
                energy_val = (
                    f"{sr.energy.energy_per_frame_mj:.3f}"
                    if sr.energy else "—"
                )
                rows.append(
                    f"<tr>"
                    f"<td>{_esc(result.tracker_name)}</td>"
                    f"<td>{_esc(sr.sequence_name)}</td>"
                    f"<td class='{iou_class}'>{sr.mean_iou:.4f}</td>"
                    f"<td>{sauc_val}</td>"
                    f"<td>{fps_val}</td>"
                    f"<td>{lat_val}</td>"
                    f"<td>{lat_p95}</td>"
                    f"<td>{mem_val}</td>"
                    f"<td>{energy_val}</td>"
                    f"</tr>"
                )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """Minimal HTML escaping for untrusted strings inserted into HTML."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="{chartjs_cdn}"></script>
<style>
  /* ---- Reset & base ---- */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    line-height: 1.6;
    padding: 0 1rem 3rem;
  }}
  a {{ color: #58a6ff; }}

  /* ---- Layout ---- */
  .container {{ max-width: 1200px; margin: 0 auto; }}

  /* ---- Header ---- */
  header {{
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #30363d;
    margin-bottom: 2rem;
  }}
  header h1 {{ font-size: 1.8rem; color: #58a6ff; margin-bottom: 0.25rem; }}
  header p  {{ color: #8b949e; font-size: 0.9rem; }}

  /* ---- Section titles ---- */
  h2 {{
    font-size: 1.15rem;
    color: #c9d1d9;
    margin: 2.5rem 0 0.75rem;
    border-left: 3px solid #58a6ff;
    padding-left: 0.6rem;
  }}

  /* ---- Tables ---- */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }}
  th, td {{
    padding: 0.55rem 0.75rem;
    text-align: right;
    border-bottom: 1px solid #21262d;
    white-space: nowrap;
  }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; cursor: pointer; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  tr:hover td {{ background: #1c2028; }}
  .iou-high {{ color: #3fb950; }}
  .iou-mid  {{ color: #f0883e; }}
  .iou-low  {{ color: #f85149; }}
  caption {{ caption-side: bottom; padding: 0.4rem; font-size: 0.75rem; color: #6e7681; }}

  /* ---- Charts ---- */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}
  .chart-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem 0.8rem;
  }}
  .chart-card h3 {{
    font-size: 0.9rem;
    color: #8b949e;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .chart-wrap {{ position: relative; height: 300px; }}

  /* ---- Sortable table JS ---- */
  th.sorted-asc::after  {{ content: " ↑"; color: #58a6ff; }}
  th.sorted-desc::after {{ content: " ↓"; color: #58a6ff; }}

  /* ---- Footer ---- */
  footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #21262d;
    font-size: 0.78rem;
    color: #484f58;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>{report_title}</h1>
  <p>Dataset: <strong>{dataset_name}</strong> &nbsp;·&nbsp;
     Generated by <strong>EOVOT Benchmark Suite</strong></p>
</header>

<!-- ================================================================ -->
<!-- SUMMARY TABLE                                                     -->
<!-- ================================================================ -->
<h2>Summary</h2>
<div class="table-wrap">
<table id="summary-table">
  <thead>
    <tr>
      <th>Tracker</th><th>Dataset</th>
      <th>mIoU</th><th>Success AUC</th><th>Precision AUC</th>
      <th>FPS</th><th>Mem (MB)</th><th>Energy (J)</th>
    </tr>
  </thead>
  <tbody>
{summary_rows}
  </tbody>
</table>
</div>

<!-- ================================================================ -->
<!-- CHARTS                                                            -->
<!-- ================================================================ -->
<h2>Charts</h2>
<div class="charts-grid">

  <div class="chart-card">
    <h3>Success Curves (IoU threshold sweep)</h3>
    <div class="chart-wrap"><canvas id="successChart"></canvas></div>
  </div>

  <div class="chart-card">
    <h3>Precision Curves (centre-distance threshold sweep)</h3>
    <div class="chart-wrap"><canvas id="precisionChart"></canvas></div>
  </div>

  <div class="chart-card">
    <h3>Throughput — Mean FPS</h3>
    <div class="chart-wrap"><canvas id="fpsChart"></canvas></div>
  </div>

  <div class="chart-card">
    <h3>Efficiency–Accuracy Trade-off (bubble area ∝ memory)</h3>
    <div class="chart-wrap"><canvas id="scatterChart"></canvas></div>
  </div>

  <div class="chart-card">
    <h3>Peak Memory Usage (MB)</h3>
    <div class="chart-wrap"><canvas id="memChart"></canvas></div>
  </div>

</div>

<!-- ================================================================ -->
<!-- PER-SEQUENCE TABLE                                                -->
<!-- ================================================================ -->
<h2>Per-Sequence Breakdown</h2>
<div class="table-wrap">
<table id="seq-table">
  <thead>
    <tr>
      <th>Tracker</th><th>Sequence</th>
      <th>mIoU</th><th>Success AUC</th>
      <th>FPS</th><th>Latency (ms)</th><th>p95 Lat (ms)</th>
      <th>Mem (MB)</th><th>E/frame (mJ)</th>
    </tr>
  </thead>
  <tbody>
{seq_rows}
  </tbody>
</table>
</div>

<footer>
  EOVOT — Edge-Optimized Visual Object Tracking Benchmark Suite &nbsp;·&nbsp;
  Charts powered by <a href="https://www.chartjs.org/" target="_blank">Chart.js</a>
</footer>

</div><!-- /container -->

<!-- ================================================================ -->
<!-- JAVASCRIPT                                                        -->
<!-- ================================================================ -->
<script>
const DATA = {chart_data_json};

const CHARTJS_DEFAULTS = {{
  plugins: {{
    legend: {{ labels: {{ color: "#c9d1d9", font: {{ size: 11 }} }} }},
  }},
  scales: {{
    x: {{ ticks: {{ color: "#8b949e" }}, grid: {{ color: "#21262d" }} }},
    y: {{ ticks: {{ color: "#8b949e" }}, grid: {{ color: "#21262d" }} }},
  }},
}};

// ---- Success curves ----
new Chart(document.getElementById("successChart"), {{
  type: "line",
  data: {{
    labels: DATA.iou_thresholds,
    datasets: DATA.success,
  }},
  options: {{
    ...CHARTJS_DEFAULTS,
    plugins: {{
      ...CHARTJS_DEFAULTS.plugins,
      title: {{ display: false }},
    }},
    scales: {{
      ...CHARTJS_DEFAULTS.scales,
      x: {{ ...CHARTJS_DEFAULTS.scales.x,
            title: {{ display: true, text: "IoU Threshold", color: "#8b949e" }} }},
      y: {{ ...CHARTJS_DEFAULTS.scales.y, min: 0, max: 1,
            title: {{ display: true, text: "Success Rate", color: "#8b949e" }} }},
    }},
    animation: false,
  }},
}});

// ---- Precision curves ----
new Chart(document.getElementById("precisionChart"), {{
  type: "line",
  data: {{
    labels: DATA.dist_thresholds,
    datasets: DATA.precision,
  }},
  options: {{
    ...CHARTJS_DEFAULTS,
    scales: {{
      ...CHARTJS_DEFAULTS.scales,
      x: {{ ...CHARTJS_DEFAULTS.scales.x,
            title: {{ display: true, text: "Centre-Distance Threshold (px)", color: "#8b949e" }} }},
      y: {{ ...CHARTJS_DEFAULTS.scales.y, min: 0, max: 1,
            title: {{ display: true, text: "Precision Rate", color: "#8b949e" }} }},
    }},
    animation: false,
  }},
}});

// ---- FPS bar chart ----
new Chart(document.getElementById("fpsChart"), {{
  type: "bar",
  data: {{
    labels: DATA.fps.labels,
    datasets: [{{
      label: "Mean FPS",
      data: DATA.fps.values,
      backgroundColor: DATA.fps.colors,
      borderRadius: 4,
    }}],
  }},
  options: {{
    ...CHARTJS_DEFAULTS,
    indexAxis: "y",
    plugins: {{ ...CHARTJS_DEFAULTS.plugins, legend: {{ display: false }} }},
    scales: {{
      ...CHARTJS_DEFAULTS.scales,
      x: {{ ...CHARTJS_DEFAULTS.scales.x,
            title: {{ display: true, text: "Mean FPS", color: "#8b949e" }} }},
    }},
    animation: false,
  }},
}});

// ---- Efficiency–Accuracy scatter (bubble chart) ----
const scatterDatasets = DATA.scatter.map(pt => ({{
  label: pt.label,
  data: [{{ x: pt.x, y: pt.y, r: pt.r }}],
  backgroundColor: pt.color + "bb",
  borderColor: pt.color,
  borderWidth: 1.5,
}}));

new Chart(document.getElementById("scatterChart"), {{
  type: "bubble",
  data: {{ datasets: scatterDatasets }},
  options: {{
    ...CHARTJS_DEFAULTS,
    plugins: {{
      ...CHARTJS_DEFAULTS.plugins,
      tooltip: {{
        callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: FPS=${{ctx.parsed.x}} mIoU=${{ctx.parsed.y}}`,
        }},
      }},
    }},
    scales: {{
      ...CHARTJS_DEFAULTS.scales,
      x: {{ ...CHARTJS_DEFAULTS.scales.x,
            title: {{ display: true, text: "Mean FPS", color: "#8b949e" }} }},
      y: {{ ...CHARTJS_DEFAULTS.scales.y, min: 0, max: 1,
            title: {{ display: true, text: "Mean IoU", color: "#8b949e" }} }},
    }},
    animation: false,
  }},
}});

// ---- Memory bar chart ----
new Chart(document.getElementById("memChart"), {{
  type: "bar",
  data: {{
    labels: DATA.memory.labels,
    datasets: [{{
      label: "Peak Memory (MB)",
      data: DATA.memory.values,
      backgroundColor: DATA.memory.colors,
      borderRadius: 4,
    }}],
  }},
  options: {{
    ...CHARTJS_DEFAULTS,
    plugins: {{ ...CHARTJS_DEFAULTS.plugins, legend: {{ display: false }} }},
    scales: {{
      ...CHARTJS_DEFAULTS.scales,
      y: {{ ...CHARTJS_DEFAULTS.scales.y,
            title: {{ display: true, text: "Peak Memory (MB)", color: "#8b949e" }} }},
    }},
    animation: false,
  }},
}});

// ---- Sortable tables ----
function makeSortable(tableId) {{
  const table = document.getElementById(tableId);
  if (!table) return;
  const headers = table.querySelectorAll("thead th");
  let sortCol = -1, sortAsc = true;

  headers.forEach((th, col) => {{
    th.addEventListener("click", () => {{
      if (sortCol === col) sortAsc = !sortAsc; else sortAsc = true;
      sortCol = col;
      headers.forEach(h => h.classList.remove("sorted-asc", "sorted-desc"));
      th.classList.add(sortAsc ? "sorted-asc" : "sorted-desc");

      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      rows.sort((a, b) => {{
        const va = a.cells[col].textContent.trim();
        const vb = b.cells[col].textContent.trim();
        const na = parseFloat(va), nb = parseFloat(vb);
        const cmp = isNaN(na) || isNaN(nb) ? va.localeCompare(vb) : na - nb;
        return sortAsc ? cmp : -cmp;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}}

makeSortable("summary-table");
makeSortable("seq-table");
</script>
</body>
</html>
"""
