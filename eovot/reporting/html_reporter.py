"""Self-contained interactive HTML report generator for EOVOT benchmarks.

Generates a single ``.html`` file that can be opened in any browser with
no internet connection or server required.  Charts are rendered using the
browser's built-in Canvas 2D API with pure JavaScript — no external
libraries are downloaded.

The report includes:

* **Summary table** — mIoU, Success AUC, FPS, latency, memory, energy
  (energy columns hidden automatically when no TDP was configured).
* **Success curves** — IoU-threshold sweep per tracker; AUC indicated.
* **Precision curves** — centre-distance sweep per tracker.
* **Efficiency scatter** — FPS vs. mean IoU with tracker labels.
* **Per-sequence breakdown** — collapsible accordion per tracker.

Typical usage::

    from eovot.reporting.html_reporter import HTMLReporter
    from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult

    engine  = BenchmarkEngine()
    results = [engine.run(tracker, dataset) for tracker in trackers]

    reporter = HTMLReporter(output_dir="results/")
    path = reporter.generate(results, name="comparison", title="My Experiment")
    print(f"Report written to {path}")

CLI shortcut via scripts/html_report.py::

    python scripts/html_report.py --demo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Colour palette for up to 8 trackers (WCAG-AA accessible on white).
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]


class HTMLReporter:
    """Generate a self-contained HTML benchmark report.

    Args:
        output_dir: Directory where ``.html`` files are written.
                    Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "results/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List["BenchmarkResult"],  # noqa: F821
        name: str = "report",
        title: str = "EOVOT Benchmark Report",
    ) -> Path:
        """Build and write a self-contained HTML report.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                     objects, one per tracker.
            name:    Base filename (without ``.html`` extension).
            title:   ``<title>`` and ``<h1>`` text for the page.

        Returns:
            :class:`pathlib.Path` of the written HTML file.
        """
        payload = self._build_payload(results)
        html = _render_html(title, payload)
        path = self.output_dir / f"{name}.html"
        path.write_text(html, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _build_payload(self, results: List[Any]) -> Dict:
        """Extract all chart and table data from BenchmarkResult objects."""
        trackers = [r.tracker_name for r in results]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(trackers))]

        # ── Summary table rows ──────────────────────────────────────────
        summary_rows = []
        has_energy = any(r.total_energy_j is not None for r in results)
        has_auc = any(r.mean_success_auc is not None for r in results)

        for r in results:
            row: Dict = {
                "tracker": r.tracker_name,
                "dataset": r.dataset_name,
                "mean_iou": round(r.mean_iou, 4),
                "mean_fps": round(r.mean_fps, 2),
                "peak_memory_mb": round(r.peak_memory_mb, 2),
            }
            if has_auc and r.mean_success_auc is not None:
                row["success_auc"] = round(r.mean_success_auc, 4)
                row["precision_auc"] = round(r.mean_precision_auc or 0.0, 4)
            if has_energy and r.total_energy_j is not None:
                row["total_energy_j"] = round(r.total_energy_j, 4)
                row["energy_per_frame_mj"] = round(r.mean_energy_per_frame_mj or 0.0, 4)
            summary_rows.append(row)

        # ── Success curves ───────────────────────────────────────────────
        iou_thresholds = np.linspace(0.0, 1.0, 101).tolist()
        success_curves = []
        for r in results:
            all_ious = np.concatenate([sr.ious for sr in r.sequence_results])
            rates = [(all_ious > t).mean() for t in iou_thresholds]
            success_curves.append({"label": r.tracker_name, "rates": rates})

        # ── Precision curves ─────────────────────────────────────────────
        dist_thresholds = np.linspace(0.0, 50.0, 51).tolist()
        precision_curves = []
        for r in results:
            seq_results_with_cd = [
                sr for sr in r.sequence_results
                if sr.center_distances is not None
            ]
            if seq_results_with_cd:
                all_dists = np.concatenate(
                    [sr.center_distances for sr in seq_results_with_cd]
                )
                rates = [(all_dists < t).mean() for t in dist_thresholds]
            else:
                rates = [0.0] * len(dist_thresholds)
            precision_curves.append({"label": r.tracker_name, "rates": rates})

        # ── Efficiency scatter (FPS vs mIoU) ────────────────────────────
        scatter_points = [
            {
                "label": r.tracker_name,
                "fps": round(r.mean_fps, 2),
                "mean_iou": round(r.mean_iou, 4),
            }
            for r in results
        ]

        # ── Per-sequence breakdown ───────────────────────────────────────
        per_tracker_sequences = []
        for r in results:
            seqs = []
            for sr in r.sequence_results:
                entry: Dict = {
                    "name": sr.sequence_name,
                    "mean_iou": round(sr.mean_iou, 4),
                    "fps": round(sr.profiling.fps, 2),
                    "latency_ms": round(sr.profiling.latency_mean_ms, 3),
                    "memory_mb": round(sr.profiling.peak_memory_mb, 2),
                }
                if sr.accuracy_metrics is not None:
                    entry["success_auc"] = round(sr.accuracy_metrics.success_auc, 4)
                seqs.append(entry)
            per_tracker_sequences.append({"tracker": r.tracker_name, "sequences": seqs})

        return {
            "trackers": trackers,
            "colors": colors,
            "summary_rows": summary_rows,
            "has_energy": has_energy,
            "has_auc": has_auc,
            "iou_thresholds": iou_thresholds,
            "success_curves": success_curves,
            "dist_thresholds": dist_thresholds,
            "precision_curves": precision_curves,
            "scatter_points": scatter_points,
            "per_tracker_sequences": per_tracker_sequences,
        }


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _render_html(title: str, payload: Dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="container">
  <h1>{_esc(title)}</h1>
  <p class="subtitle">Generated by <strong>EOVOT</strong> &mdash; Edge-Optimized Visual Object Tracking Benchmark Suite</p>

  <h2>Summary</h2>
  <div id="summary-table"></div>

  <h2>Success Curves</h2>
  <div class="chart-wrap"><canvas id="success-canvas" width="700" height="380"></canvas></div>

  <h2>Precision Curves</h2>
  <div class="chart-wrap"><canvas id="precision-canvas" width="700" height="380"></canvas></div>

  <h2>Efficiency: FPS vs. Accuracy</h2>
  <div class="chart-wrap"><canvas id="scatter-canvas" width="700" height="380"></canvas></div>

  <h2>Per-Sequence Breakdown</h2>
  <div id="sequences-accordion"></div>
</div>

<script>
const DATA = {data_json};
{_JS}
</script>
</body>
</html>"""


def _esc(s: str) -> str:
    """Minimal HTML escaping for text content."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Embedded CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f8f9fa; color: #212529; line-height: 1.6;
}
.container { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; color: #1a1a2e; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.75rem; color: #1a1a2e;
     border-bottom: 2px solid #dee2e6; padding-bottom: 0.4rem; }
.subtitle { color: #6c757d; font-size: 0.9rem; margin-bottom: 1.5rem; }
.chart-wrap { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
              padding: 1rem; display: inline-block; width: 100%; overflow-x: auto; }
canvas { display: block; max-width: 100%; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #dee2e6; border-radius: 6px; overflow: hidden;
        font-size: 0.88rem; }
thead tr { background: #1a1a2e; color: #fff; }
th, td { padding: 0.55rem 0.9rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tbody tr:nth-child(even) { background: #f8f9fa; }
tbody tr:hover { background: #e9ecef; }
.best { font-weight: 700; color: #198754; }
.accordion-btn {
  width: 100%; text-align: left; background: #fff; border: 1px solid #dee2e6;
  border-radius: 6px; padding: 0.65rem 1rem; font-size: 0.95rem; cursor: pointer;
  margin-bottom: 0.4rem; display: flex; justify-content: space-between;
  align-items: center; transition: background 0.15s;
}
.accordion-btn:hover { background: #e9ecef; }
.accordion-body { display: none; margin-bottom: 1rem; overflow-x: auto; }
.accordion-body.open { display: block; }
.arrow { font-size: 0.75rem; transition: transform 0.2s; }
.arrow.open { transform: rotate(90deg); }
"""

# ---------------------------------------------------------------------------
# Embedded JavaScript
# ---------------------------------------------------------------------------

_JS = r"""
// ── Colour helpers ──────────────────────────────────────────────────────────
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return {r,g,b};
}
function rgba(hex, a) {
  const {r,g,b} = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`;
}

// ── Generic line chart ───────────────────────────────────────────────────────
function drawLineChart(canvasId, xData, curves, colors, xLabel, yLabel, xRange, yRange) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {top:20, right:140, bottom:50, left:55};
  const pw = W - pad.left - pad.right, ph = H - pad.top - pad.bottom;

  ctx.clearRect(0,0,W,H);

  // Grid + axes
  ctx.strokeStyle = '#dee2e6'; ctx.lineWidth = 1;
  const yTicks = 5;
  for (let i=0; i<=yTicks; i++) {
    const y = pad.top + ph * (1 - i/yTicks);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left+pw, y); ctx.stroke();
    ctx.fillStyle='#6c757d'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText((yRange[0]+(yRange[1]-yRange[0])*i/yTicks).toFixed(1), pad.left-6, y+4);
  }
  const xTicks = 5;
  for (let i=0; i<=xTicks; i++) {
    const x = pad.left + pw * i/xTicks;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top+ph); ctx.stroke();
    const xVal = xRange[0]+(xRange[1]-xRange[0])*i/xTicks;
    ctx.fillStyle='#6c757d'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(xVal.toFixed(1), x, pad.top+ph+16);
  }

  // Axis labels
  ctx.fillStyle='#495057'; ctx.font='12px sans-serif'; ctx.textAlign='center';
  ctx.fillText(xLabel, pad.left+pw/2, H-6);
  ctx.save(); ctx.translate(14, pad.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(yLabel, 0, 0); ctx.restore();

  // Curves
  curves.forEach((curve, ci) => {
    const col = colors[ci % colors.length];
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath();
    xData.forEach((xv, i) => {
      const x = pad.left + pw * (xv - xRange[0]) / (xRange[1]-xRange[0]);
      const y = pad.top  + ph * (1 - (curve[i] - yRange[0]) / (yRange[1]-yRange[0]));
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
  });

  // Legend
  curves.forEach((_, ci) => {
    const col = colors[ci % colors.length];
    const ly = pad.top + ci * 22;
    ctx.fillStyle=col;
    ctx.fillRect(W-pad.right+12, ly, 14, 3);
    ctx.fillStyle='#212529'; ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillText(DATA.trackers[ci], W-pad.right+30, ly+4);
  });
}

// ── Scatter chart ────────────────────────────────────────────────────────────
function drawScatter(canvasId, points, colors) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {top:25, right:30, bottom:50, left:65};
  const pw = W - pad.left - pad.right, ph = H - pad.top - pad.bottom;

  ctx.clearRect(0,0,W,H);

  const allFps = points.map(p=>p.fps), allIou = points.map(p=>p.mean_iou);
  const maxFps = Math.max(...allFps)*1.15 || 10;
  const maxIou = Math.min(Math.max(...allIou)*1.2, 1.0) || 1.0;

  // Grid
  ctx.strokeStyle='#dee2e6'; ctx.lineWidth=1;
  [0,0.2,0.4,0.6,0.8,1.0].forEach(frac => {
    const y = pad.top + ph*(1-frac*(maxIou));  // approximate grid
    if (frac*maxIou > maxIou) return;
    const yv = frac*maxIou;
    const yp = pad.top + ph*(1-yv/maxIou);
    ctx.beginPath(); ctx.moveTo(pad.left,yp); ctx.lineTo(pad.left+pw,yp); ctx.stroke();
    ctx.fillStyle='#6c757d'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText(yv.toFixed(2), pad.left-6, yp+4);
  });

  // Axis labels
  ctx.fillStyle='#495057'; ctx.font='12px sans-serif'; ctx.textAlign='center';
  ctx.fillText('FPS (higher is faster)', pad.left+pw/2, H-6);
  ctx.save(); ctx.translate(14, pad.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Mean IoU (higher is better)', 0, 0); ctx.restore();

  // Points
  points.forEach((p, i) => {
    const x = pad.left + pw*(p.fps/maxFps);
    const y = pad.top  + ph*(1 - p.mean_iou/maxIou);
    const col = colors[i % colors.length];
    ctx.beginPath(); ctx.arc(x,y,7,0,2*Math.PI);
    ctx.fillStyle = rgba(col,0.85); ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth=1.5; ctx.stroke();
    ctx.fillStyle='#212529'; ctx.font='bold 11px sans-serif'; ctx.textAlign='left';
    ctx.fillText(p.label, x+10, y+4);
  });
}

// ── Summary table ────────────────────────────────────────────────────────────
function buildSummaryTable(rows, hasAuc, hasEnergy) {
  const cols = ['tracker','dataset','mean_iou'];
  const heads = ['Tracker','Dataset','mIoU'];
  if (hasAuc) { cols.push('success_auc','precision_auc'); heads.push('Success AUC','Precision AUC'); }
  cols.push('mean_fps'); heads.push('FPS');
  if (hasEnergy) { cols.push('energy_per_frame_mj'); heads.push('E/frame (mJ)'); }

  // Find best per numeric column
  const best = {};
  cols.slice(2).forEach(c => {
    const vals = rows.map(r=>r[c]||0);
    best[c] = (c==='mean_fps'||c==='success_auc'||c==='precision_auc'||c==='mean_iou')
      ? Math.max(...vals) : Math.min(...vals);
  });

  let html = '<table><thead><tr>';
  heads.forEach(h => html += `<th>${h}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(c => {
      const v = row[c];
      const fmt = typeof v==='number' ? v.toFixed(4) : (v||'—');
      const cls = (typeof v==='number' && best[c]!==undefined && Math.abs(v-best[c])<1e-6) ? ' class="best"' : '';
      html += `<td${cls}>${fmt}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('summary-table').innerHTML = html;
}

// ── Per-sequence accordion ───────────────────────────────────────────────────
function buildAccordion(perTracker) {
  const el = document.getElementById('sequences-accordion');
  perTracker.forEach(({tracker, sequences}) => {
    const btn = document.createElement('button');
    btn.className = 'accordion-btn';
    btn.innerHTML = `<span>${tracker} <small style="color:#6c757d">(${sequences.length} sequences)</small></span><span class="arrow">&#9658;</span>`;

    const body = document.createElement('div');
    body.className = 'accordion-body';

    const hasSauc = sequences.some(s=>s.success_auc!==undefined);
    let th = '<tr><th>Sequence</th><th>mIoU</th>';
    if (hasSauc) th += '<th>Success AUC</th>';
    th += '<th>FPS</th><th>Latency (ms)</th><th>Mem (MB)</th></tr>';

    let rows = '';
    sequences.forEach(s => {
      rows += `<tr><td>${s.name}</td><td>${s.mean_iou.toFixed(4)}</td>`;
      if (hasSauc) rows += `<td>${s.success_auc!==undefined ? s.success_auc.toFixed(4) : '—'}</td>`;
      rows += `<td>${s.fps.toFixed(1)}</td><td>${s.latency_ms.toFixed(2)}</td><td>${s.memory_mb.toFixed(1)}</td></tr>`;
    });

    body.innerHTML = `<table><thead>${th}</thead><tbody>${rows}</tbody></table>`;

    btn.addEventListener('click', () => {
      body.classList.toggle('open');
      btn.querySelector('.arrow').classList.toggle('open');
    });

    el.appendChild(btn);
    el.appendChild(body);
  });
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
buildSummaryTable(DATA.summary_rows, DATA.has_auc, DATA.has_energy);

drawLineChart(
  'success-canvas',
  DATA.iou_thresholds,
  DATA.success_curves.map(c=>c.rates),
  DATA.colors,
  'IoU Threshold', 'Success Rate',
  [0,1], [0,1]
);

drawLineChart(
  'precision-canvas',
  DATA.dist_thresholds,
  DATA.precision_curves.map(c=>c.rates),
  DATA.colors,
  'Centre Distance Threshold (px)', 'Precision',
  [0,50], [0,1]
);

drawScatter('scatter-canvas', DATA.scatter_points, DATA.colors);
buildAccordion(DATA.per_tracker_sequences);
"""
