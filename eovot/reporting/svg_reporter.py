"""Zero-dependency HTML report generator for EOVOT benchmark results.

Produces a single self-contained HTML file with embedded CSS and inline SVG
charts — no CDN links, no JavaScript libraries, and no network access required
at view time.  The file can be opened in any browser, attached to a CI artifact,
embedded in a notebook cell, or distributed offline to reviewers.

This complements the existing :class:`~eovot.reporting.reporter.BenchmarkReporter`
(which exports JSON/CSV/Markdown) and is deliberately independent of matplotlib,
making it suitable for headless CI environments and edge-lab machines without a
display server.

Charts included
---------------
- **Success curves** — fraction of frames with IoU > threshold, one coloured
  line per tracker; AUC value in the legend (standard VOT evaluation plot).
- **Efficiency scatter** — FPS vs mIoU bubble chart; bubble area encodes peak
  memory footprint.  Identifies the best accuracy-throughput trade-off at a glance.
- **Metric bar charts** — side-by-side bars for mIoU, FPS, latency, and memory.
- **Summary table** — colour-coded HTML table with best-value highlighting.

Example::

    from eovot.reporting.svg_reporter import SvgHtmlReporter
    import json

    with open("results/MOSSE-OTB100.json") as f:
        mosse = json.load(f)
    with open("results/KCF-OTB100.json") as f:
        kcf = json.load(f)

    reporter = SvgHtmlReporter(output_dir="results/")
    path = reporter.generate([mosse, kcf], name="classical_comparison")
    print(f"Report written to {path}")
"""

from __future__ import annotations

import html as _html_mod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (10 distinguishable, colour-blind-friendly hues)
# ---------------------------------------------------------------------------

_PALETTE: List[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
]


def _colour(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Success-curve computation (mirrors MetricsEngine.success_curve)
# ---------------------------------------------------------------------------

def _compute_success_curve(
    ious: np.ndarray,
    n_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the VOT success curve from a flat IoU array."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    rates = np.array([(ious > t).mean() for t in thresholds])
    return thresholds, rates


def _collect_ious(result: Dict[str, Any]) -> np.ndarray:
    """Gather all per-frame IoU values from a benchmark result dict."""
    parts: List[float] = []
    for seq in result.get("sequences", []):
        raw = seq.get("ious")
        if raw:
            parts.extend(raw)
        else:
            parts.append(float(seq.get("mean_iou", 0.0)))
    return np.array(parts, dtype=np.float64) if parts else np.zeros(1)


# ---------------------------------------------------------------------------
# SVG layout constants
# ---------------------------------------------------------------------------

_W, _H = 520, 330
_PL, _PR, _PT, _PB = 54, 20, 24, 52
_PW = _W - _PL - _PR
_PH = _H - _PT - _PB


def _svg_open(w: int = _W, h: int = _H) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" '
        f'style="font-family:sans-serif;font-size:11px;">'
    )


def _esc(s: str) -> str:
    return _html_mod.escape(str(s))


def _rect(x: float, y: float, w: float, h: float, fill: str, **kw) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" {attrs}/>'


def _text(x: float, y: float, content: str, **kw) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{_esc(str(content))}</text>'


def _line(x1, y1, x2, y2, **kw) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attrs}/>'


def _polyline(pts: List[Tuple[float, float]], stroke: str, **kw) -> str:
    pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<polyline points="{pts_str}" fill="none" stroke="{stroke}" stroke-width="2" {attrs}/>'


def _grid_and_axes(
    y_ticks: int = 4,
    x_ticks: int = 5,
    x_max: float = 1.0,
    y_max: float = 1.0,
    x_fmt: str = ".1f",
    y_fmt: str = ".1f",
) -> List[str]:
    """Shared grid lines and axis tick labels."""
    lines = [
        _rect(_PL, _PT, _PW, _PH, "#fafafa", stroke="#ccc", **{"stroke-width": "1"}),
    ]
    for i in range(1, y_ticks + 1):
        frac = i / y_ticks
        y = _PT + _PH * (1 - frac)
        lines.append(_line(_PL, y, _PL + _PW, y, stroke="#e0e0e0", **{"stroke-dasharray": "4"}))
        lines.append(_text(_PL - 6, y + 4, format(y_max * frac, y_fmt),
                           **{"text-anchor": "end", "fill": "#555"}))
    if x_ticks > 0:
        for i in range(x_ticks + 1):
            frac = i / x_ticks
            x = _PL + _PW * frac
            lines.append(_line(x, _PT + _PH, x, _PT + _PH + 4, stroke="#888"))
            lines.append(_text(x, _PT + _PH + 16, format(x_max * frac, x_fmt),
                               **{"text-anchor": "middle", "fill": "#555"}))
    return lines


# ---------------------------------------------------------------------------
# Chart: success curves
# ---------------------------------------------------------------------------

def _chart_success_curves(results: List[Dict[str, Any]]) -> str:
    parts: List[str] = [_svg_open()]
    parts += _grid_and_axes()

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]

    for idx, res in enumerate(results):
        colour = _colour(idx)
        ious = _collect_ious(res)
        thr, rates = _compute_success_curve(ious)
        auc = float(_trapz(rates, thr))
        pts = [(_PL + _PW * t, _PT + _PH * (1 - r)) for t, r in zip(thr, rates)]
        parts.append(_polyline(pts, colour))
        name = res.get("summary", {}).get("tracker", f"Tracker {idx + 1}")
        parts.append(_text(
            _PL + _PW - 6, pts[-1][1] - 6,
            f"{name} [{auc:.3f}]",
            **{"text-anchor": "end", "fill": colour, "font-size": "10"},
        ))

    parts.append(_text(_PL + _PW / 2, _H - 8, "IoU Threshold",
                       **{"text-anchor": "middle", "fill": "#333"}))
    parts.append(
        f'<text x="14" y="{_PT + _PH / 2:.0f}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90,14,{_PT + _PH / 2:.0f})">Success Rate</text>'
    )
    parts.append(_text(_PL + _PW / 2, _PT - 8, "Success Curves (AUC in legend)",
                       **{"text-anchor": "middle", "fill": "#222", "font-weight": "bold"}))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chart: efficiency scatter (FPS × mIoU, bubble = memory)
# ---------------------------------------------------------------------------

def _chart_efficiency_scatter(results: List[Dict[str, Any]]) -> str:
    summaries = [r.get("summary", {}) for r in results]
    fps_vals = [float(s.get("mean_fps", 0.0)) for s in summaries]
    iou_vals = [float(s.get("mean_iou", 0.0)) for s in summaries]
    mem_vals = [float(s.get("peak_memory_mb", 1.0)) for s in summaries]

    max_fps = max(fps_vals) if any(v > 0 for v in fps_vals) else 1.0
    max_mem = max(mem_vals) if any(v > 0 for v in mem_vals) else 1.0

    parts: List[str] = [_svg_open()]
    parts += _grid_and_axes(x_max=max_fps, x_fmt=".0f")

    for idx, (fps, iou, mem, s) in enumerate(zip(fps_vals, iou_vals, mem_vals, summaries)):
        colour = _colour(idx)
        cx = _PL + _PW * (fps / max_fps if max_fps > 0 else 0)
        cy = _PT + _PH * (1.0 - min(iou, 1.0))
        r = max(5.0, 16.0 * mem / max_mem)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{colour}" opacity="0.75"/>')
        name = s.get("tracker", f"T{idx}")
        parts.append(_text(cx, cy - r - 4, name[:12],
                           **{"text-anchor": "middle", "fill": colour, "font-size": "10"}))

    parts.append(_text(_PL + _PW / 2, _H - 8, "Mean FPS",
                       **{"text-anchor": "middle", "fill": "#333"}))
    parts.append(
        f'<text x="14" y="{_PT + _PH / 2:.0f}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90,14,{_PT + _PH / 2:.0f})">Mean IoU</text>'
    )
    parts.append(_text(_PL + _PW / 2, _PT - 8,
                       "Efficiency Scatter (bubble area ∝ peak memory)",
                       **{"text-anchor": "middle", "fill": "#222", "font-weight": "bold"}))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chart: bar chart for a single metric
# ---------------------------------------------------------------------------

def _chart_bars(
    results: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    y_label: str,
) -> str:
    summaries = [r.get("summary", {}) for r in results]
    values = [float(s.get(metric_key, 0.0)) for s in summaries]
    names = [s.get("tracker", f"T{i}") for i, s in enumerate(summaries)]
    max_val = max(values) if values else 1.0

    n = max(len(values), 1)
    slot_w = _PW / n
    bar_w = min(55.0, slot_w * 0.65)

    parts: List[str] = [_svg_open()]
    parts += _grid_and_axes(y_max=max_val, y_fmt=".2g", x_ticks=0)

    for idx, (val, name) in enumerate(zip(values, names)):
        colour = _colour(idx)
        frac = val / max_val if max_val > 0 else 0.0
        bh = _PH * frac
        bx = _PL + slot_w * idx + (slot_w - bar_w) / 2
        by = _PT + _PH - bh
        parts.append(_rect(bx, by, bar_w, bh, colour, opacity="0.85"))
        parts.append(_text(bx + bar_w / 2, _PT + _PH + 14, name[:12],
                           **{"text-anchor": "middle", "fill": "#333", "font-size": "10"}))
        parts.append(_text(bx + bar_w / 2, max(by - 3, _PT + 12),
                           format(val, ".2g"),
                           **{"text-anchor": "middle", "fill": "#333", "font-size": "10"}))

    parts.append(_text(_PL + _PW / 2, _H - 8, "Tracker",
                       **{"text-anchor": "middle", "fill": "#333"}))
    parts.append(
        f'<text x="14" y="{_PT + _PH / 2:.0f}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90,14,{_PT + _PH / 2:.0f})">{_esc(y_label)}</text>'
    )
    parts.append(_text(_PL + _PW / 2, _PT - 8, title,
                       **{"text-anchor": "middle", "fill": "#222", "font-weight": "bold"}))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("tracker",        "Tracker",       False),
    ("dataset",        "Dataset",       False),
    ("mean_iou",       "mIoU",          True),
    ("success_auc",    "Success AUC",   True),
    ("precision_auc",  "Precision AUC", True),
    ("mean_fps",       "FPS",           True),
    ("peak_memory_mb", "Mem (MB)",      False),
    ("num_sequences",  "Seqs",          False),
]


def _summary_table(results: List[Dict[str, Any]]) -> str:
    # Find the best value per numeric column for highlighting
    best: Dict[str, float] = {}
    for key, _, numeric in _TABLE_COLS:
        if not numeric:
            continue
        vals = []
        for r in results:
            v = r.get("summary", {}).get(key)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            best[key] = max(vals)

    rows = []
    for res in results:
        s = res.get("summary", {})
        cells = []
        for key, _, numeric in _TABLE_COLS:
            raw = s.get(key)
            if raw is None:
                cells.append("<td>—</td>")
                continue
            if numeric:
                try:
                    fval = float(raw)
                    formatted = f"{fval:.4f}" if key not in ("mean_fps",) else f"{fval:.1f}"
                    is_best = best.get(key) is not None and abs(fval - best[key]) < 1e-9
                    style = ' style="background:#d4edda;font-weight:bold;"' if is_best else ""
                    cells.append(f"<td{style}>{_esc(formatted)}</td>")
                except (TypeError, ValueError):
                    cells.append(f"<td>{_esc(str(raw))}</td>")
            else:
                cells.append(f"<td>{_esc(str(raw))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    header = "".join(f"<th>{_esc(label)}</th>" for _, label, _ in _TABLE_COLS)
    return (
        '<table class="summary">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:0;padding:20px 28px;background:#f5f6fa;color:#333}
h1{font-size:1.55em;border-bottom:2px solid #2c7be5;padding-bottom:8px}
h2{font-size:1.15em;color:#2c7be5;margin-top:28px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:16px 0}
.box{background:#fff;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.1);
  padding:12px;overflow:hidden}
table.summary{border-collapse:collapse;width:100%;background:#fff;
  border-radius:6px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.1)}
table.summary th{background:#2c7be5;color:#fff;padding:7px 11px;
  text-align:right;font-size:.88em}
table.summary th:first-child,table.summary th:nth-child(2){text-align:left}
table.summary td{padding:6px 11px;text-align:right;
  border-bottom:1px solid #eee;font-size:.86em}
table.summary td:first-child,table.summary td:nth-child(2){text-align:left}
table.summary tr:hover td{background:#f0f7ff}
footer{margin-top:36px;font-size:.78em;color:#aaa;text-align:center}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

class SvgHtmlReporter:
    """Generate a zero-dependency, self-contained HTML benchmark report.

    The output is a single HTML file with all charts rendered as inline SVG.
    No CDN, no external scripts, no matplotlib — the file works fully offline.

    This is designed for:

    - **CI artifact uploads** — the report is one file with no side-car assets.
    - **Offline edge labs** — no internet connection required to view.
    - **Reproducibility archives** — the report captures a snapshot of results
      in a human-readable format without requiring Python to render.

    Args:
        output_dir: Directory where HTML files are written.  Created
            automatically if it does not exist.  Default: ``"results/"``.
    """

    def __init__(self, output_dir: str = "results/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List[Dict[str, Any]],
        name: str = "benchmark_report",
        title: Optional[str] = None,
    ) -> Path:
        """Build and write the HTML report.

        Args:
            results:  List of result dicts, one per tracker/dataset combination.
                      Each dict is the output of
                      :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name:     Base filename without extension.  Default: ``"benchmark_report"``.
            title:    Page heading.  Defaults to
                      ``"EOVOT Benchmark Report — <name>"``.

        Returns:
            :class:`pathlib.Path` of the written ``.html`` file.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results list is empty — nothing to report.")

        page_title = title or f"EOVOT Benchmark Report — {name}"

        sc_svg   = _chart_success_curves(results)
        eff_svg  = _chart_efficiency_scatter(results)
        iou_svg  = _chart_bars(results, "mean_iou", "Mean IoU", "mIoU")
        fps_svg  = _chart_bars(results, "mean_fps", "Throughput", "FPS")
        lat_svg  = _chart_bars(results, "mean_latency_ms", "Mean Latency", "ms/frame")
        mem_svg  = _chart_bars(results, "peak_memory_mb", "Peak Memory", "MB")
        table    = _summary_table(results)

        page = (
            "<!doctype html>\n"
            "<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n"
            f"<title>{_esc(page_title)}</title>\n"
            f"<style>\n{_CSS}\n</style>\n"
            "</head>\n<body>\n"
            f"<h1>{_esc(page_title)}</h1>\n"
            "<h2>Accuracy</h2>\n"
            f'<div class="grid2"><div class="box">{sc_svg}</div>'
            f'<div class="box">{eff_svg}</div></div>\n'
            "<h2>Per-Metric Breakdown</h2>\n"
            f'<div class="grid2">'
            f'<div class="box">{iou_svg}</div>'
            f'<div class="box">{fps_svg}</div>'
            f'<div class="box">{lat_svg}</div>'
            f'<div class="box">{mem_svg}</div>'
            f'</div>\n'
            "<h2>Summary</h2>\n"
            f"{table}\n"
            "<footer>Generated by EOVOT &mdash; "
            "Edge-Optimized Visual Object Tracking Benchmark Suite</footer>\n"
            "</body>\n</html>"
        )

        path = self.output_dir / f"{name}.html"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page)
        return path

    def generate_from_json_files(
        self,
        json_paths: List[str],
        name: str = "benchmark_report",
        title: Optional[str] = None,
    ) -> Path:
        """Load result dicts from JSON files, then call :meth:`generate`.

        Args:
            json_paths: Paths to JSON files written by
                :meth:`~eovot.reporting.reporter.BenchmarkReporter.save_json`.
            name:   Base filename without extension.
            title:  Optional page heading.

        Returns:
            :class:`pathlib.Path` of the written HTML file.
        """
        loaded = []
        for p in json_paths:
            with open(p, encoding="utf-8") as fh:
                loaded.append(json.load(fh))
        return self.generate(loaded, name=name, title=title)
