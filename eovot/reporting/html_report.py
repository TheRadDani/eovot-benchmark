"""Self-contained HTML benchmark report generator for EOVOT.

Generates a single ``.html`` file that can be shared, opened offline, and
viewed in any browser — no external CDN, no matplotlib, no network required.

All charts are rendered as inline SVG computed directly from the benchmark
result dicts produced by :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

Charts included:
- **Success curves** — overlap threshold vs. success rate, one line per tracker.
- **Precision curves** — centre-distance threshold vs. precision rate.
- **FPS-IoU scatter** — edge-deployment view with optional Pareto front.
- **Latency percentile bars** — p50 / p95 / p99 grouped bar chart.

Example::

    from eovot.reporting.html_report import HTMLReportGenerator

    gen = HTMLReportGenerator()

    # from in-memory result dicts
    gen.save([mosse_result_dict, kcf_result_dict], "results/report.html")

    # or from BenchmarkResult objects
    from eovot.benchmark.engine import BenchmarkResult
    results = [BenchmarkResult.load("mosse.json"), BenchmarkResult.load("kcf.json")]
    gen.save([r.to_dict() for r in results], "results/report.html")
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (high-contrast, colour-blind-friendly)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#65a30d",  # lime
]


def _colour(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _svg_header(width: int, height: int, id_: str = "") -> str:
    id_attr = f' id="{id_}"' if id_ else ""
    return (
        f'<svg{id_attr} xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )


def _axis_labels(
    *,
    width: int,
    height: int,
    pad_l: int,
    pad_b: int,
    pad_r: int,
    pad_t: int,
    x_label: str,
    y_label: str,
    x_ticks: int = 5,
    y_ticks: int = 5,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> str:
    """Return SVG elements for axes, ticks, and axis labels."""
    parts = []
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # Axes
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        f'stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="#94a3b8" stroke-width="1"/>'
    )

    # X ticks
    for i in range(x_ticks + 1):
        xv = x_min + (x_max - x_min) * i / x_ticks
        xpx = pad_l + plot_w * i / x_ticks
        parts.append(
            f'<line x1="{xpx:.1f}" y1="{pad_t + plot_h}" '
            f'x2="{xpx:.1f}" y2="{pad_t + plot_h + 4}" '
            f'stroke="#94a3b8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xpx:.1f}" y="{pad_t + plot_h + 16}" '
            f'text-anchor="middle" font-size="11" fill="#64748b">{xv:.2f}</text>'
        )
        # Grid line
        parts.append(
            f'<line x1="{xpx:.1f}" y1="{pad_t}" x2="{xpx:.1f}" '
            f'y2="{pad_t + plot_h}" stroke="#e2e8f0" stroke-width="1" '
            f'stroke-dasharray="4,4"/>'
        )

    # Y ticks
    for i in range(y_ticks + 1):
        yv = y_min + (y_max - y_min) * i / y_ticks
        ypx = pad_t + plot_h - plot_h * i / y_ticks
        parts.append(
            f'<line x1="{pad_l - 4}" y1="{ypx:.1f}" x2="{pad_l}" '
            f'y2="{ypx:.1f}" stroke="#94a3b8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{ypx + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="#64748b">{yv:.2f}</text>'
        )
        parts.append(
            f'<line x1="{pad_l}" y1="{ypx:.1f}" x2="{pad_l + plot_w}" '
            f'y2="{ypx:.1f}" stroke="#e2e8f0" stroke-width="1" '
            f'stroke-dasharray="4,4"/>'
        )

    # Axis labels
    parts.append(
        f'<text x="{pad_l + plot_w // 2}" y="{height - 4}" '
        f'text-anchor="middle" font-size="12" font-weight="600" '
        f'fill="#475569">{html.escape(x_label)}</text>'
    )
    # Rotated Y label
    cy = pad_t + plot_h // 2
    parts.append(
        f'<text x="14" y="{cy}" '
        f'text-anchor="middle" font-size="12" font-weight="600" '
        f'fill="#475569" transform="rotate(-90,14,{cy})">'
        f'{html.escape(y_label)}</text>'
    )

    return "\n".join(parts)


def _data_to_svg_point(
    x: float, y: float,
    *,
    pad_l: int, pad_t: int, plot_w: int, plot_h: int,
    x_min: float, x_max: float, y_min: float, y_max: float,
) -> Tuple[float, float]:
    """Map a data-space (x, y) to SVG pixel coordinates."""
    px = pad_l + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
    py = pad_t + plot_h - (y - y_min) / max(y_max - y_min, 1e-9) * plot_h
    return px, py


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_success_curves(results: List[Dict[str, Any]]) -> str:
    """Build an SVG success-curve plot from benchmark result dicts."""
    W, H = 560, 340
    PAD_L, PAD_R, PAD_T, PAD_B = 55, 20, 24, 50
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    thresholds = np.linspace(0.0, 1.0, 101)

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]

    parts = [_svg_header(W, H)]
    parts.append(_axis_labels(
        width=W, height=H, pad_l=PAD_L, pad_b=PAD_B, pad_r=PAD_R, pad_t=PAD_T,
        x_label="IoU Threshold", y_label="Success Rate",
        x_ticks=5, y_ticks=5, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
    ))

    legend_items = []
    for idx, result in enumerate(results):
        s = result.get("summary", {})
        name = s.get("tracker") or s.get("tracker_name", f"tracker_{idx}")
        ious_list: List[float] = []
        for seq in result.get("sequences", []):
            raw = seq.get("ious")
            if raw:
                ious_list.extend(raw)
            elif "mean_iou" in seq:
                ious_list.append(float(seq["mean_iou"]))
        if not ious_list:
            continue
        ious = np.array(ious_list, dtype=np.float64)
        success_rates = np.array([(ious > t).mean() for t in thresholds])
        auc = float(_trapz(success_rates, thresholds))

        points = []
        for t, sr in zip(thresholds, success_rates):
            px, py = _data_to_svg_point(
                t, sr, pad_l=PAD_L, pad_t=PAD_T, plot_w=plot_w, plot_h=plot_h,
                x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            )
            points.append(f"{px:.2f},{py:.2f}")

        colour = _colour(idx)
        parts.append(
            f'<polyline points="{" ".join(points)}" '
            f'fill="none" stroke="{colour}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        legend_items.append((name, auc, colour))

    # Legend
    lx, ly = PAD_L + 10, PAD_T + 10
    for i, (name, auc, colour) in enumerate(legend_items):
        ey = ly + i * 20
        parts.append(f'<line x1="{lx}" y1="{ey + 6}" x2="{lx + 22}" y2="{ey + 6}" '
                     f'stroke="{colour}" stroke-width="2.5"/>')
        label = f"{html.escape(name)} (AUC={auc:.3f})"
        parts.append(f'<text x="{lx + 28}" y="{ey + 10}" font-size="11" fill="#1e293b">'
                     f'{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _build_precision_curves(results: List[Dict[str, Any]]) -> str:
    """Build an SVG precision-curve plot."""
    W, H = 560, 340
    PAD_L, PAD_R, PAD_T, PAD_B = 55, 20, 24, 50
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    thresholds = np.linspace(0.0, 50.0, 101)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]

    parts = [_svg_header(W, H)]
    parts.append(_axis_labels(
        width=W, height=H, pad_l=PAD_L, pad_b=PAD_B, pad_r=PAD_R, pad_t=PAD_T,
        x_label="Centre-Distance Threshold (px)", y_label="Precision Rate",
        x_ticks=5, y_ticks=5, x_min=0.0, x_max=50.0, y_min=0.0, y_max=1.0,
    ))

    legend_items = []
    for idx, result in enumerate(results):
        s = result.get("summary", {})
        name = s.get("tracker") or s.get("tracker_name", f"tracker_{idx}")
        dists_list: List[float] = []
        for seq in result.get("sequences", []):
            raw = seq.get("center_distances")
            if raw:
                dists_list.extend(raw)
        if not dists_list:
            continue
        dists = np.array(dists_list, dtype=np.float64)
        prec_rates = np.array([(dists < t).mean() for t in thresholds])
        score_at_20 = float(np.interp(20.0, thresholds, prec_rates))

        points = []
        for t, pr in zip(thresholds, prec_rates):
            px, py = _data_to_svg_point(
                t, pr, pad_l=PAD_L, pad_t=PAD_T, plot_w=plot_w, plot_h=plot_h,
                x_min=0.0, x_max=50.0, y_min=0.0, y_max=1.0,
            )
            points.append(f"{px:.2f},{py:.2f}")

        colour = _colour(idx)
        parts.append(
            f'<polyline points="{" ".join(points)}" '
            f'fill="none" stroke="{colour}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        legend_items.append((name, score_at_20, colour))

    lx, ly = PAD_L + 10, PAD_T + 10
    for i, (name, score, colour) in enumerate(legend_items):
        ey = ly + i * 20
        parts.append(f'<line x1="{lx}" y1="{ey + 6}" x2="{lx + 22}" y2="{ey + 6}" '
                     f'stroke="{colour}" stroke-width="2.5"/>')
        label = f"{html.escape(name)} (@20px={score:.3f})"
        parts.append(f'<text x="{lx + 28}" y="{ey + 10}" font-size="11" fill="#1e293b">'
                     f'{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _build_fps_iou_scatter(results: List[Dict[str, Any]]) -> str:
    """Build an SVG FPS-IoU scatter plot with Pareto front."""
    W, H = 560, 360
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 24, 24, 55
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    points_data: List[Tuple[str, float, float]] = []
    for idx, result in enumerate(results):
        s = result.get("summary", {})
        name = s.get("tracker") or s.get("tracker_name", f"tracker_{idx}")
        fps = float(s.get("mean_fps", 0.0))
        miou = float(s.get("mean_iou", 0.0))
        if fps > 0 and miou > 0:
            points_data.append((name, fps, miou))

    if not points_data:
        return (
            f'{_svg_header(W, H)}'
            f'<text x="{W//2}" y="{H//2}" text-anchor="middle" '
            f'font-size="13" fill="#94a3b8">No data to display</text></svg>'
        )

    max_fps = max(p[1] for p in points_data) * 1.15

    parts = [_svg_header(W, H)]
    parts.append(_axis_labels(
        width=W, height=H, pad_l=PAD_L, pad_b=PAD_B, pad_r=PAD_R, pad_t=PAD_T,
        x_label="Mean FPS", y_label="Mean IoU",
        x_ticks=5, y_ticks=5, x_min=0.0, x_max=max_fps, y_min=0.0, y_max=1.0,
    ))

    # Pareto front
    if len(points_data) > 1:
        sorted_pts = sorted(points_data, key=lambda p: p[1], reverse=True)
        pf: List[Tuple[float, float]] = []
        best_iou = -1.0
        for _, fps, iou in sorted_pts:
            if iou >= best_iou:
                pf.append((fps, iou))
                best_iou = iou
        pf.sort(reverse=True)
        step_pts = []
        for i, (fps, iou) in enumerate(pf):
            px, py = _data_to_svg_point(
                fps, iou, pad_l=PAD_L, pad_t=PAD_T, plot_w=plot_w, plot_h=plot_h,
                x_min=0.0, x_max=max_fps, y_min=0.0, y_max=1.0,
            )
            if i > 0:
                prev_px, _ = _data_to_svg_point(
                    pf[i - 1][0], iou,
                    pad_l=PAD_L, pad_t=PAD_T, plot_w=plot_w, plot_h=plot_h,
                    x_min=0.0, x_max=max_fps, y_min=0.0, y_max=1.0,
                )
                step_pts.append(f"{prev_px:.2f},{py:.2f}")
            step_pts.append(f"{px:.2f},{py:.2f}")
        if step_pts:
            parts.append(
                f'<polyline points="{" ".join(step_pts)}" fill="none" '
                f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>'
            )
            parts.append(
                f'<text x="{PAD_L + plot_w - 4}" y="{PAD_T + 14}" '
                f'text-anchor="end" font-size="10" fill="#94a3b8">— Pareto front</text>'
            )

    # Points
    for idx, (name, fps, miou) in enumerate(points_data):
        px, py = _data_to_svg_point(
            fps, miou, pad_l=PAD_L, pad_t=PAD_T, plot_w=plot_w, plot_h=plot_h,
            x_min=0.0, x_max=max_fps, y_min=0.0, y_max=1.0,
        )
        colour = _colour(idx)
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" '
            f'fill="{colour}" fill-opacity="0.85" stroke="white" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{px + 10:.2f}" y="{py + 4:.2f}" '
            f'font-size="11" fill="#1e293b">{html.escape(name)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_latency_bars(results: List[Dict[str, Any]]) -> str:
    """Build an SVG grouped latency bar chart (p50 / p95 / p99)."""
    W, H = 560, 300
    PAD_L, PAD_R, PAD_T, PAD_B = 65, 24, 30, 60
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    tracker_data: List[Tuple[str, float, float, float]] = []
    for result in results:
        s = result.get("summary", {})
        name = s.get("tracker") or s.get("tracker_name", "?")
        seqs = result.get("sequences", [])
        latencies = [float(sq.get("mean_latency_ms", 0.0)) for sq in seqs if "mean_latency_ms" in sq]
        if not latencies:
            continue
        arr = np.array(latencies)
        tracker_data.append((
            name,
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        ))

    if not tracker_data:
        return ""

    max_ms = max(max(p[1], p[2], p[3]) for p in tracker_data) * 1.15
    n = len(tracker_data)
    group_w = plot_w / n
    bar_w = group_w * 0.22

    def _bar_y(val: float) -> float:
        return PAD_T + plot_h - (val / max(max_ms, 1e-9)) * plot_h

    parts = [_svg_header(W, H)]
    # Y axis
    y_ticks = 5
    for i in range(y_ticks + 1):
        yv = max_ms * i / y_ticks
        ypx = PAD_T + plot_h - plot_h * i / y_ticks
        parts.append(
            f'<text x="{PAD_L - 6}" y="{ypx + 4:.1f}" '
            f'text-anchor="end" font-size="10" fill="#64748b">{yv:.1f}</text>'
        )
        parts.append(
            f'<line x1="{PAD_L}" y1="{ypx:.1f}" x2="{PAD_L + plot_w}" '
            f'y2="{ypx:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,4"/>'
        )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" '
        f'y2="{PAD_T + plot_h}" stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + plot_h}" x2="{PAD_L + plot_w}" '
        f'y2="{PAD_T + plot_h}" stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="14" y="{PAD_T + plot_h // 2}" text-anchor="middle" '
        f'font-size="12" font-weight="600" fill="#475569" '
        f'transform="rotate(-90,14,{PAD_T + plot_h // 2})">Latency (ms)</text>'
    )

    bar_colours = ["#2563eb", "#d97706", "#dc2626"]
    labels = ["p50", "p95", "p99"]

    for gi, (name, p50, p95, p99) in enumerate(tracker_data):
        cx = PAD_L + group_w * (gi + 0.5)
        vals = [p50, p95, p99]
        for bi, (val, colour, label) in enumerate(zip(vals, bar_colours, labels)):
            bx = cx + (bi - 1) * (bar_w + 2)
            by = _bar_y(val)
            bh = PAD_T + plot_h - by
            parts.append(
                f'<rect x="{bx - bar_w / 2:.2f}" y="{by:.2f}" '
                f'width="{bar_w:.2f}" height="{bh:.2f}" '
                f'fill="{colour}" fill-opacity="0.8" rx="2"/>'
            )
        # Tracker name below axis
        parts.append(
            f'<text x="{cx:.1f}" y="{PAD_T + plot_h + 16}" '
            f'text-anchor="middle" font-size="11" fill="#1e293b">'
            f'{html.escape(name)}</text>'
        )

    # Legend
    lx = PAD_L + 10
    ly = PAD_T + 6
    for i, (colour, label) in enumerate(zip(bar_colours, labels)):
        parts.append(f'<rect x="{lx + i * 60}" y="{ly}" width="14" height="10" '
                     f'fill="{colour}" fill-opacity="0.8" rx="2"/>')
        parts.append(f'<text x="{lx + i * 60 + 18}" y="{ly + 9}" '
                     f'font-size="11" fill="#1e293b">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Leaderboard table builder
# ---------------------------------------------------------------------------

def _build_leaderboard_html(results: List[Dict[str, Any]]) -> str:
    """Build an HTML leaderboard table ranked by success AUC then mIoU."""
    if not results:
        return "<p>No results to display.</p>"

    rows = []
    for r in results:
        s = r.get("summary", {})
        miou = float(s.get("mean_iou", 0.0))
        sauc = float(s.get("success_auc", miou))
        rows.append({
            "tracker": s.get("tracker") or s.get("tracker_name", "?"),
            "dataset": s.get("dataset") or s.get("dataset_name", "?"),
            "sequences": int(s.get("num_sequences", 0)),
            "miou": miou,
            "success_auc": sauc,
            "precision_auc": float(s.get("precision_auc", 0.0)),
            "fps": float(s.get("mean_fps", 0.0)),
            "mem_mb": float(s.get("peak_memory_mb", 0.0)),
            "energy_mj": s.get("mean_energy_per_frame_mj"),
        })
    rows.sort(key=lambda x: x["success_auc"], reverse=True)

    lines = [
        '<table class="leaderboard">',
        "<thead><tr>",
        "<th>Rank</th><th>Tracker</th><th>Dataset</th>",
        "<th>Sequences</th><th>mIoU</th><th>Success AUC</th>",
        "<th>Precision AUC</th><th>FPS</th><th>Mem (MB)</th>",
        "<th>Energy (mJ/fr)</th></tr></thead><tbody>",
    ]
    for rank, row in enumerate(rows, start=1):
        energy_cell = f'{row["energy_mj"]:.3f}' if row["energy_mj"] is not None else "—"
        lines.append(
            f'<tr>'
            f'<td class="rank">#{rank}</td>'
            f'<td class="tracker-name">{html.escape(row["tracker"])}</td>'
            f'<td>{html.escape(row["dataset"])}</td>'
            f'<td class="num">{row["sequences"]}</td>'
            f'<td class="num">{row["miou"]:.4f}</td>'
            f'<td class="num highlight">{row["success_auc"]:.4f}</td>'
            f'<td class="num">{row["precision_auc"]:.4f}</td>'
            f'<td class="num">{row["fps"]:.1f}</td>'
            f'<td class="num">{row["mem_mb"]:.1f}</td>'
            f'<td class="num">{energy_cell}</td>'
            f'</tr>'
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  line-height: 1.5;
}
header {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color: #f8fafc;
  padding: 32px 40px;
}
header h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: 6px; }
header p  { font-size: 0.9rem; color: #94a3b8; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 20px; }
section { background: #fff; border-radius: 10px; padding: 28px;
          margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
section h2 {
  font-size: 1.1rem; font-weight: 700; color: #0f172a;
  margin-bottom: 20px; padding-bottom: 10px;
  border-bottom: 2px solid #e2e8f0;
}
.charts-row { display: flex; flex-wrap: wrap; gap: 24px; }
.chart-box { flex: 1 1 calc(50% - 12px); min-width: 280px; }
.chart-box h3 {
  font-size: 0.9rem; font-weight: 600; color: #475569;
  margin-bottom: 10px; text-transform: uppercase; letter-spacing: .05em;
}
svg { display: block; max-width: 100%; height: auto; overflow: visible; }
table.leaderboard { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.leaderboard thead tr {
  background: #f1f5f9; text-transform: uppercase;
  font-size: 0.75rem; letter-spacing: .05em; color: #64748b;
}
table.leaderboard th { padding: 10px 12px; text-align: left; font-weight: 600; }
table.leaderboard td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9; }
table.leaderboard tbody tr:hover { background: #f8fafc; }
td.rank { font-weight: 700; color: #2563eb; }
td.tracker-name { font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.highlight { font-weight: 700; color: #16a34a; }
footer {
  text-align: center; padding: 20px; font-size: 0.8rem; color: #94a3b8;
}
"""


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class HTMLReportGenerator:
    """Generate a self-contained HTML benchmark report from result dicts.

    No external dependencies are required at report-generation time: all
    charts are computed as inline SVG in pure Python.

    Args:
        title: Page title and ``<h1>`` heading.  Default:
            ``"EOVOT Benchmark Report"``.

    Example::

        from eovot.reporting.html_report import HTMLReportGenerator

        gen = HTMLReportGenerator(title="Classical Tracker Comparison")
        gen.save([mosse_dict, kcf_dict], "results/report.html")
    """

    def __init__(self, title: str = "EOVOT Benchmark Report") -> None:
        self.title = title

    def generate(self, results: List[Dict[str, Any]]) -> str:
        """Return the full HTML document as a string.

        Args:
            results: List of result dicts, each produced by
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

        Returns:
            A self-contained HTML string.  Write it to a ``.html`` file or
            serve it directly.
        """
        from datetime import datetime

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        tracker_names = [
            (r.get("summary", {}).get("tracker") or
             r.get("summary", {}).get("tracker_name", "?"))
            for r in results
        ]
        subtitle = (
            f"{len(results)} tracker(s): {', '.join(tracker_names)} &nbsp;·&nbsp; "
            f"Generated {now}"
        )

        success_svg = _build_success_curves(results)
        precision_svg = _build_precision_curves(results)
        scatter_svg = _build_fps_iou_scatter(results)
        latency_svg = _build_latency_bars(results)
        leaderboard_html = _build_leaderboard_html(results)

        has_latency = any(
            "mean_latency_ms" in seq
            for r in results
            for seq in r.get("sequences", [])
        )
        has_precision = any(
            "center_distances" in seq and seq["center_distances"]
            for r in results
            for seq in r.get("sequences", [])
        )

        charts_row = f"""
        <div class="charts-row">
          <div class="chart-box">
            <h3>Success Curves</h3>
            {success_svg}
          </div>
          {"<div class='chart-box'><h3>Precision Curves</h3>" + precision_svg + "</div>" if has_precision else ""}
        </div>
        <div class="charts-row" style="margin-top:24px">
          <div class="chart-box">
            <h3>FPS vs. Accuracy (Edge Deployment View)</h3>
            {scatter_svg}
          </div>
          {"<div class='chart-box'><h3>Latency Percentiles (p50 / p95 / p99)</h3>" + latency_svg + "</div>" if has_latency else ""}
        </div>
        """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(self.title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(self.title)}</h1>
  <p>{subtitle}</p>
</header>
<main>
  <section>
    <h2>Leaderboard</h2>
    {leaderboard_html}
  </section>
  <section>
    <h2>Performance Charts</h2>
    {charts_row}
  </section>
</main>
<footer>
  Generated by <strong>EOVOT</strong> &mdash; Edge-Optimized Visual Object Tracking Benchmark Suite
</footer>
</body>
</html>"""

    def save(
        self,
        results: List[Dict[str, Any]],
        path: str,
    ) -> Path:
        """Write the HTML report to *path* and return the resolved path.

        Parent directories are created automatically.

        Args:
            results: List of result dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            path:    Destination file path.  A ``.html`` extension is
                     appended when the path has none.

        Returns:
            The resolved :class:`pathlib.Path` that was written.

        Example::

            gen = HTMLReportGenerator()
            out = gen.save([mosse_dict, kcf_dict], "results/report")
            print(f"Report → {out}")
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".html")
        p.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.generate(results)
        p.write_text(html_content, encoding="utf-8")
        return p
