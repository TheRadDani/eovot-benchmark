"""Self-contained HTML benchmark report generator for EOVOT.

Produces a single ``.html`` file that bundles:

- A **multi-tracker comparison table** with sortable columns.
- An **overlap success-curve** plot (fraction of frames with IoU > threshold).
- An **efficiency scatter plot** — mean IoU on the x-axis, FPS on the y-axis,
  with optional bubble sizing by peak memory usage.
- A per-tracker **loss-rate section** if any tracker exposes
  :class:`~eovot.trackers.confidence.ConfidenceAdaptiveTracker` history.

All plots are generated with matplotlib (SVG backend) and embedded as inline
``<svg>`` blocks — no external assets, no CDN, no server required.  The
resulting HTML file is fully portable and renders identically in any modern
browser.

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.reporting.html_report import HTMLReportGenerator

    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    engine  = BenchmarkEngine(verbose=False)

    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
    ]

    gen = HTMLReportGenerator()
    path = gen.generate(results, "results/benchmark_report.html")
    print(f"Report written to {path}")

Matplotlib is an optional dependency.  When not installed, the report is
generated without plots but includes the full comparison table and summary.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# ---------------------------------------------------------------------------
# Colour palette — colour-blind-friendly, consistent with VOT paper style
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]


def _try_import_matplotlib():
    """Return (plt, io_module) or raise ImportError with a helpful message."""
    try:
        import matplotlib
        matplotlib.use("svg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
    return float(fn(y, x))


def _success_curve(ious: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, rates) for a success curve."""
    thresholds = np.linspace(0.0, 1.0, 101)
    rates = np.array([(ious >= t).mean() for t in thresholds])
    return thresholds, rates


def _collect_ious(result: "BenchmarkResult") -> np.ndarray:
    """Gather all per-frame IoU values from a BenchmarkResult."""
    parts = [r.ious for r in result.sequence_results if r.ious is not None and len(r.ious) > 0]
    if not parts:
        return np.array([result.mean_iou])
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# SVG figure generators
# ---------------------------------------------------------------------------

def _render_success_curves_svg(results: List["BenchmarkResult"]) -> Optional[str]:
    """Generate an SVG string of the multi-tracker success curve plot."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, result in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        ious = _collect_ious(result)
        thr, rates = _success_curve(ious)
        auc = _trapz(rates, thr)
        label = f"{result.tracker_name} [AUC={auc:.3f}]"
        ax.plot(thr, rates, color=colour, linewidth=2.0, label=label)

    ax.set_xlabel("IoU Threshold", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Overlap Success Curves", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_efficiency_scatter_svg(results: List["BenchmarkResult"]) -> Optional[str]:
    """Generate an SVG string of the IoU–FPS efficiency scatter plot.

    Bubble size is proportional to peak memory usage when available.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, result in enumerate(results):
        colour = _PALETTE[i % len(_PALETTE)]
        iou = result.mean_iou
        fps = result.mean_fps
        mem = result.peak_memory_mb

        # Bubble area proportional to peak memory; default size when unavailable.
        size = max(80.0, min(mem * 3.0, 800.0)) if mem > 0 else 120.0

        ax.scatter(fps, iou, s=size, color=colour, alpha=0.75,
                   edgecolors="white", linewidths=0.8, label=result.tracker_name,
                   zorder=3)
        ax.annotate(
            result.tracker_name,
            xy=(fps, iou),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )

    ax.set_xlabel("Mean FPS (higher is better →)", fontsize=10)
    ax.set_ylabel("Mean IoU (higher is better ↑)", fontsize=10)
    ax.set_title("Efficiency–Accuracy Trade-off", fontsize=12, fontweight="bold")
    note = "(Bubble size ∝ peak memory usage)"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="#888888")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(bottom=max(0.0, min(r.mean_iou for r in results) - 0.1))
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML assembler
# ---------------------------------------------------------------------------

_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8f9fa;
    color: #212529;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }
  h1 { font-size: 1.8rem; margin-bottom: 0.4rem; }
  h2 { font-size: 1.2rem; margin: 1.8rem 0 0.7rem; color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 0.3rem; }
  .meta { color: #6c757d; font-size: 0.85rem; margin-bottom: 1.5rem; }
  .figures { display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .figures > div { flex: 1; min-width: 320px; background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; }
  .figures svg { width: 100%; height: auto; }
  table { border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }
  thead { background: #343a40; color: #fff; }
  th { padding: 0.6rem 0.9rem; font-size: 0.85rem; text-align: left; white-space: nowrap; cursor: pointer; user-select: none; }
  th:hover { background: #495057; }
  th .sort-arrow { margin-left: 4px; opacity: 0.5; font-size: 0.7rem; }
  th.sorted-asc .sort-arrow::after { content: "▲"; opacity: 1; }
  th.sorted-desc .sort-arrow::after { content: "▼"; opacity: 1; }
  th .sort-arrow::after { content: "⇅"; }
  td { padding: 0.5rem 0.9rem; font-size: 0.85rem; border-bottom: 1px solid #f1f3f5; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f8f9fa; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge-good  { background: #d4edda; color: #155724; }
  .badge-ok    { background: #fff3cd; color: #856404; }
  .badge-poor  { background: #f8d7da; color: #721c24; }
  .no-plots { background: #fff3cd; border: 1px solid #ffc107; padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.85rem; color: #856404; margin-bottom: 1rem; }
  footer { margin-top: 2rem; color: #adb5bd; font-size: 0.78rem; text-align: center; }
</style>
"""

_SORT_JS = """
<script>
(function () {
  function getCellValue(row, idx) {
    var text = row.children[idx].textContent.trim();
    return isNaN(text) || text === '' ? text.toLowerCase() : parseFloat(text);
  }
  function sortTable(th) {
    var table = th.closest('table');
    var tbody = table.querySelector('tbody');
    var ths = Array.from(th.parentElement.children);
    var idx = ths.indexOf(th);
    var asc = !th.classList.contains('sorted-asc');
    ths.forEach(function(h) { h.classList.remove('sorted-asc', 'sorted-desc'); });
    th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {
      var va = getCellValue(a, idx), vb = getCellValue(b, idx);
      return (va > vb ? 1 : va < vb ? -1 : 0) * (asc ? 1 : -1);
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
  }
  document.querySelectorAll('th[data-sortable]').forEach(function(th) {
    th.innerHTML += '<span class="sort-arrow"></span>';
    th.addEventListener('click', function() { sortTable(th); });
  });
})();
</script>
"""


def _iou_badge(iou: float) -> str:
    if iou >= 0.55:
        cls = "badge-good"
    elif iou >= 0.35:
        cls = "badge-ok"
    else:
        cls = "badge-poor"
    return f'<span class="badge {cls}">{iou:.3f}</span>'


def _build_summary_table(results: List["BenchmarkResult"]) -> str:
    headers = [
        ("Tracker", True),
        ("Dataset", False),
        ("Sequences", True),
        ("mIoU", True),
        ("Success AUC", True),
        ("Precision AUC", True),
        ("FPS", True),
        ("Latency (ms)", True),
        ("Memory (MB)", True),
        ("Energy (J)", True),
    ]

    header_html = "".join(
        f'<th {"data-sortable" if sortable else ""}>{name}</th>'
        for name, sortable in headers
    )

    rows_html = ""
    for result in results:
        s = result.mean_iou
        sauc = result.mean_success_auc
        pauc = result.mean_precision_auc
        fps = result.mean_fps
        lat = float(np.mean([r.profiling.latency_mean_ms for r in result.sequence_results]))
        mem = result.peak_memory_mb
        energy = result.total_energy_j

        sauc_str = f"{sauc:.4f}" if sauc is not None else "—"
        pauc_str = f"{pauc:.4f}" if pauc is not None else "—"
        energy_str = f"{energy:.4f}" if energy is not None else "—"

        rows_html += (
            f"<tr>"
            f"<td><strong>{result.tracker_name}</strong></td>"
            f"<td>{result.dataset_name}</td>"
            f"<td>{len(result.sequence_results)}</td>"
            f"<td>{_iou_badge(s)}</td>"
            f"<td>{sauc_str}</td>"
            f"<td>{pauc_str}</td>"
            f"<td>{fps:.1f}</td>"
            f"<td>{lat:.2f}</td>"
            f"<td>{mem:.1f}</td>"
            f"<td>{energy_str}</td>"
            f"</tr>\n"
        )

    return f"""
<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>
{rows_html}  </tbody>
</table>
"""


def _build_per_sequence_table(results: List["BenchmarkResult"]) -> str:
    """Per-sequence breakdown table showing individual sequence mIoU and FPS."""
    headers = [
        ("Tracker", False),
        ("Sequence", False),
        ("mIoU", True),
        ("FPS", True),
        ("Latency (ms)", True),
    ]
    header_html = "".join(
        f'<th {"data-sortable" if sortable else ""}>{name}</th>'
        for name, sortable in headers
    )

    rows_html = ""
    for result in results:
        for seq in result.sequence_results:
            rows_html += (
                f"<tr>"
                f"<td>{result.tracker_name}</td>"
                f"<td>{seq.sequence_name}</td>"
                f"<td>{_iou_badge(seq.mean_iou)}</td>"
                f"<td>{seq.profiling.fps:.1f}</td>"
                f"<td>{seq.profiling.latency_mean_ms:.2f}</td>"
                f"</tr>\n"
            )

    return f"""
<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>
{rows_html}  </tbody>
</table>
"""


class HTMLReportGenerator:
    """Generate a self-contained HTML benchmark comparison report.

    The report embeds all plots as inline SVG and all styling as inline CSS
    so the resulting ``.html`` file can be opened directly in any browser,
    emailed, or committed to a repository without any companion files.

    Args:
        output_dir: Directory where the HTML file is written.  Created
            automatically if it does not exist.  Default: ``"results/"``.
    """

    def __init__(self, output_dir: str = "results/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List["BenchmarkResult"],
        filename: str = "benchmark_report.html",
        title: str = "EOVOT Benchmark Report",
    ) -> Path:
        """Generate the HTML report and write it to disk.

        Args:
            results:  List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects — one per (tracker, dataset) combination.
            filename: Output filename (relative to :attr:`output_dir`).
                A ``.html`` extension is appended if absent.
            title:    Title displayed in the browser tab and page heading.

        Returns:
            :class:`pathlib.Path` of the written HTML file.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results list must not be empty.")

        p = Path(filename)
        if not p.suffix:
            p = p.with_suffix(".html")
        if not p.is_absolute():
            p = self.output_dir / p

        html = self._build_html(results, title)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_html(self, results: List["BenchmarkResult"], title: str) -> str:
        import datetime

        mpl_available = _try_import_matplotlib() is not None
        no_plots_notice = ""
        if not mpl_available:
            no_plots_notice = (
                '<div class="no-plots">'
                "⚠ matplotlib is not installed — plots are omitted. "
                "Install it with <code>pip install matplotlib</code> to include figures."
                "</div>"
            )

        success_svg = _render_success_curves_svg(results) if mpl_available else None
        scatter_svg = _render_efficiency_scatter_svg(results) if mpl_available else None

        figures_html = ""
        if success_svg or scatter_svg:
            parts = []
            if success_svg:
                parts.append(f"<div><h3 style='font-size:0.95rem;margin-bottom:0.5rem'>Success Curves</h3>{success_svg}</div>")
            if scatter_svg:
                parts.append(f"<div><h3 style='font-size:0.95rem;margin-bottom:0.5rem'>Efficiency Scatter</h3>{scatter_svg}</div>")
            figures_html = f'<div class="figures">{"".join(parts)}</div>'

        summary_table = _build_summary_table(results)
        seq_table = _build_per_sequence_table(results)

        dataset_names = ", ".join(sorted({r.dataset_name for r in results}))
        tracker_names = ", ".join(r.tracker_name for r in results)
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <title>{title}</title>
          {_CSS}
        </head>
        <body>
          <h1>{title}</h1>
          <p class="meta">
            Generated: {now} &nbsp;|&nbsp;
            Trackers: {tracker_names} &nbsp;|&nbsp;
            Datasets: {dataset_names}
          </p>

          {no_plots_notice}
          {figures_html}

          <h2>Tracker Comparison</h2>
          {summary_table}

          <h2>Per-Sequence Breakdown</h2>
          {seq_table}

          <footer>
            Generated by <strong>EOVOT Benchmark Suite</strong>
            &mdash; Edge-Optimized Visual Object Tracking
          </footer>

          {_SORT_JS}
        </body>
        </html>
        """)
        return html
