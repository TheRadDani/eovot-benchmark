# Edge Efficiency Score & Pareto-Front Analysis

EOVOT's defining contribution is evaluating trackers on **both accuracy and efficiency simultaneously**.  This document explains the Edge Efficiency Score (EES), Pareto-front identification, and how to use them in experiments.

---

## Why Efficiency Metrics Matter

Classical VOT benchmarks (OTB, GOT-10k, LaSOT) rank trackers by accuracy only.  A tracker that achieves 0.68 IoU at 3 FPS dominates the leaderboard while being useless on a Raspberry Pi.

EOVOT's efficiency metrics make deployment constraints first-class citizens of evaluation.

---

## Edge Efficiency Score (EES)

The EES is a single scalar that rewards **high accuracy at high throughput within an acceptable memory envelope**:

```
EES = mean_iou × log1p(fps) / (1 + peak_memory_mb / memory_budget_mb)
```

### Components

| Term | Role |
|------|------|
| `mean_iou` | Accuracy — the tracking quality signal |
| `log1p(fps)` | Throughput with **diminishing returns**: 5→50 FPS matters far more than 500→550 FPS on edge hardware |
| `1 + peak_memory_mb / memory_budget_mb` | Memory penalty — exceeding the budget reduces the score but does not hard-cut a tracker |

### Interpretation

- **Higher EES = better edge suitability**
- A tracker scoring 1.8 is a better edge deployment choice than one scoring 2.4 with 10× the memory
- The default `memory_budget_mb = 512.0` targets constrained devices (Raspberry Pi 4, Jetson Nano)

---

## Pareto-Front Analysis

The Pareto front identifies trackers that **no other tracker dominates in both objectives**:

> Tracker A dominates B iff `A.mean_iou ≥ B.mean_iou` AND `A.ees ≥ B.ees` with at least one strict inequality.

Pareto-optimal trackers represent the **true accuracy–efficiency frontier** — the set of operating points where improving one objective requires sacrificing the other.

### Example

| Tracker | mIoU | FPS | Mem (MB) | EES | Pareto |
|---------|-----:|----:|---------:|----:|:------:|
| MOSSE   | 0.41 | 487 | 48 | 2.18 | ✓ |
| KCF     | 0.55 | 182 | 52 | 2.11 | ✓ |
| CSRT    | 0.68 | 34  | 59 | 1.81 | |

MOSSE and KCF are both Pareto-optimal (MOSSE is faster, KCF is more accurate — neither dominates the other). CSRT is dominated because KCF achieves comparable EES while being far faster.

---

## Usage

### Python API

```python
from eovot.benchmark.engine import BenchmarkEngine
from eovot.metrics.efficiency import EfficiencyMetricsEngine
from eovot.reporting.visualizer import BenchmarkVisualizer

# Run benchmark for each tracker
engine = BenchmarkEngine()
results = [engine.run(tracker, dataset) for tracker in trackers]

# Compute EES and Pareto front
eff = EfficiencyMetricsEngine(memory_budget_mb=512.0)
ranking = eff.rank_trackers(results)

# Print ranked table
print(eff.to_markdown_table(ranking))

# Inspect Pareto-optimal trackers
for entry in ranking:
    if entry.on_pareto_front:
        print(f"{entry.tracker_name}: EES={entry.ees:.4f}")

# Save efficiency-accuracy scatter plot
viz = BenchmarkVisualizer(output_dir="results/plots")
viz.plot_efficiency_accuracy(results, filename="efficiency_tradeoff.png")
```

### Efficiency-Accuracy Scatter Plot

`plot_efficiency_accuracy()` generates EOVOT's signature visualisation:
- **X-axis**: mean FPS (efficiency)
- **Y-axis**: mean IoU (accuracy)
- **Bubble area**: proportional to peak memory footprint
- **Bold outline**: Pareto-optimal trackers

This is the plot to include in papers and reports to communicate the accuracy–efficiency trade-off.

---

## Configuration

```python
# Adjust memory budget for your target device
eff = EfficiencyMetricsEngine(memory_budget_mb=2048.0)   # Laptop / workstation
eff = EfficiencyMetricsEngine(memory_budget_mb=512.0)    # Raspberry Pi 4 (default)
eff = EfficiencyMetricsEngine(memory_budget_mb=256.0)    # Ultra-constrained MCU
```

---

## API Reference

### `EfficiencyMetricsEngine`

| Method | Description |
|--------|-------------|
| `edge_efficiency_score(mean_iou, fps, peak_memory_mb)` | Compute EES for a single (accuracy, efficiency) point |
| `compute_pareto_front(entries)` | Mark Pareto-optimal entries in-place |
| `rank_trackers(results)` | Build sorted `EfficiencyEntry` list from `BenchmarkResult` objects |
| `to_markdown_table(entries)` | Format ranked table as Markdown string |

### `EfficiencyEntry`

| Field | Type | Description |
|-------|------|-------------|
| `tracker_name` | str | Tracker identifier |
| `dataset_name` | str | Dataset name |
| `mean_iou` | float | Mean IoU across all frames |
| `fps` | float | Mean throughput |
| `peak_memory_mb` | float | Peak RSS memory |
| `ees` | float | Edge Efficiency Score |
| `on_pareto_front` | bool | True if Pareto-optimal |
