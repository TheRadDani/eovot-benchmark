# Edge Efficiency Scoring

EOVOT's central thesis is that visual tracking models must be evaluated not only
by accuracy, but by their **deployability on edge hardware**.  This page describes
the efficiency scoring system and Pareto trade-off analysis that form the
core research contribution of the framework.

## Motivation

A tracker that achieves 0.65 mIoU but requires 2 GB of RAM and runs at 5 FPS is
**not deployable** on a Raspberry Pi 4 or Jetson Nano.  Conversely, a tracker
with 0.50 mIoU that runs at 500 FPS with 80 MB of RAM is a viable edge solution.

Standard benchmarks rank trackers by accuracy alone.  EOVOT adds a complementary
axis: **composite edge efficiency score**.

## Efficiency Model

Three hardware dimensions are measured and normalised to `[0, 1]`:

```
fps_score    = min(fps / target_fps, 1.0)
memory_score = max(0, 1 − peak_memory_mb / max_memory_mb)
energy_score = max(0, 1 − energy_per_frame_mj / max_energy_mj)   # optional
```

These are combined into a weighted composite:

```
composite = (fps_score × w_fps + memory_score × w_mem + energy_score × w_energy)
            / (w_fps + w_mem + w_energy)
```

**Default weights:** `w_fps = 0.5`, `w_mem = 0.3`, `w_energy = 0.2`

Rationale: real-time throughput (FPS) is the most critical constraint on most
edge platforms, followed by memory (which determines which devices can run the
tracker at all), followed by energy (relevant for battery-constrained platforms).

## Edge Device Presets

| Device | `target_fps` | `max_memory_mb` | `max_energy_mj` | `tdp_watts` |
|--------|-------------|-----------------|-----------------|-------------|
| Raspberry Pi 4 | 15 | 512 | 2.0 | 6.0 |
| Jetson Nano | 30 | 3800 | 5.0 | 10.0 |
| Laptop CPU | 30 | 4096 | 10.0 | 15.0 |
| Desktop Workstation | 60 | 16384 | 20.0 | 65.0 |

## Quick Start

```python
from eovot.benchmark.engine import BenchmarkEngine
from eovot.metrics.efficiency import EdgeEfficiencyAnalyzer

# Run benchmark with energy profiling (set tdp_watts to your device)
engine = BenchmarkEngine(tdp_watts=10.0)  # Jetson Nano
results = [engine.run(tracker, dataset) for tracker in trackers]

# Score trackers against Jetson Nano hardware budget
analyzer = EdgeEfficiencyAnalyzer(
    target_fps=30.0,
    max_memory_mb=3800.0,
    max_energy_mj=5.0,
)
scores = analyzer.analyze(results)

# Print ranked table
print(analyzer.ranking_table(scores))
```

Example output:

```
| Rank | Tracker    | mIoU   | Efficiency | FPS Score | Mem Score | FPS   | Mem (MB) |
|------|------------|-------:|-----------:|----------:|----------:|------:|---------:|
| 1    | MOSSE      | 0.5023 | 0.9412     | 1.000     | 0.980     | 512.1 | 82.3     |
| 2    | KCF        | 0.5534 | 0.8756     | 1.000     | 0.960     | 215.4 | 98.1     |
| 3    | MedianFlow | 0.4489 | 0.7201     | 1.000     | 0.921     | 118.2 | 104.7    |
| 4    | CSRT       | 0.6502 | 0.5134     | 0.820     | 0.710     | 42.1  | 152.0    |
```

## Pareto Frontier Analysis

The Pareto frontier is the set of trackers that are **not dominated** in
both accuracy (mIoU) and efficiency (composite score).

A tracker A *dominates* B if:
- A's mIoU ≥ B's mIoU, **AND**
- A's composite score ≥ B's composite score, **AND**
- A is strictly better in at least one dimension

```python
frontier = analyzer.pareto_frontier(scores)

print("Pareto-optimal trackers for this device:")
for s in frontier:
    print(f"  {s.tracker_name:15s}  mIoU={s.mean_iou:.4f}  eff={s.composite_score:.4f}")
```

Interpretation:
- **If accuracy is the priority** → choose the Pareto tracker with the highest mIoU
- **If efficiency is the priority** → choose the Pareto tracker with the highest composite score
- Dominated trackers are never the right choice — there always exists a Pareto tracker that is better or equal in both dimensions

## Visualisation

```python
from eovot.visualization.pareto import plot_pareto_frontier, plot_efficiency_radar

# Scatter plot: mIoU vs efficiency, with Pareto frontier highlighted
plot_pareto_frontier(scores, frontier, output_path="results/pareto.png")

# Radar chart: per-tracker breakdown across all efficiency axes
plot_efficiency_radar(scores, output_path="results/radar.png")
```

### Pareto Scatter Plot

- **Coloured markers** = Pareto-optimal trackers
- **Grey markers** = dominated trackers (never the best choice)
- **Dashed line** = Pareto frontier boundary

### Radar Chart

Axes:
- **mIoU** — accuracy
- **FPS Score** — throughput relative to target
- **Memory Score** — how much memory headroom remains
- **Energy Score** — energy efficiency per frame (when enabled)

Larger area = more desirable overall.

## Running the Pre-built Experiment

```bash
python scripts/run_experiment.py \
    --config configs/experiments/efficiency_tradeoff.yaml
```

Edit the config to set your `dataset.root`, `tdp_watts`, and `efficiency.*`
hardware budgets before running.

## API Reference

### `EdgeEfficiencyAnalyzer`

```python
from eovot.metrics.efficiency import EdgeEfficiencyAnalyzer

analyzer = EdgeEfficiencyAnalyzer(
    target_fps=30.0,      # required FPS for real-time
    max_memory_mb=512.0,  # device RAM budget
    max_energy_mj=None,   # energy budget (None disables energy axis)
    fps_weight=0.5,
    memory_weight=0.3,
    energy_weight=0.2,
)

score     = analyzer.score(single_result)          # -> EdgeEfficiencyScore
scores    = analyzer.analyze(list_of_results)       # -> List[EdgeEfficiencyScore]
frontier  = analyzer.pareto_frontier(scores)        # -> List[EdgeEfficiencyScore]
table_md  = analyzer.ranking_table(scores)          # -> str (Markdown)
df        = analyzer.to_dataframe(scores)           # -> pandas.DataFrame
```

### `EdgeEfficiencyScore`

```python
s = scores[0]
print(s.tracker_name)         # "KCF"
print(s.mean_iou)             # 0.5534
print(s.composite_score)      # 0.8756
print(s.fps_score)            # 1.0
print(s.memory_score)         # 0.96
print(s.energy_score)         # 0.72   (0.0 if energy not profiled)
print(s.mean_fps)             # 215.4
print(s.peak_memory_mb)       # 98.1
print(s.energy_per_frame_mj)  # 1.44  (None if not profiled)
print(s.has_energy)           # True
```
