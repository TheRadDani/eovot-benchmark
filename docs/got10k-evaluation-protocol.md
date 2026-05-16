# GOT-10k Evaluation Protocol

EOVOT implements the official GOT-10k evaluation protocol as defined in:

> Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild." **IEEE TPAMI 2021**.

## Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Average Overlap | **AO** | Mean IoU across all frames except the initialisation frame |
| Success Rate (50%) | **SR₅₀** | Fraction of frames with IoU ≥ 0.5 |
| Success Rate (75%) | **SR₇₅** | Fraction of frames with IoU ≥ 0.75 — discriminates high-accuracy trackers |

The initialisation frame (frame 0) is **always excluded** from AO and SR computation, because the tracker is given the ground-truth bounding box at initialisation and the IoU is trivially 1.0 by construction.

## Quick Start

```python
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.got10k import GOT10kDataset
from eovot.trackers.kcf import KCFTracker
from eovot.metrics.got10k_eval import GOT10kEvaluator

# 1. Run the benchmark
dataset   = GOT10kDataset("/data/GOT-10k", split="val")
engine    = BenchmarkEngine(verbose=True)
result    = engine.run(KCFTracker(), dataset, dataset_name="GOT-10k-val")

# 2. Evaluate with the GOT-10k protocol
evaluator = GOT10kEvaluator(split="val")
report    = evaluator.evaluate(result)
print(report)
# GOT-10k [KCF / val] AO=0.4123  SR50=0.3891  SR75=0.1540  (180 sequences)

# 3. Per-sequence breakdown
for s in report.per_sequence[:5]:
    print(s)
```

## Multi-Tracker Comparison Table

```python
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.got10k import GOT10kDataset
from eovot.metrics.got10k_eval import GOT10kEvaluator
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker

dataset   = GOT10kDataset("/data/GOT-10k", split="val")
engine    = BenchmarkEngine()
evaluator = GOT10kEvaluator(split="val")

trackers = [MOSSETracker(), KCFTracker()]
reports  = []
for tracker in trackers:
    result = engine.run(tracker, dataset, dataset_name="GOT-10k-val")
    reports.append(evaluator.evaluate(result))

print(evaluator.to_markdown_table(reports))
```

Output:

```
| Rank | Tracker | AO     | SR₅₀   | SR₇₅   | Sequences |
|------|---------|-------:|-------:|-------:|----------:|
| 1    | KCF     | 0.4123 | 0.3891 | 0.1540 | 180       |
| 2    | MOSSE   | 0.3421 | 0.3012 | 0.0987 | 180       |
```

## Official Submission Format Export

Results can be exported in the format accepted by the GOT-10k evaluation server at http://got-10k.aitestunion.com/

```python
evaluator.export_submission(report, result, output_dir="submissions/")
```

This creates:
```
submissions/
  KCF/
    GOT-10k_Val_000001.txt   # one x,y,w,h box per line (frames 2..N)
    GOT-10k_Val_000002.txt
    ...
    got10k_report.json       # AO/SR50/SR75 summary
```

Each TXT file contains predictions starting from **frame 2** (frame 1 is initialization).

## API Reference

### `compute_ao(ious)`

```python
from eovot.metrics.got10k_eval import compute_ao
ao = compute_ao(ious_array)  # excludes ious[0]
```

### `compute_sr(ious, threshold=0.5)`

```python
from eovot.metrics.got10k_eval import compute_sr
sr50 = compute_sr(ious_array, 0.5)
sr75 = compute_sr(ious_array, 0.75)
```

### `GOT10kEvaluator`

```python
from eovot.metrics.got10k_eval import GOT10kEvaluator

evaluator = GOT10kEvaluator(split="val")
report    = evaluator.evaluate(benchmark_result)   # -> GOT10kReport
table     = evaluator.to_markdown_table([r1, r2])  # -> str
path      = evaluator.save_report(report, "out/")  # -> Path
submit_dir = evaluator.export_submission(report, result, "submit/")  # -> Path
```

## Notes on GOT-10k Test Split

The GOT-10k test set ground-truth annotations are **withheld** by the evaluation server. To run on the test split:

1. Run the benchmark with `split="test"` — this will only succeed if you have the test annotations, which are not publicly released.
2. Use `export_submission()` to generate the prediction TXT files.
3. Zip the `<tracker_name>/` directory and upload to http://got-10k.aitestunion.com/

For local development, always use `split="val"`.
