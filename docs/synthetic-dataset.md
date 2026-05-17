# Synthetic Dataset Generator

`SyntheticDataset` generates tracking sequences programmatically — no external data download required.  It is the primary entry point for new contributors and the backbone of EOVOT's CI integration tests.

---

## Overview

Each sequence renders a **coloured filled rectangle** (the target) moving against a **static noise background**.  The target colour and background texture are unique per sequence; every sequence is fully reproducible via a seed.

**Three motion patterns are available:**

| Pattern | Description | Use case |
|---------|-------------|----------|
| `"linear"` | Constant-velocity drift with wall-bounce | Baseline tracking evaluation |
| `"circular"` | Target orbits the frame centre (3 full rotations) | Smooth curved-motion evaluation |
| `"random"` | Gaussian random walk clamped to frame boundaries | Unpredictable motion stress test |

---

## Quick Start

```python
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.mosse import MOSSETracker

dataset = SyntheticDataset(
    num_sequences=10,
    num_frames=100,
    motion="linear",
    seed=42,
)

engine = BenchmarkEngine(verbose=True)
result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-Linear")
print(result)
# BenchmarkResult[MOSSE on Synthetic-Linear] mIoU=0.64  FPS=421.3  mem=52.1 MiB  (10 sequences)
```

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_sequences` | int | 10 | Number of sequences to generate |
| `num_frames` | int | 100 | Frames per sequence |
| `frame_size` | (W, H) | (320, 240) | Frame dimensions in pixels |
| `bbox_size` | (W, H) | (40, 40) | Target rectangle size in pixels |
| `motion` | str | `"linear"` | Motion pattern: `"linear"`, `"circular"`, or `"random"` |
| `seed` | int | 42 | Base RNG seed; sequence `i` uses `seed + i` |

---

## Motion Patterns

### Linear

Constant-velocity drift with wall-bounce.  Velocity is randomly sampled per sequence.

```python
ds = SyntheticDataset(motion="linear", num_frames=200, seed=0)
```

### Circular

Target completes 3 full orbits around the frame centre.  Useful for evaluating trackers on smooth curved motion without abrupt direction changes.

```python
ds = SyntheticDataset(motion="circular", num_frames=90, seed=7)
```

### Random

Gaussian random walk: at each frame the target moves by a random step in both axes, clamped to stay within the frame.

```python
ds = SyntheticDataset(motion="random", num_frames=100, seed=13)
```

---

## Reproducibility

Sequences are deterministic:

```python
ds1 = SyntheticDataset(num_sequences=3, num_frames=50, seed=99)
ds2 = SyntheticDataset(num_sequences=3, num_frames=50, seed=99)

import numpy as np
for f1, f2 in zip(ds1[0], ds2[0]):
    np.testing.assert_array_equal(f1, f2)  # identical frames
```

Different seeds produce different sequences:
```python
ds_a = SyntheticDataset(seed=1)
ds_b = SyntheticDataset(seed=2)
# ds_a[0] and ds_b[0] will have different backgrounds and target colours
```

---

## Using in Experiment Configs

`SyntheticDataset` is registered in `ExperimentRunner` and can be used from YAML configs:

```yaml
# configs/experiments/synthetic_demo.yaml
dataset:
  loader: SyntheticDataset
  name: Synthetic-Linear
  motion: linear
  num_sequences: 10
  num_frames: 100
  frame_size: [320, 240]
  bbox_size: [40, 40]
  seed: 42

trackers:
  - name: MOSSE
    params: {}
  - name: KCF
    params: {}
```

Run with:
```bash
python scripts/run_experiment.py --config configs/experiments/synthetic_demo.yaml
```

A complete demo config is included at `configs/experiments/synthetic_demo.yaml`.

---

## Design Notes

### In-Memory Frames (`_InMemorySequence`)

Frames are stored as numpy arrays — no disk I/O during iteration.  This makes synthetic sequences much faster than file-based ones, which is ideal for tests and quick experiments.

### Lazy Generation + Caching

Sequences are rendered on first access (`ds[i]`) and cached for the lifetime of the dataset object.  Multiple accesses to the same index return the same Python object:

```python
seq_a = ds[0]
seq_b = ds[0]
assert seq_a is seq_b  # True — cached
```

### Target Visibility

The background intensity is sampled uniformly in [40, 100] and the target colour in [160, 256] per channel.  This guarantees sufficient contrast for classical correlation filters (MOSSE, KCF) to track successfully.

---

## Integration Tests

`tests/test_synthetic_dataset.py` provides 39 tests covering the full pipeline without any external data:

```bash
pytest tests/test_synthetic_dataset.py -v
```

These tests are included in CI and serve as regression guards for the benchmark loop.
