# Sparse Hadamard Experiments

Code for the experiments comparing **SPHT** (ours) with ℓ1 compressed sensing (CS) and the authors' original SWHT and Stobbe implementations for recovering Fourier-sparse set functions from oracle queries. Every trial is deterministic given its seed and is cached, so runs can be interrupted and resumed and figures can be regenerated without recomputation.

## Installation

Graph and scaling experiments use Python 3.12; phase diagrams (and the deferred robustness and HPO suites) use Python 3.14. Both environments are pinned:

```bash
python3.12 -m venv .venv-bench && .venv-bench/bin/pip install -r requirements.txt
python3.14 -m venv .venv && .venv/bin/pip install -r requirements-phase.txt
.venv-bench/bin/python main.py fetch
```

`fetch` downloads the baseline repository at commit [`4c3e161`](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/tree/4c3e161e13062a356afcba1f41d0493b0373336c) into `data/` and builds its Hadamard extension, which needs a C++ compiler and Python headers.

## Reproducing the results

```bash
.venv-bench/bin/python main.py start    # phase, graphs and scaling in tmux sessions sht-phase, sht-graphs, sht-scaling
.venv-bench/bin/python main.py status   # progress of every suite
.venv-bench/bin/python main.py plot --suite graphs
.venv-bench/bin/python main.py stop --suite graphs
```

`main.py run --suite NAME` runs a suite in the foreground; `--suite robust` and `--suite hpo` run the deferred suites. `run` and `plot` always execute in the suite's own environment, so figures are rendered with the pinned matplotlib. Each suite is pinned to one CPU core (phase on the last core, graphs on the second-to-last, the others share the fourth-to-last and take turns) and runs trials sequentially with one numerical thread.

Results go to `results/<suite>/cache.sqlite`, which holds every finished trial, every oracle response and the solver checkpoints of interrupted trials. A suite refuses to resume if its settings, Python or package versions, or CPU model changed. When a suite completes, the cache is audited against the planned trial list, and PNG/PDF figures are written to `results/figures/` along with `summary.csv` (and, for graphs and scaling, `coverage.csv` and `paired-comparisons.csv`) in the suite directory.

## Experiments

| Suite | Setup | Repetitions | Trials | Figures |
| --- | --- | ---: | ---: | --- |
| Phase | N = 2⁸, 2¹⁰ with Q/N ∈ [0.10, 0.95], k/Q ∈ [0.05, 0.95], Rademacher and Gaussian; N = 2¹², 2¹⁴, 2¹⁶ with Q/N ∈ [0.005, 0.08], k/Q ∈ [0.025, 0.30], Rademacher | 10 per cell | 33,120 | `phase_diagrams`, `phase_sparse` |
| Graphs | Authors' snapshots 1, 3, 5, 7; 100 vertices with 20–70 edges; 50 edges with 50–800 vertices | 20 per instance | 8,240 | `graphs`, `graph_recovery` |
| Scaling | Unrestricted n ∈ {12,…,20}, k ∈ {4, 8, 16, 32}; degree d ∈ {2, 3} with n ∈ {30,…,800}, k ∈ {8, 16, 32} | 5 per configuration | 560 | `scaling_runtime`, `scaling_queries` |

**Phase diagrams.** Both methods get a budget of Q = δN unique queries for a k = ρQ sparse signal. Recovery succeeds when the relative coefficient error is at most 10⁻⁶. CS solves basis pursuit with the same compiled ADMM solver that SPHT uses for its local problems. Heatmaps show empirical recovery probability with bilinear display interpolation and the 50% contour; the black curve is the Donoho–Tanner Gaussian asymptote, shown for reference only.

**Graphs.** Learning a graph cut function: the Fourier support is the constant term plus one coefficient per edge. SWHT and Stobbe run the authors' unmodified code. SWHT uses the fastest published setting that reaches 90% recovery, Stobbe uses the first published C that does, and Stobbe is only evaluated up to the published 210 vertices. CS uses [SPGL1 0.0.3](https://github.com/drrelyea/spgl1) basis pursuit (`bp_tol = ls_tol = 1e-6`, `opt_tol = 1e-4`) on ⌈2.4 (|E|+1) log₂ n⌉ random queries over the constant-plus-pairs dictionary, followed by top-k least-squares debiasing. Success uses the original `DataGraph` equality criterion. Repetitions vary the algorithms' randomness on the released fixed graphs.

**Scaling.** Random Gaussian-coefficient k-sparse signals, either over all 2ⁿ frequencies or over frequencies of degree at most d. SPHT uses the BCH syndrome reduction for degree-limited signals. CS uses SPGL1 with ⌈2.4 k ln(max(e, D/k))⌉ queries, where D is the dictionary size, which is capped at 2 million columns. Cost plots use only jointly successful CS/SPHT pairs; `coverage.csv` reports recovery rates, timeouts and resource limits for all trials.

**Measurement.** Every method gets the same 1,800 s wall-time cap per trial. Queries are unique oracle evaluations and include all validation and retry samples. Runtime is process CPU time minus oracle and cache I/O time, measured after compiled kernels, dictionaries and BCH maps are prepared; BCH preparation time is recorded separately. Graph and scaling cost plots show medians with 95% bootstrap intervals (10,000 draws, resampled within snapshot for graphs), and recovery rates use Wilson intervals.

SPHT settings are fixed per experiment and were not tuned on these evaluations:

| Experiment | Initial factor / offset | Extension factor | Sample factor | Validation factor / min | Attempts |
| --- | --- | ---: | ---: | --- | ---: |
| Phase, N ≤ 2¹⁰ | 2.0 / 8 | 0.5 | 2.4 | 0 / 0 | 1 |
| Phase, N ≥ 2¹² | 2.0 / 10 | 0.5 | 2.4 | 0 / 0 | 1 |
| Graphs (BCH) | 1.5 / 0 | 0.75 | 1.5 | 0.25 / 8 | 2 |
| Scaling, unrestricted | 2.0 / 8 | 0.5 | 2.4 | 0 / 0 | 1 |
| Scaling, degree-limited (BCH) | 2.0 / 1 | 1.0 | 2.4 | 0.25 / 8 | 2 |

At least one validation query is always reserved. Unrestricted SPHT reuses samples across prefix stages.

## Results

Scaling is complete (560 trials). Exact recovery rates:

| Signal class | CS | SPHT |
| --- | ---: | ---: |
| Unrestricted, n ≤ 20 | 100/100 | 90/100 |
| Degree 2, n ≤ 800 | 90/90 | 90/90 |
| Degree 3, n ≤ 800 | 58/90 (30 over the 2M-column limit, 2 timeouts) | 90/90 |

Geometric-mean CPU speedup of SPHT over CS on jointly successful pairs, with SPHT's query count relative to CS in parentheses:

| | n = 30 | n = 60 | n = 100 | n = 200 | n = 400 | n = 800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d = 2, k = 8 | 4.5× (1.39) | 3.8× (1.47) | 7.6× (1.38) | 34× (1.41) | 178× (1.32) | 1,022× (1.35) |
| d = 2, k = 32 | 3.2× (1.33) | 3.0× (1.33) | 6.5× (1.27) | 43× (1.22) | 153× (1.23) | 671× (1.16) |
| d = 3, k = 8 | 6.0× (1.52) | 95× (1.43) | 439× (1.50) | 3,739× (1.46) | — | — |
| d = 3, k = 32 | 4.6× (1.44) | 63× (1.37) | 307× (1.33) | 11,934× (1.31) | — | — |

For unrestricted signals with k = 4, the speedup grows from 0.5× at N = 2¹² to 216× at N = 2²⁰ with at most 7% more queries; at k = 8 it reaches 36× at N = 2²⁰. At k = 32, CS is as fast or faster at these sizes (0.2–1.5×). Per-configuration values are in `results/scaling/paired-comparisons.csv`.

The phase and graph suites are still running, and robustness and HPO are deferred.

## Acknowledgements

The SWHT and Stobbe baselines, graph instances and published settings come from [Amrollahi et al., *Efficiently Learning Fourier Sparse Set Functions*](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions). CS uses [SPGL1](https://github.com/drrelyea/spgl1). The phase-diagram reference curve follows [Donoho and Tanner (2009)](https://arxiv.org/abs/0906.2530).
