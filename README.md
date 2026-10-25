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

SPHT settings are fixed per experiment:

| Experiment | Initial factor / offset | Extension factor | Sample factor | Validation factor / min | Attempts | Iteration cap | LS candidates |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Phase, N ≤ 2¹⁰ | 2.0 / 8 | 0.5 | 2.4 | 0 / 0 | 1 | 6,000 | k |
| Phase, N ≥ 2¹² | 2.0 / 10 | 0.5 | 2.4 | 0 / 0 | 1 | 6,000 | k |
| Graphs (BCH) | 1.5 / 0 | 0.75 | 1.5 | 0.25 / 8 | 2 | 6,000 | k |
| Scaling, unrestricted | 1.0 / 6 | 1.5 | 0.8 | 0 / 5 | 6 | 300 | 1.5k |
| Scaling, degree-limited (BCH) | 2.0 / 1 | 0.75 | 0.8 | 0 / 2 | 6 | 300 | 1.5k |

At least one validation query is always reserved, and samples are reused across prefix stages. A stage that fails validation draws a fresh batch of samples and refits on all samples collected so far, so a small sample factor with several attempts spends queries adaptively. The iteration cap bounds each ADMM solve, so an under-sampled attempt fails fast instead of running to the 6,000-iteration evaluation limit. With 1.5k least-squares candidates, the stage fit uses the top-k ADMM coefficients plus the columns most correlated with the ADMM residual, then refits on the top k. This recovers coefficients that ADMM leaves at zero.

The scaling settings were tuned on a separate seed split (`tuning-v1`, 20 repetitions per configuration, disjoint from the evaluation seeds) by choosing the setting with no recovery failures and the fewest queries relative to CS. CS keeps its sample factor of 2.4. The phase and graph settings were not tuned.

## Results

Scaling is complete (560 trials). Exact recovery rates:

| Signal class | CS | SPHT |
| --- | ---: | ---: |
| Unrestricted, n ≤ 20 | 100/100 | 100/100 |
| Degree 2, n ≤ 800 | 90/90 | 90/90 |
| Degree 3, n ≤ 800 | 58/90 (30 over the 2M-column limit, 2 timeouts) | 90/90 |

Geometric-mean CPU speedup of SPHT over CS on jointly successful pairs, with SPHT's query count relative to CS in parentheses:

| | n = 30 | n = 60 | n = 100 | n = 200 | n = 400 | n = 800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d = 2, k = 8 | 2.2× (0.90) | 3.3× (0.89) | 6.1× (1.02) | 32× (0.89) | 102× (0.90) | 503× (0.87) |
| d = 2, k = 32 | 1.3× (0.87) | 1.9× (0.87) | 3.6× (0.84) | 26× (0.80) | 104× (0.77) | 457× (0.74) |
| d = 3, k = 8 | 3.1× (1.07) | 52× (0.99) | 360× (0.90) | 2,960× (0.95) | — | — |
| d = 3, k = 32 | 2.4× (0.94) | 41× (0.85) | 187× (0.85) | 7,610× (0.84) | — | — |

| | N = 2¹² | N = 2¹⁴ | N = 2¹⁶ | N = 2¹⁸ | N = 2²⁰ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unrestricted, k = 4 | 6.0× (0.96) | 27× (1.01) | 76× (0.94) | 620× (0.87) | 1,790× (0.91) |
| Unrestricted, k = 32 | 0.95× (0.78) | 6.4× (0.69) | 25× (0.68) | 72× (0.68) | 104× (0.82) |

SPHT uses fewer queries than CS in 47 of 50 configurations. The three exceptions (ratios 1.01–1.07) have 95% intervals that include 1, as does the one runtime ratio below 1 (k = 32, N = 2¹²: 0.44–1.63×). Per-configuration values are in `results/scaling/paired-comparisons.csv`. The untuned scaling run is kept in `results/archive/scaling-untuned.sqlite`.

The phase and graph suites are still running, and robustness and HPO are deferred.

## Acknowledgements

The SWHT and Stobbe baselines, graph instances and published settings come from [Amrollahi et al., *Efficiently Learning Fourier Sparse Set Functions*](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions). CS uses [SPGL1](https://github.com/drrelyea/spgl1). The phase-diagram reference curve follows [Donoho and Tanner (2009)](https://arxiv.org/abs/0906.2530).
