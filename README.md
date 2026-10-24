# Sparse Hadamard Experiments

Experiments comparing **SPHT (Ours)** with compressed sensing and the released SWHT and Stobbe implementations.

## Structure

```text
main.py
requirements.lock
requirements-phase.lock
src/
    config.py
    experiments.py
    plotting.py
    hpo.py
    data.py
    numerics.py
    methods/
        spht.py
        cs.py
        swht.py
        stobbe.py
```

`main.py` provides the command-line interface and experiment runner. `src/config.py` holds settings, `experiments.py` generates and evaluates trials, and `plotting.py` produces figures. `data.py` handles datasets, original-source verification, caching, and measurement scheduling. `numerics.py` contains shared transforms, operators, solvers, and recovery metrics. `hpo.py` contains the HPO experiments. Each recovery method has one file under `src/methods/`; SPHT includes its BCH helper.

## Setup

Graph and scaling comparisons use one pinned Python 3.12 environment for every method:

```bash
python3.12 -m venv .venv-bench
.venv-bench/bin/python -m pip install -r requirements.lock
.venv-bench/bin/python main.py fetch
```

Preparation downloads checksum-verified original sources, datasets, and published baseline settings, then builds the authors' Hadamard extension. A C++ compiler is required. On Ubuntu, missing Python headers can be extracted locally without a system installation.

The existing phase experiment uses Python 3.14 in `.venv`, pinned by `requirements-phase.lock`. Preserve that environment when resuming its checkpoints. A fresh installation can create it with:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase.lock
```

## Run

```bash
.venv-bench/bin/python main.py start --suite graphs
.venv-bench/bin/python main.py start --suite scaling
.venv/bin/python main.py start --suite phase
.venv-bench/bin/python main.py status
```

`start` launches persistent `sht-graphs`, `sht-scaling`, and `sht-phase` tmux sessions. A shared FIFO queue serializes measurements across these suites. Each recovery uses one CPU and one numerical thread. Completed trials, oracle responses, and SPHT stage checkpoints are stored in SQLite. Interrupted graph/scaling trials restart from their fixed seeds for valid runtime measurement; completed trials are skipped.

```bash
tmux attach -t sht-graphs
.venv-bench/bin/python main.py stop --suite graphs
.venv-bench/bin/python main.py run --suite graphs
.venv-bench/bin/python main.py plot --suite graphs
```

`main.py` selects the appropriate installed environment for execution. `start` without `--suite` starts phase, graph, and scaling experiments. Robustness and HPO remain available as explicit suites and are deferred by default. `--output PATH` selects a separate result directory.

## Protocol

| Experiment | Repetitions | Figure names |
| --- | ---: | --- |
| Main and sparse phase diagrams | 10 per cell | `phase_diagrams`, `phase_sparse` |
| Graph recovery | 20 per instance and method | `graphs`, `graph_recovery` |
| Scaling | 5 per configuration | `scaling_runtime`, `scaling_queries` |

Graph SWHT and Stobbe call the [authors' original implementations](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/tree/4c3e161e13062a356afcba1f41d0493b0373336c), pinned to commit `4c3e161e13062a356afcba1f41d0493b0373336c`. For each published graph configuration, SWHT uses the fastest saved setting reaching 90% recovery; Stobbe uses the first saved setting reaching that threshold, following the repository's analysis. Source file checksums and the selected settings are retained with the results. Stobbe is limited to the published vertex range; its absence beyond that range is explicit in the experiment configuration.

For graphs and scaling, CS uses the unmodified [SPGL1 0.0.3 basis-pursuit solver](https://spgl1.readthedocs.io/en/latest/api/generated/spgl1.spg_bp.html), solving `min ||a||_1 subject to Phi*a=b`. The separate Stobbe curve uses the released penalized L1/APG implementation. The baseline repository's file named `compressiveSensing.py` implements matching pursuit, so it does not define the L1 CS curve here. The repository has no corresponding general scaling CS experiment; that experiment is an explicitly documented adaptation.

SPGL1 uses its published default tolerances (`bp_tol=1e-6`, `ls_tol=1e-6`, `opt_tol=1e-4`), its default iteration limit, and no optional subspace acceleration. CS retains the earlier sampling budgets: `ceil(2.4*(edges+1)*log2(vertices))` for graphs and `ceil(2.4*k*ln(max(e,D/k)))` for scaling, where `D` is the dictionary size. Recovery uses the graph constant-and-pairs dictionary or the full indicated Fourier dictionary. Known sparsity is provided to SPHT and CS; CS applies top-k least-squares debiasing after the L1 solve. Phase experiments retain their existing basis-pursuit implementation and settings.

SPHT's graph/scaling numerical code and tuned parameters are restored from commit `c1585ce`. Prefix extension, BCH decoding, compiled local basis pursuit, validation, and retry rules are preserved. Validation and retry queries are counted. No custom support-search or certificate shortcut is used. Graph/scaling replacement trials use a fresh, fixed seed split and do not reuse rejected solver results. No result is selected according to which method wins.

All graph/scaling methods have the same 1,800-second trial cap. Cost records distinguish CPU computation, oracle work, checkpoint IO, and wall time. Compiled-kernel warm-up, reusable dictionary preparation, and BCH-map preparation precede measurement; BCH preparation time is recorded separately. Thus runtime plots measure recovery with prepared reusable components. All unique oracle queries, including validation and retries, count toward the query cost.

Graph/scaling cost plots use medians and 95% bootstrap intervals. Graph cost curves include completed unsuccessful trials and timeouts, with resource-limit outcomes and recovery coverage exported separately. Graph recovery intervals use Wilson's method. Scaling cost comparisons use jointly successful CS/SPHT pairs; their coverage is exported to avoid hiding missing comparisons. Five repetitions provide coarse uncertainty estimates. Phase heatmaps use bilinear display interpolation, retain empirical probabilities underneath, and have no error bars.

## Outputs and resuming

Ignored `data/` holds the required original sources and datasets. Ignored `results/` holds each suite's SQLite cache, logs, summaries, and figures. Settings, source identities, package versions, and completion audits are stored inside the SQLite cache; there are no separate JSON manifests. Changing numerical inputs causes a resume error rather than mixing incompatible measurements.

Each completed suite generates PNG and PDF figures in `results/figures/` after checking expected trial keys, repetition counts, and implementation errors. Incomplete suites do not generate preview figures. A blank curve in a final scaling plot can still mean that no trials were jointly successful or a dictionary exceeded the configured limit; `coverage.csv` distinguishes those cases from unfinished work. Complete phase and robustness outputs from the existing protocol can be retained while unfinished or rejected figure exports are removed.
