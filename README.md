# Sparse Hadamard experiments

The repository benchmarks SPHT, compressed sensing, and released sparse-Hadamard baselines. `main.py` is the experiment entry point; implementation modules live in `src/`. Numerical settings and repetition targets live in `config.json`.

The current paper revision covers phase diagrams, graph recovery, and scaling. Graphs use **20 repetitions** and scaling uses **5**. Both phase diagrams use **10 repetitions per cell**, with their existing heatmap format and no error bars. HPO and robustness are retained for later work.

[EXPERIMENTAL_DETAILS.md](EXPERIMENTAL_DETAILS.md) contains the methods, source references, configuration-selection rules, uncertainty definitions, resource accounting, and caption text for the paper.

## Environments

The existing phase environment is `.venv`; its lock file was generated with Python 3.14. On a different supported Python version, install the project dependencies from `pyproject.toml` rather than forcing incompatible locked wheels.

The original baseline extension uses a separate pinned Python 3.12 environment shared by all graph and scaling methods:

```bash
python3.12 -m venv .paper-venv
.paper-venv/bin/python -m pip install -r requirements-paper.lock
.venv/bin/python main.py paper --suite graphs --paper-stage prepare
```

Preparation downloads checksum-verified original sources and historical configuration records, and rebuilds the authors' C++ Hadamard extension. A C++ compiler is required. Missing Ubuntu Python development headers are downloaded and extracted locally when needed. Original baseline sources are not rewritten.

## Run and resume

```bash
.venv/bin/python main.py start --suite graphs
.venv/bin/python main.py start --suite scaling
.venv/bin/python main.py status
```

These commands launch persistent `sht-graphs` and `sht-scaling` tmux sessions. They first run a bounded CS screen on independent seeds and then use frozen evaluation configurations. The screen has at most 24 graph or 28 scaling trials, a 60-second cap per attempt, and a ten-minute accumulated compute budget per suite. Evaluation repetitions remain 20 for graphs and 5 for scaling. A persistent first-come timing queue shares measurement slots with phase extensions, preventing one suite from starving the others. Rerun the commands after interruption to resume completed trials and saved oracle values. An interrupted optimizer fit restarts for valid timing.

```bash
tmux attach -t sht-graphs
.venv/bin/python main.py stop --suite graphs
.venv/bin/python main.py plot --suite scaling
```

Foreground execution is available through `main.py run --suite graphs` or `main.py paper --suite scaling`. Paper runs use complete frozen protocols; method filtering and trial limits are disabled for these measurements. HPO and robustness are not started by the two commands above.

Phase runs use `main.py run --suite phase`. `refine.py --suite phase` resumes the validated expansion of the sparse grid to `k/Q = 0.3`; it does not change the main heatmap format. The main phase experiment must be complete before that refinement prepares replacement data.

## Figures and data

All figures have PNG and PDF versions. Paper graph/scaling replacements are published only after all repetitions and implementation checks pass; until then their existing figures are preliminary.

| Figure | Content |
| --- | --- |
| `phase_diagrams` | Two rows comparing CS and SPHT across small domains and coefficient distributions |
| `phase_sparse` | Larger domains and the expanded sparse recovery regime |
| `graphs` | Runtime and query counts for the principal edge and vertex sweeps, with error bars |
| `graph_recovery` | Shared-domain recovery fractions for all four methods, with error bars |
| `scaling_runtime` | Runtime on jointly successful unrestricted, degree-two, and degree-three instances |
| `scaling_queries` | Query counts on those same jointly successful instances |
| `robustness_error`, `robustness_cost`, `hpo` | Existing experiments outside the current paper revision |

Graph competitors run the original SWHT hashing/transform code and the original Stobbe proximal optimizer. Their configurations are frozen from the authors' saved results using their own plotting rules. CS uses a fast support search followed by a full-dictionary primal/dual optimality certificate for basis pursuit, with the exact Gram-matrix ADMM implementation as fallback. Sampling factors are frozen on a small independent screen. The search uses neither evaluation outcomes nor true support locations. SPHT settings are frozen before the fresh evaluation seeds. Both graph and scaling cost figures use 95% bootstrap intervals; graph recovery uses Wilson intervals. See the experimental-details document for the conditioning and interpretation of each statistic.

`results/paper/` holds calibration, pending paper evaluations, and progress logs. Completed replacements live in `results/graphs/` and `results/scaling/`, with `audit.json`, manifests, raw SQLite records, and summary tables. Scaling also retains `coverage.csv` so missing joint-success points can be distinguished from resource-limited baselines. Final images are under `results/figures/`.

Phase heatmaps use bilinear interpolation of observed recovery fractions equally for both methods. The white contour marks 50% recovery; the black main-grid curve is a Gaussian asymptotic reference. Interpolation does not create measurements or extrapolate over missing neighboring observations. Phase replacement checkpoints and validation records remain under `results/refinement/`.

All suites save progress on the fly. Source, configuration, environment, and seed identities are checked before paper results are reused. Preserve the saved manifests when moving caches. Changes to numerical inputs require a fresh output directory. Plots can be regenerated from completed results without rerunning numerical trials.
