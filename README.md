# Sparse Hadamard experiments

Experiments comparing **SPHT (Ours)** with compressed sensing and the original SWHT and Stobbe implementations. Run everything through `main.py`; settings are in `config.json` and implementation code is in `src/`.

## Setup

Graph and scaling experiments use the same pinned Python 3.12 environment for every method:

```bash
python3.12 -m venv .paper-venv
.paper-venv/bin/python -m pip install -r requirements-paper.lock
.paper-venv/bin/python main.py paper --suite graphs --paper-stage prepare
```

Preparation downloads checksum-verified sources, datasets, and published baseline settings, then builds the authors' original Hadamard extension. A C++ compiler is required. Missing Ubuntu Python headers are extracted locally without a system installation.

Existing phase runs use `.venv` and `requirements.lock`. Keep that environment when resuming their saved results.

## Run

```bash
.paper-venv/bin/python main.py start --suite graphs
.paper-venv/bin/python main.py start --suite scaling
.venv/bin/python main.py start --suite phase
.paper-venv/bin/python main.py status
```

Experiments run in persistent `sht-graphs`, `sht-scaling`, and `sht-phase` tmux sessions. A shared first-come queue gives each suite measurement slots. Each recovery uses one CPU and one numerical thread. Completed trials and oracle values are cached; rerunning a command resumes unfinished work. An interrupted optimizer restarts from its fixed seed for valid timing.

```bash
tmux attach -t sht-graphs
.paper-venv/bin/python main.py stop --suite graphs
.paper-venv/bin/python main.py plot --suite graphs
```

Foreground execution uses `main.py run --suite graphs`. HPO and robustness remain available but are not started by the commands above.

## Protocol

| Experiment | Repetitions | Outputs |
| --- | ---: | --- |
| Main and sparse phase diagrams | 10 per cell | `phase_diagrams`, `phase_sparse` |
| Graph recovery | 20 per instance and method | `graphs`, `graph_recovery` |
| Scaling | 5 per configuration | `scaling_runtime`, `scaling_queries` |

Graph baselines run the original SWHT hashing/transform code and Stobbe proximal optimizer from [the released repository](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/tree/4c3e161e13062a356afcba1f41d0493b0373336c). Their configurations follow the authors' saved results and selection rules. CS and SPHT solve basis-pursuit systems with shared numerical primitives, identical accuracy requirements, and common evaluation time limits. All SPHT validation and retry queries are counted.

Parameters are chosen on separate calibration seeds and frozen before confirmation runs. Preliminary results used during development are kept separate. Successes, failures, timeouts, and resource limits remain in the raw records; publication does not depend on which method wins.

Graph and scaling cost plots show medians with 95% bootstrap intervals. Graph recovery uses Wilson intervals on shared configurations. Scaling cost comparisons use jointly successful pairs and report their coverage separately. Phase heatmaps retain bilinear display interpolation and have no error bars.

## Results

Every figure is saved as PNG and PDF under `results/figures/`. These files show the latest rendered results, including partial previews while runs are active. Pending evaluations and logs live under `results/paper/`; complete datasets are published to `results/graphs/` and `results/scaling/` only after the repetition and implementation audits pass. Check the suite status and audit before treating a figure as final. Empty panels in partial graph previews are marked as pending.

Each completed suite includes its raw SQLite cache, configuration and source identity, summary tables, and audit. Scaling also records coverage, and paired comparisons include confidence intervals for CPU speedup and query ratios. Preserve manifests when moving results. Local environments, downloaded data, results, TOML files, and the local `EXPERIMENTAL_DETAILS.md` are ignored by Git.
