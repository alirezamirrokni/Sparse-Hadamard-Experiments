# Sparse Hadamard experiments

Run the comparisons for **Fast and Sample-Efficient Sparse Hadamard Transform** from one entry point. All experiment settings live in `config.json`. The repository root contains `main.py`; implementation modules live in `src/`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python main.py start
```

The pinned environment uses Python 3.14. On another supported Python version, install dependencies with `.venv/bin/python -m pip install .` instead of the lock file.

`start` runs five persistent tmux sessions, one process and one BLAS thread per suite. They survive terminal and SSH disconnects. Rerun the command after interruption or reboot to resume the saved trials.

```bash
.venv/bin/python main.py status
tmux attach -t sht-phase
.venv/bin/python main.py stop
```

Use `.venv/bin/python main.py` to run everything sequentially in the foreground. Individual suites are available through `--suite phase`, `graphs`, `scaling`, `robust`, or `hpo`. `main.py plan` prints the trial counts, and `main.py plot` refreshes the figures from completed results. To run only newly added methods, use `.venv/bin/python main.py run --suite graphs --methods cs` or `.venv/bin/python main.py run --suite robust --methods stobbe swht`; these commands resume their own completed trials without running other methods.

## Eight figures, saved as PNG and PDF

Every update overwrites the same eight named figure pairs under `results/figures/`.

| Figure | Panels |
| --- | --- |
| `phase_diagrams` | CS in the top row and SPHT in the bottom row; columns vary domain size and coefficient distribution |
| `phase_sparse` | The same comparison at larger domains and smaller query fractions |
| `graphs` | Runtime and query counts for the principal edge and vertex sweeps |
| `graph_recovery` | Exact-recovery fractions for the two principal graph sweeps |
| `scaling` | Runtime, query counts, and recovery quality; unrestricted and low-degree spectra |
| `robustness_error` | Recovery error divided by the mixed-norm tail budget, shown on log scales |
| `robustness_cost` | Fraction within the 4E target, query counts, and runtime |
| `hpo` | Validation quality versus evaluations and total cost, held-out test quality, synthetic objectives, and optimizer overhead |

There is no separate DCT/Fourier reproduction of the historical Figure 3, and no collection of individual panel files. Phase heatmaps use bilinear interpolation of observed recovery fractions, with the same interpolation for both methods. Black curves in the main phase comparison are Gaussian asymptotic references; white curves mark the interpolated 50% contour. Interpolation does not add trials, fit a probability model, or extrapolate beyond measured coverage. Regions lacking the neighboring observations needed for interpolation remain gray. Raw counts and probabilities remain in the SQLite cache and summary CSV. Figures omit overall titles and use balanced serif typography: 14-point labels/titles, 12-point ticks, and compact legends at the saved figure size. Robustness figures use Seaborn dark-grid styling, a dark-blue SPHT curve, and one horizontal legend below each figure. Their error limits focus on observed values; the 4E criterion appears in the separate cost/recovery figure.

## Experiments and interpretation

Phase experiments use equal total distinct-query budgets for CS and prefix recovery, including validation queries. They pair generated spectra across methods. The main grids use ten repetitions and the sparse grids five. Cells are visited before repetitions are added, so the comparison develops across the whole grid early. Recovery means relative coefficient error at most `1e-6` after common support debiasing.

The largest unrestricted domain is `N = 2^20`. Low-degree experiments and the principal graph sweeps reach 800 Boolean variables. These implicit domains are handled through BCH reduction. Resource-limited global baselines are recorded separately, rather than converted into failed numerical recoveries. Scaling runtimes use jointly successful CS/prefix instances; recovery panels and CSV tables show quality alongside timing.

The graph suite contains only the principal edge and vertex sweeps. It compares CS basis pursuit, SPHT, the Stobbe degree-two Lasso baseline, and upstream SWHT. CS and Stobbe use the known constant-plus-pairwise cut-function dictionary. CS uses sampling factor 2.4 and a 300-second solver cap; its timeouts are reported as resource-limited, not recovery failures. Operating points select the fastest tested setting with observed recovery of at least 90%; they are descriptive selections, not held-out estimates. The separate recovery figure uses all assessed settings. Missing measurements are omitted, not recorded as zero. Upstream snapshots and the three executed SWHT source files are pinned and checksum verified. Graphs and robustness share a slate/blue/green/orange palette for CS/SPHT/Stobbe/SWHT and one horizontal legend below each figure.

The robust method keeps the same coarse-IHT, residual-subtraction, median-of-means, and coordinate-retention architecture. Its practical settings use sampling factor 8, retention 4k, nine median groups of 8k samples, and up to 1,000 coarse iterations. These reduced constants were selected on separate calibration seeds and passed 36/36 held-out checks, compared with 23/36 and 26/36 for the previous practical settings. This finite check does not certify the manuscript's worst-case guarantee. New production results use separate instances. Robustness also evaluates upstream SWHT on the same approximately sparse signals and the Stobbe Lasso objective over the full Hadamard dictionary, using the same sample count and regularization as CS. Stobbe and CS use common top-k least-squares debiasing. SWHT is the upstream implementation evaluated under spectral tails; this does not assert that it implements the paper's separate robust algorithm. Query costs are reported for every method.

HPO compares Random Search, GP expected improvement, Harmonica, and SPHT at equal unique-evaluation budgets. Digits uses fixed validation/test splits per replicate and excludes dummy variables from objective-cache keys. It is a real classifier-training workload, not a reproduction of historical CIFAR GPU timings. Synthetic sparse and hierarchical objectives include both configured noise levels. The Harmonica baseline may cache up to 2 GiB of its dictionary to avoid repeatedly regenerating a large matrix.

## Cache, logs, and reproducibility

`results/<suite>/cache.sqlite` stores oracle evaluations, algorithm checkpoints, and complete trials. Each suite also has one manifest, `summary.csv`, and `status.json`. Graphs additionally save the selected operating points. The SQLite records are the raw results; redundant JSONL copies and historical output archives are not generated.

Logs are in `results/logs/`. Compiled ADMM preserves the original update equations and uses reusable work arrays; `solver_backend: python` remains available for numerical checks. Compiled BP checks checkpoint eligibility every 100 iterations and saves approximately every two seconds. Other iterative solvers checkpoint every 100 iterations, synthetic query batches contain at most 128 points, and expensive model evaluations are saved individually. An in-progress model fit or LP call must restart after interruption.

Numerical source and resolved configurations are checked before resuming a cache; plotting changes do not invalidate numerical results. Use a fresh `--output` directory after changing numerical code or settings. All kernels warm before trial timing. Concurrent suites and other machine workloads can affect timings; an exclusive run is preferable for final publication speed claims. Failures, uncertainty intervals, and resource limits remain in the result tables.

References: [Donoho and Tanner](https://arxiv.org/abs/0906.2530), [Amrollahi et al. and upstream experiments](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/tree/4c3e161e13062a356afcba1f41d0493b0373336c), and [Hazan, Klivans, and Yuan](https://arxiv.org/abs/1706.00764).
