# Experimental details: phase diagrams, graph recovery, and scaling

This document describes the locked measurement protocol and provides methods and caption text for the paper. It does not assert that queued measurements have finished. Graph and scaling outputs are final only when `results/<suite>/audit.json` has `publication_ready: true`; until then their previous figures remain preliminary. HPO and robustness are outside this revision.

## Workloads and repetitions

| Experiment | Methods | Repetitions per configuration |
| --- | --- | --- |
| Main phase diagrams | CS, SPHT | 10 per cell |
| Sparse phase diagrams | CS, SPHT | 10 per cell |
| Graph recovery | CS, original Stobbe, original SWHT, SPHT | 20 per graph instance and configuration |
| Scaling | CS, SPHT | 5 per dimension, sparsity, and degree configuration |

A graph instance fixes the graph snapshot pair, retained vertex count, and edge count. Repetitions change the measurement and algorithm randomization, not the underlying graph. The four snapshot families are fixed benchmarks, not independent samples from an unrestricted graph population. Runtime and query uncertainty is conditional on those benchmark instances.

Graph evaluation has 8,240 trials: 2,220 each for CS, SWHT, and SPHT, and 1,580 for Stobbe. Scaling has 560 trials. All scheduled trials, including failures and resource-limited attempts, remain in the raw cache. Implementation errors stop publication rather than silently becoming recovery failures.

## Phase diagrams

The phase experiments use Boolean-domain Walsh–Hadamard measurements. They are inspired by the random Fourier phase diagrams of [Donoho and Tanner](https://arxiv.org/abs/0906.2530), rather than a literal reproduction of a Fourier or DCT experiment.

Let the ambient dimension be `n` and the number of possible Fourier coefficients be `N = 2^n`. The nominal coordinates are the query fraction `delta = Q/N` and sparsity fraction `rho = k/Q`. We set `Q = round(delta N)` and `k = max(1, min(Q, round(rho Q)))`. Integer rounding matters at the smallest budgets: adjacent nominal cells can represent the same realized sparsity. Actual job specifications and realized sparsities remain in the trial cache.

The main diagram uses `N` in {256, 1,024}, independent Rademacher or standard Gaussian coefficients, query fractions from 0.10 through 0.95 in steps of 0.05, and sparsity fractions from 0.05 through 0.95 in steps of 0.05. Each cell has ten independent repetitions. The expanded sparse diagram uses `N` in {4,096, 16,384, 65,536}, Rademacher coefficients, query fractions {0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08}, and sparsity fractions from 0.025 through 0.30 in steps of 0.025, with ten repetitions per cell.

A support is drawn uniformly without replacement. Both methods receive the same generated signal and the same total distinct-query budget, including SPHT validation queries. Relative Fourier coefficient error at most `1e-6` constitutes recovery. Local least-squares debiasing is included in the implementations. SPHT's prefix, extension, and validation parameters are frozen in the completed suite manifest. The separate sparse-grid parameter calibration and its validation results are retained under `results/refinement/`; the main diagram is unchanged.

Heatmaps show empirical recovery fractions. Bilinear interpolation is applied equally to both methods for display, and the white line marks the interpolated 50% contour. Missing neighboring measurements remain unfilled; interpolation neither adds repetitions nor supplies an estimate outside measured coverage. The black curve in the main diagram is a Gaussian compressed-sensing asymptotic reference, not a theorem for the proposed method. These figures retain their current two-row layout and have no error bars.

Suggested caption: **Empirical recovery phase diagrams.** Columns identify ambient domain size and coefficient distribution; rows compare CS and SPHT. Color represents the fraction of exact recoveries at equal total distinct-query budgets. The white contour marks 50% recovery. Both diagrams use ten repetitions per cell. Colors and contours are bilinearly interpolated between measured cells.

## Graph data and common recovery task

We use the released graph data and graph-construction procedure from [Amrollahi et al.'s experiment repository](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/tree/4c3e161e13062a356afcba1f41d0493b0373336c). The pinned commit is `4c3e161e13062a356afcba1f41d0493b0373336c`. Every downloaded source, historical result table, and data file is checked against its Git blob hash in `src/paper/original_manifest.json`.

The original `getGraph` routine takes the symmetric difference of snapshots `(s, s+2)` for `s` in {1, 3, 5, 7}, restricts to common vertices, retains the requested number of highest-degree vertices, and selects edges using the authors' fixed graph-generation seed. Its output is cached outside recovery timing and cloned with an empty oracle cache for each trial.

The edge sweep fixes 100 vertices and varies the number of edges over {20, 25, ..., 70}. The vertex sweep fixes 50 edges and uses {50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 300, 400, 500, 600, 700, 800} vertices. As in the released scripts, the 50-vertex case is omitted for snapshot family 5, and Stobbe's vertex sweep stops at 210 vertices. Its absence above 210 is outside the reproduced evaluation range, not a failed trial.

All methods query the same cut function,

`f(x) = sum_{(u,v) in E} 1[x_u != x_v]`.

Its Fourier coefficients are `|E|/2` at the constant frequency and `-1/2` at each edge. CS uses the known constant-plus-pairwise dictionary. SPHT uses its degree-two BCH reduction. The methods receive the dimension and edge count, but never the unknown edge locations. The original SWHT and Stobbe interfaces receive `|E|` as in the authors' drivers; CS and SPHT use Fourier sparsity `|E|+1` to include the constant coefficient.

Recovery uses the original `DataGraph.create_from_FT` and graph-equality check: the reconstructed edge set and edge weights must match, with the authors' `1e-4` relative weight tolerance. The constant coefficient is not evaluated by that original graph comparison. We retain recovered edges and weights for auditing.

## Original baseline implementations and reported configurations

The graph baselines execute the released implementations, not equivalent replacements:

- **SWHT:** the original `SWHT` class, original `Hashing.do_TimeHash`, original `WHT.py`, and original C++/FXT Hadamard kernel. The native extension is rebuilt from the unchanged sources using the authors' setup interface, SWIG, and the pinned Python environment. There is no batched replacement of the hashing routine or substituted transform implementation.
- **Stobbe:** the original `ProximalMethod.run` and its released APG optimizer. The original overparameterized matrix representation, proximal update, step-size adaptation, stopping rule, coefficient threshold, and graph postprocessing are preserved. The previous coordinate-descent Lasso replacement is not used in these graph measurements. The released APG defaults are 2,500 iterations and tolerance `1e-6`; the graph experiments use `lambda = 1`.

The published configurations are selected before new evaluation measurements from the authors' saved twenty-repetition result tables, following [their plotting notebook](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/PlotData2.ipynb):

| Baseline | Released search space | Frozen selection rule |
| --- | --- | --- |
| SWHT | `C` in {1.1, 1.2}, bucket ratio in {1.4, 1.3}, 3 coordinate bins, 1 detection iteration | Lowest reported mean runtime among settings with reported recovery at least 90% |
| Stobbe | `C` in {1.5, 2, 2.5, 3}, `lambda = 1` | First, hence smallest, reported `C` reaching at least 90% recovery |

Stobbe uses `m = int(C * ceil(|E| * log2(n)))`, exactly as in its released code. These choices can differ by graph size and snapshot family. The selected settings and original result-table hashes are written to `published-configurations.json`. No setting is selected from our new evaluation outcomes.

Source references: [SWHT edge driver](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/runSWHTEdge.py), [SWHT vertex driver](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/runSWHTVertex.py), [Stobbe edge driver](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/runStobbeEdge.py), [Stobbe vertex driver](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/runStobbeVertex.py), and [original proximal implementation](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions/blob/4c3e161e13062a356afcba1f41d0493b0373336c/code/proximalMethod.py).

## CS formulation and independent calibration

CS solves basis pursuit, `min ||a||_1 subject to Phi a = b`, using normalized sensing operators and least-squares support debiasing. A fast path identifies a candidate sparse support through residual correlations and restricted least-squares fits. It accepts this candidate only after checking primal feasibility and a dual optimality certificate across the entire dictionary: `Phi_S.T y = sign(a_S)` and `||Phi.T y||_infinity <= 1`. The certificate is refined with constrained minimum-norm solves when necessary. The full-dictionary pricing, certificate construction, and support search are included in runtime. The saved solver report contains primal residual, dual feasibility, active-sign error, both objectives, and the duality gap. This is a numerical certificate for the same basis-pursuit objective, not acceptance based on access to the true signal. The acceptance residual and dual tolerances are at most `1e-9`; the relative objective-gap tolerance is `1e-6`. If no certificate is found within the bounded fast path, CS falls back to the existing ADMM solver with tolerance `1e-6` and at most 6,000 iterations. It receives the true sparsity, as does SPHT. The previous CS-only 300-second limit is removed; the common trial wall-time limit is 1,800 seconds for every graph and scaling method.

The ADMM fallback for a degree-constrained dictionary uses the exact affine projection

`v + Phi.T (Phi Phi.T)^dagger (b - Phi v)`.

The Gram matrix is evaluated analytically and factored once per trial, rather than recomputed through repeated iterative projections. For graph cuts, if `t` is the inner product between two sign-encoded query rows, the unnormalized Gram entry is `1 + (t^2 - n)/2`. For complete degree-two or degree-three dictionaries, the degree-one and degree-three Krawtchouk terms are also included. This changes the implementation of the projection, not the sensing model or the basis-pursuit objective. Adjoint and projection identities are checked against explicit small sensing matrices during development.

CS is screened independently for graphs and scaling using the fixed sampling factors {1.5, 2.4}. The fallback ADMM shrinkage parameter is 0.01. The support search uses at most 16 rounds, certificate refinement at most 10 rounds, and a 20-second soft fast-path budget before fallback. On graphs, the sample count is `ceil(C (|E|+1) log2(n))`. On scaling signals, it is `ceil(C k log(max(e, D/k)))`, where `D` is the dictionary size. `D` is `2^n` for unrestricted spectra and `sum_{j=0}^d binom(n,j)` for degree at most `d`.

The graph calibration cases are `(vertices, edges, snapshot)` = {(100,20,1), (100,50,3), (100,70,7), (50,50,1), (210,50,3), (800,50,7)}, with two independent screening repetitions per case. Scaling calibration uses `(n,k,d)` = {(12,8,unrestricted), (18,16,unrestricted), (20,4,unrestricted), (100,16,2), (400,16,2), (60,16,3), (100,16,3)}, with two screening repetitions per case.

The smallest declared sampling factor passing both repetitions on every screening case is frozen. A candidate is rejected after its first failed screening trial. Each screening trial has a 60-second wall-time cap and each suite has a 600-second accumulated screening-compute budget; waiting for other suites does not consume that budget. The budget is checked between trials, so the last attempt can exceed it by at most one capped trial plus delayed native-kernel timeout delivery. There are at most 24 graph and 28 scaling screening trials before early rejection or budget termination. This replaces the previous fourteen-setting, twenty-repetition graph calibration, whose large-dictionary ADMM solves delayed evaluation. Original screening records and old CS timings are kept separately as provenance, not mixed with the new solver's timing results.

Two screening repetitions are a pilot for choosing numerical settings, not evidence that population recovery is at least 90%. Final recovery is measured on the independent twenty-repetition graph or five-repetition scaling evaluation. If no candidate passes all screening cases within budget, the largest declared factor is frozen with `target_met: false`, and that fact must accompany interpretation of the evaluation. Neither this finite screen nor passing it establishes globally optimal query cost or matched recovery. The actual factors, screen outcomes, stopping reason, and timing are saved in `calibration.json`.

SPHT uses frozen parameters selected before these fresh evaluation seeds. In graph recovery they are initial prefix factor 1.5, initial offset 0, extension factor 0.75, and sample factor 1.5, with the existing validation and sample-reuse rules. The procedure is not adjusted using final graph outcomes. All unique validation and retry queries count toward SPHT's reported query cost.

Calibration seeds use the distinct `paper-calibration-v1` role and base seed 20261024. Final graph and scaling seeds use the separate `paper-evaluation-v1` split. Trials are paired by benchmark configuration and repetition index; each method uses its own sampling scheme. Equal seeds do not imply identical query sets for algorithms with different measurement designs.

## Scaling workloads

The unrestricted scaling experiment uses `N` in {4,096, 16,384, 65,536, 262,144, 1,048,576} and `k` in {4, 8, 16, 32}. The degree-two and degree-three experiments use `n` in {30, 60, 100, 200, 400, 800} and `k` in {8, 16, 32}. Supports are sampled without replacement and nonzero coefficients are independent standard Gaussian variables. The generation seed is shared between CS and SPHT within a repetition.

The two scaling figures show CPU runtime and unique query count. Each has three panels: unrestricted spectra, degree two, and degree three. Both figures use exactly the same jointly successful CS/SPHT pairs at each configuration. A pair is successful when both relative coefficient errors are at most `1e-6`. Consequently a curve describes conditional cost in the jointly recoverable regime; it is not an unconditional recovery guarantee.

`coverage.csv` reports completed trials, successes, joint-pair counts, timeouts, and resource limits. A configured point need not appear in a joint-success plot if no pair succeeded. In particular, explicit degree-three CS dictionaries have 10,667,001 and 85,334,001 columns at `n=400` and `n=800`, exceeding the declared 2,000,000-column limit. These outcomes are recorded as resource limits, not numerical recovery failures. SPHT results for those configurations remain in the raw data even when a joint-cost point is unavailable.

## Timing, resources, and cache semantics

The graph and scaling measurements use the pinned Python 3.12 environment in `requirements-paper.lock`: NumPy 1.26.4, SciPy 1.14.1, Numba 0.61.2, and the original APG and native Hadamard sources. All four graph methods run in that same environment. Phase diagrams retain their existing environment and measurement protocol.

The measured host reports an Intel Xeon E5-2699 v4 CPU at 2.20 GHz and approximately 75.6 GiB of visible physical memory. Recovery is pinned to one available logical CPU, with BLAS, OpenMP, and Numba thread counts set to one. Paper trials and phase-extension trials are serialized through a persistent first-come queue and a common process lock. Every completed trial releases its slot, so calibration cannot repeatedly reacquire the lock ahead of waiting evaluation or phase work. Queue entries from interrupted processes are discarded after checking process identity. Compilation and other preparation can run outside measurement slots. The host is shared with other workloads, so these measurements characterize the released software on this host rather than a machine-independent speed guarantee.

Primary runtime is process CPU time spent in online recovery. Oracle-wrapper execution and checkpoint I/O are measured and subtracted for every method. End-to-end wall time, optimizer wall time, and process CPU time are also retained. Query generation, hashing, sample-dependent sensing-matrix construction and factorization, optimization, and decoding remain inside recovery timing. Compilation and dataset construction are outside timing. Data-independent BCH maps and CS frequency-index lists are cached by problem dimensions outside the online timer; preparation times are retained separately.

The common evaluation-trial wall-time budget is 1,800 seconds, checked by the same external Python timer. Independent CS screening uses the shorter 60-second cap described above; screening timeouts are never counted as final evaluation outcomes. A currently executing native kernel can delay delivery of the timeout; actual elapsed time is saved. The numerical implementations permit an 8-GiB dense sensing-matrix cache and impose a 2,000,000-column explicit/local dictionary limit. The cache budget is not claimed to be a hard bound on total process memory.

Queries are distinct Boolean input locations requested during an attempt, including validation, retries, and cache hits at locations not previously requested by that attempt. Duplicate requests do not increase this count. The same original cut-function oracle is used for all graph methods. Oracle values are saved as they are computed. Completed trials are atomic cache entries, and reruns skip them. An interrupted original solver cannot resume its internal iteration state; that trial restarts from its fixed seed with saved oracle values available. Iterative fits also restart for clean timing rather than combining partial timing measurements. These timing restarts do not create additional statistical repetitions.

## Statistical summaries and figure captions

Cost points are medians. Error bars are two-sided 95% nonparametric percentile-bootstrap intervals for the median, using 10,000 deterministic bootstrap draws. Graph bootstraps resample repetitions separately within each fixed snapshot family before combining them; they do not infer uncertainty over an unseen graph population. Scaling bootstraps resample the jointly successful pairs at the plotted configuration. Intervals are omitted when fewer than two eligible observations exist; paired sample counts are available in `coverage.csv`. With five scaling repetitions, these intervals can be coarse.

Graph cost summaries retain completed and time-limited attempts, rather than selecting only the best observed outcome or the fastest setting on evaluation data. They describe compute/query cost spent per attempt under the stated resource budget. Resource-allocation failures are retained in the tables and are not plotted as zero-cost observations. Time-limited attempts and their success indicators remain explicit in the raw data. Runtime comparisons must be interpreted together with recovery, rather than treating a fast unsuccessful attempt as a faster successful recovery.

The graph-recovery figure compares exactly the shared configurations and repetition blocks for all four methods. Its edge panel includes the full edge sweep; its vertex panel uses `n <= 210`, the original Stobbe domain. Thus every method contributes the same 880 edge trials and 700 shared-domain vertex trials when the runs are complete. Error bars there are 95% Wilson intervals for the aggregate recovery proportions on this fixed benchmark. Full-range recovery and resource outcomes remain in the per-configuration summary CSV.

Suggested graph-cost caption: **Graph recovery using the original released baselines.** The edge sweep fixes 100 vertices; the vertex sweep fixes 50 edges. Points show median online CPU runtime or distinct query count from twenty randomized repetitions per fixed graph instance. Error bars are 95% bootstrap intervals, stratified by the fixed snapshot families. Baseline configurations are frozen from the authors' published results; CS is independently calibrated. Recovery and resource outcomes are reported separately.

Suggested graph-recovery caption: **Recovery on shared graph configurations.** Bars show exact graph recovery within the common resource budget, using the same evaluated instances and twenty repetition blocks for every method. The vertex comparison is restricted to at most 210 vertices, matching the released Stobbe experiment. Error bars are 95% Wilson intervals.

Suggested scaling caption: **Scaling on jointly successful instances.** The panels show unrestricted, degree-two, and degree-three spectra. Points show median CPU runtime or distinct query count over the same jointly successful CS/SPHT pairs, drawn from five repetitions per configuration. Error bars are 95% bootstrap intervals for the median. Missing points indicate unavailable jointly successful pairs; the coverage table distinguishes numerical failures from resource limits.

The existing Seaborn style, palette, balanced font sizes, horizontal shared legends, and `SPHT (Ours)` labels are retained. Figures have no overall titles. Every figure is written as both PNG and PDF.

## Reproduction and completion checks

```bash
python3.12 -m venv .paper-venv
.paper-venv/bin/python -m pip install -r requirements-paper.lock
.venv/bin/python main.py paper --paper-stage prepare --suite graphs
.venv/bin/python main.py start --suite graphs
.venv/bin/python main.py start --suite scaling
.venv/bin/python main.py status
```

The original extension requires a C++ compiler. If Python development headers are missing on Ubuntu, preparation downloads and extracts the matching development package locally; it does not require a system package installation. Original source files and their licenses remain unchanged in the downloaded data directory.

Paper calibration, pending evaluations, and logs are under `results/paper/`. Final figures are published to `results/figures/` only after every required trial passes the repetition and implementation-integrity audit. Superseded graph/scaling caches are removed after successful publication; statistical failures and resource-limited outcomes in the new data are preserved. This publication check does not require SPHT to outperform a baseline.

The final graph artifacts are `graphs.png/pdf` and `graph_recovery.png/pdf`; scaling artifacts are `scaling_runtime.png/pdf` and `scaling_queries.png/pdf`. Phase artifacts remain `phase_diagrams.png/pdf` and `phase_sparse.png/pdf`. Each final suite retains its raw SQLite cache, manifest, calibration record, summary CSV, completion audit, and coverage data where applicable. No final numerical speedup or sample-efficiency claim should be inserted into the paper until the corresponding audit is complete.

`paired-comparisons.csv` additionally reports configuration-wise geometric-mean CPU speedup `CS/SPHT` and query-cost ratio `SPHT/CS`, restricted to trials where both methods recovered successfully. Their confidence intervals resample the paired repetition indices, rather than dividing independently resampled marginal estimates. These are conditional comparisons and must be accompanied by the paired success counts. Configuration-selection uncertainty and generalization to new graph families are not included in these intervals. The completion audit records the runtime environment, CPU affinity, compiler version, and compiled original-kernel checksum.
