# Sparse Hadamard experiments

Experiment code for the manuscript **Fast and Sample-Efficient Sparse Hadamard Transform**. The implementation includes random invertible binary mixing, prefix extension, structured coset operators, sparse validation, BCH syndrome reduction with Berlekamp–Massey and deterministic trace splitting, and a practical robust variant with residual median-of-means refinement.

The experiments measure whether the proposed method matches compressed sensing. Matching phase transitions or achieving a speedup is an experimental question. The code retains unsuccessful trials and reports their outcomes.

## Install

The pinned environment was validated on Python 3.14. Run these commands from the repository directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
shtbench fetch
```

The last command downloads the upstream graph snapshots and baseline source files at a fixed commit and checks their Git blob hashes. It never loads the upstream pickle result files. Source files, downloaded datasets, logs, caches, and generated plots stay outside version control.

For another supported Python version (3.11 or later), install with `python -m pip install -e .` instead of the lock file so pip can resolve compatible package versions. Keep the resulting run manifest with your figures.

## Main runs

```bash
shtbench plan configs/figure3.json
shtbench run configs/figure3.json --output outputs/figure3
shtbench run configs/phase.json --output outputs/phase
shtbench run configs/phase_sparse.json --output outputs/phase_sparse
shtbench run configs/graphs.json --output outputs/graphs
shtbench run configs/scaling.json --output outputs/scaling
shtbench run configs/robust.json --output outputs/robust
shtbench run configs/hpo.json --output outputs/hpo
```

The supplied production grids are large. `plan` reports their exact trial counts before execution. Run the suites separately to obtain useful figures progressively. `--limit 1000` completes only the next 1,000 uncached trials; rerunning the same command continues with the next pending trials. `--no-plots` delays rendering while a long run is in progress.

`configs/graphs.json` includes the principal graph runners, reconstruction experiments, and both exploratory sweeps from upstream `runExperiments.py`. Every requested graph experiment is retained in this single configuration.

Use one worker for publication timing. More workers improve throughput but introduce resource contention:

```bash
shtbench run configs/graphs.json --output outputs/graphs --workers 4
```

For a cluster, launch separate shards into separate output directories and merge their results:

```bash
shtbench run configs/phase.json --output outputs/phase0 --shard 0 --shards 2 --no-plots
shtbench run configs/phase.json --output outputs/phase1 --shard 1 --shards 2 --no-plots
shtbench merge outputs/phase0/cache.sqlite outputs/phase1/cache.sqlite --output outputs/phase
```

A merge imports completed results; it does not merge resumable intermediate solver state or oracle caches. Each original shard remains resumable.

## Frozen parameters

The calibrated settings are embedded directly in the production configurations. The phase suites use initialization offset 6 and extension factor 0.5. The ordinary prefix method in the scaling suite uses the larger-domain profile, with initialization offset 10 and extension factor 1. Graph, BCH scaling, robust, and HPO methods retain their task-specific practical settings.

The small-domain search evaluated 378 candidates and completed 7,040 method-trials including independent validation. In the 128 held-out instances per method, CS recovered 127 and prefix recovered 125. Only sixteen prefix instances used multiple rounds, recovering thirteen. The supplemental larger-domain pilot recovered all six held-out instances for both methods, with four multiround prefix instances. These finite calibration results do not establish a matching full phase boundary.

## Recovery and sampling conventions

`figure3.json` uses a real partial DCT dictionary with ambient size 200 and equality-constrained basis pursuit, following the Fourier suite specified in Table 1 of Donoho and Tanner. It adds a Gaussian reference ensemble. The figure's six-digit recovery criterion is a relative coefficient error of at most `1e-6` here. The original figure does not specify every implementation detail needed for pixel-identical reproduction.

`phase.json` uses partial Walsh–Hadamard measurements at power-of-two ambient sizes. This is the comparison appropriate for the manuscript's Boolean oracle model. Its panels show global CS, validated prefix recovery, the unvalidated prefix variant, and measured probability differences. Horizontal coordinates are the prescribed total distinct-query budget `Q/N`; `summary.csv` also records the realized number of distinct queries. Every prefix fitting and validation query consumes this budget. Baseline and prefix methods see the same generated spectrum, and have independent allowed sensing designs. They need not query the same points.

For practicality, default fitting locations are sampled without replacement within a face. Exact recovery reuses earlier face measurements as additional constraints at later levels, at no additional oracle cost; this is valid because the faces are nested. Set `reuse_samples` to `false` to disable that practical optimization. Fresh fitting rows exclude previously queried locations where possible. Set `sampling` to `replacement` and `reuse_samples` to `false` for the manuscript's independent, with-replacement fitting convention. Median-of-means refinement uses independent groups with replacement. These practical settings and reduced constants do not inherit the finite-confidence constants in the theory.

The initializer can span the entire domain at small sizes or high sparsity. Such a run has one round and does not demonstrate the method's asymptotic runtime advantage. `paired_phase.csv` and the speedup plots explicitly distinguish one-round and multiround runs. `phase_sparse.json` adds smaller query fractions at dimensions 12, 14, and 16, where genuine prefix extension is possible. The scaling suite extends this comparison further.

Basis pursuit uses matrix-free ADMM. The full Hadamard case uses its analytic row Gram structure, and coset extensions use cached small-block pseudoinverses of their block-diagonal measurement Gram matrices. Other operators use conjugate gradients for affine projections. `solver: lp` uses SciPy HiGHS for an independent small-problem baseline. `fista` and `iht` are practical penalized and hard-thresholding solvers. All solver reports include iteration and convergence information. Exact recovery variants debias the selected support using least squares before validation, using the same postprocessing for CS and prefix recovery. The DCT reference is judged directly from its LP output.

The robust experiment uses reduced retention and median-of-means constants, with an IHT coarse step. It reports error relative to the supplied mixed-norm tail budget. Its current finite-iteration coarse solver and practical sampling constants are empirical implementations, not numerical certifications of the manuscript's RIP assumptions or worst-case `4E` guarantee. The spectral tail is generated explicitly; it is distinct from additional observation noise.

## HPO experiments

The HPO suite compares Random Search, Gaussian-process expected improvement, low-degree Lasso Harmonica, and a BCH-prefix replacement for spectrum recovery. They use equal total budgets of unique objective evaluations. The Gaussian-process baseline uses an isotropic Matérn kernel with fixed length scale; it is not the historical Spearmint implementation.

The objectives are a sparse polynomial, the three-level hierarchical function described in Section 5.4 of Hazan et al., and actual classifier training on sklearn's digits dataset. Digits has twelve meaningful binary choices and optionally additional dummy variables. The dummy variables are excluded from objective-cache keys, so duplicate effective model configurations consume neither new training work nor a new unique-evaluation budget. Splits and training randomness are fixed within a replicate; independent replicates use separate splits. Validation error drives selection, and the test partition is evaluated only for the final selected configuration.

The digits choices cover loss, four regularization strengths, penalty type, four step sizes, learning-rate schedule, iterate averaging, intercept fitting, class weighting, training iterations, and feature scaling. This is a new real HPO workload. It does not reproduce the historical CIFAR-10/ResNet GPU numbers. The experiment can demonstrate a useful speedup only if comparable validation quality is preserved after accounting for all oracle work.

The plots include incumbent objective versus objective evaluations, incumbent objective versus charged cold-oracle cost plus measured optimizer overhead, timing decomposition, and optimizer scaling with dummy dimension. The hierarchical function's noise is a deterministic per-configuration realization to keep caching and paired comparisons reproducible. Current Harmonica uses a single minimizing restriction (`t=1`) and a random final search, both options allowed by the paper's algorithm.

## Persistent tmux runs

```bash
.venv/bin/python scripts/run_all.py start
.venv/bin/python scripts/run_all.py status
tmux attach -t sht-phase
```

Four persistent sessions use one process and one BLAS thread each. `sht-phase` runs the full phase diagram followed by the sparse phase grid; `sht-graphs` runs all graph experiments; `sht-hpo` runs HPO; `sht-other` runs the Figure 3 references, robustness, and scaling in sequence. Each lane uses a separate CPU when affinity is available. Other machine workloads can still affect timing.

The launcher renders PNG and PDF figures after each completed batch. Logs and session status are in `outputs/sessions/`; each suite keeps its cache and figures under `outputs/<suite>/`. Completed sessions remain visible in tmux. Rerunning `start` resumes stopped sessions from their caches and leaves active sessions alone. Sessions survive terminal and SSH disconnects; restart them after a machine reboot. The launcher stops between batches if fewer than 5 GiB of disk space remain. Keep source and configurations fixed during a run; the launcher rejects changed inputs in an existing output directory.

## Outputs and recovery after interruption

Every run writes:

| File | Contents |
| --- | --- |
| `cache.sqlite` | Transactional query cache, solver checkpoints, prefix stages, HPO stages, and trial results |
| `manifest_*.json` | Resolved configuration, source fingerprint, dependency versions, machine information, and Git revision |
| `active_run.json` | Source and configuration identity for the current run |
| `progress.log` | Trial outcomes, prefix rounds, solver progress, elapsed time, and progress estimates |
| `trials.jsonl` | Full raw completed-trial results |
| `summary.csv` | Aggregated metrics, statuses, sample counts, times, and Wilson recovery intervals |
| `paired_phase.csv` | Paired successes, query counts, round counts, and speedups |
| `graph_operating_points.csv` | Descriptive operating points selected from the graph sweep |
| `figures/` | Every figure in both PNG and PDF |

Rerun the same command after interruption. Individual expensive objective evaluations are committed as they finish; synthetic oracle evaluations are committed in batches of at most 128 points. Iterative solvers checkpoint every 100 iterations. Completed prefix and HPO stages restore their logical query accounting as well as their numerical results. An interruption inside a solver can repeat up to 99 iterations, and an interruption during a synthetic query batch can repeat that batch. An in-progress individual model fit or LP call must restart. Completed trials are reused exactly.

Query caches are shared between methods for the same synthetic instance. Runtime records distinguish charged original objective evaluation time from current execution time, so a cached HPO result is not represented as a zero-cost training run. Timing should be collected with a fixed worker count, BLAS threads, and machine load. Numba kernels are warmed before trial timing. BCH construction, binary mixing, decoding, fitting, and validation are included in prefix algorithm timing.

The cache identity includes the resolved trial configuration, worker count, and Python source fingerprint. Modified code or worker counts create new trial identities. The active run filters source versions and worker counts during plotting. Use a separate output directory for a different experiment design. Completed solver or algorithm failures remain measured failures. `--retry-errors` retries infrastructure or resource-error results, while ordinary failed recoveries remain cached. Intermediate solver and stage checkpoints are reclaimed atomically after the complete trial result is saved, so finished iterations do not accumulate redundant numerical state.

Resource limits and invalid configurations are counted separately from assessed numerical recoveries. Their status and total count remain in `summary.csv`; they do not become zero-valued recovery probabilities. `solver_seconds` defaults to 1,800 seconds for iterative solvers, checked between iterations. An individual expensive operator product can exceed that interval before control returns. HPO model-training caches are shared across methods, budgets, and dummy dimensions when the actual objective is identical.

Render the current completed results at any time:

```bash
shtbench plot outputs/phase
```

Graph operating-point plots follow the upstream convention of selecting settings with observed success of at least 90%. This is a descriptive selection on the sweep itself; it is not a held-out estimate. Raw results and Wilson intervals are provided to assess uncertainty. Infeasible or resource-limited trials are retained with explicit statuses and are never converted into successful measurements.

## References and provenance

- Donoho and Tanner, [Observed Universality of Phase Transitions in High-Dimensional Geometry](https://arxiv.org/abs/0906.2530).
- Amrollahi et al., [Efficiently Learning Fourier Sparse Set Functions](https://github.com/andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions), pinned at `4c3e161e13062a356afcba1f41d0493b0373336c`.
- Hazan, Klivans, and Yuan, [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764).
