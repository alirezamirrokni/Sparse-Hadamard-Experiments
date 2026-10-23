# Experiment map

| Source | Supplied experiment | Additional method or output |
| --- | --- | --- |
| Donoho–Tanner Figure 3 and Table 1 Fourier suite | `figure3.json`, partial DCT, `N=200` | Gaussian reference, empirical 50% contours, six-digit recovery |
| Donoho–Tanner Hadamard ensemble and Figure 4 comparison | `phase.json` | Validated and unvalidated prefix panels, probability differences, paired timing |
| `runSWHTEdge.py`, `runStobbeEdge.py` | `graphs.json`, `graph_edges` | BCH-prefix sweeps; snapshots 1, 3, 5, 7; 100 vertices; 20–70 edges |
| `runSWHTVertex.py`, `runStobbeVertex.py` | `graphs.json`, `graph_vertices` | BCH-prefix sweeps; 50 edges; 50–800 vertices; snapshot 5 starts at 70 |
| `runReconstExp.py` | `graphs.json`, `graph_reconstruction` | Symmetric edge difference versus measured queries, including prefix recovery |
| `runExperiments.py`, exploratory edge sweep | `upstream_all.json`, `graph_exploratory_edges` | Full `C` and ratio grid plus prefix settings |
| `runExperiments.py`, exploratory vertex sweep | `upstream_all.json`, `graph_exploratory_vertices` | Original dimension and `C` grids plus prefix settings |
| `PlotData2.ipynb` and `PlotData.ipynb` principal plots | `plot` command | Edge/vertex runtime, samples, uncertainty, and time-versus-samples plots |
| `Reconstruction Error.ipynb` | `plot` command | Separate reconstruction figure per snapshot pair |
| Hazan et al. Algorithm 3 and Section 5.4 | `hpo.json` | Harmonica with a prefix/BCH spectrum-recovery step; sparse and hierarchical objectives |
| New real learning workload | `hpo.json`, digits | Training-based validation/test metrics, dummy-variable scaling, GP-EI and Random Search |
| Manuscript robust-tail and asymptotic scaling claims | `robust.json`, `scaling.json` | Actual recovery errors, mixed-norm normalization, sample and runtime scaling |

## Upstream compatibility

Graph snapshots are downloaded from the exact pinned repository commit and verified by Git blob hashes before use. The graph constructor follows the executable upstream behavior: it compares snapshot `j` with `j+2`, takes their symmetric edge difference on common vertices, selects vertices of highest degree with the upstream tie order, and samples edges using `random.Random(1)`. This differs from an outdated comment in the original source that mentions `j+1`.

The upstream SWHT class is executed from its verified source definitions. Local imports and its `__main__` driver are excluded. It receives the original hashing matrices and inner-product utility, random generator behavior, graph-only frequency checks, and a compatible normalized portable WHT. Time-hash queries are batched in exactly the original binary index order, reducing Python loops and transactional overhead without changing the queried points or transform values. The native Python-3.7 extension is replaced by the compiled Numba transform. This changes transform execution time; results are a portable rerun, not a reproduction of 2019 hardware timings.

The Stobbe baseline implements the same graph quadratic L1 objective over the identifiable constant and degree-two monomials. The redundant `n × n` matrix parameterization is eliminated. A constant monomial represents the sum of diagonal parameters, while each off-diagonal monomial represents the sum of its two symmetric entries. The minimal L1 cost is unchanged. Modern coordinate descent replaces the historical APG package. Its support threshold and unweighted-edge reconstruction follow the upstream experiment. Consequently, timings compare modern implementations rather than deliberately retaining the original expensive intermediate matrices.

The original vertex SWHT sweep extends to 800 vertices; the Stobbe runner stops at 210. The configuration preserves this method-specific range through `method_max_dimensions`. Prefix recovery runs through 800 vertices. A resource-limit result is visible in `status.json` and `summary.csv`; increase `max_dense_bytes` and the method-specific dimension limit if a larger dense degree-two baseline is desired.

The exploratory first vertex cell requests 20 edges on 20 high-degree vertices. Some snapshot-derived graphs can have fewer than the requested number of induced edges. Such cells are reported as explicit configuration failures rather than silently replacing the graph.

The old notebooks contain disconnected scratch cells, manual sample rescaling by 15, stale variable names, and a shared final edge count used to normalize multiple graphs. The new plots use actual query counts and normalize each reconstruction error by that graph's own true edge count. They recreate the defined experiments and principal figure types without importing historical pickle outputs or copying inconsistent scratch-cell arithmetic.

## Practical constants

The implementation uses `initial_factor`, `initial_offset`, `extension_factor`, `sample_factor`, `validation_factor`, and `validation_min` in place of proof-level confidence constants. Robust defaults retain `2k` or `4k` prefixes, use five or nine median groups, and use a group size of `2k` or `4k`, with a minimum group size of eight. Each choice is recorded in the resolved configuration. These values are exposed for calibration, not presented as a proof that small constants satisfy the theoretical lemmas.

The strong-separation robust algorithm and the exact theorem-level QCBP solver are not separately certified numerical modes. The supplied robust experiment exercises the unconditional architecture: coarse IHT, residual subtraction, coordinatewise median-of-means estimation, and enlarged retention. Its final output keeps the largest `k` recovered coefficients, with deterministic ties.

## Recommended interpretation

Use the Fourier/DCT panel to establish reference behavior. Use the Hadamard CS-versus-prefix panel for the actual algorithm comparison. Report realized query counts, the validated and unvalidated modes separately, failed-recovery rates, and multiround timings. Use sparse scaling and low-degree graph recovery to assess computational gains where the local dictionaries are meaningfully smaller than the global dictionary. HPO claims should use the quality-versus-total-cost plots and report spectrum-fitting overhead separately from model-training cost.
