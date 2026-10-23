# Calibration record

The primary supplied calibration searched 378 combinations of initialization multiplier, initialization offset, extension multiplier, and validation overhead. Successive halving used one replicate per cell for all 378 candidates, three for 126 candidates, and eight for 42 candidates. The cells use dimensions 8, 10, and 12 with sparsities 4–32 and identical total query budgets for CS and prefix recovery.

The search and independent validation completed 7,040 method-trials. Calibration and validation use different seed namespaces. The seed is 20261024; production evaluation uses another namespace. The full search database and search trace are not repository artifacts. The frozen main-run settings and summarized evidence are in `configs/calibrated.json`.

The primary selected initializer is `min(n, ceil(2 log2(k)) + 6)`. Fitting uses distinct queries, reuses earlier face measurements, and reserves at least one fresh validation point per round. This is a practical initialization offset, not the manuscript's proof-level confidence constant. The search cannot identify a preferred extension width from cells with no extension or only two remaining bits; the supplemental scaling calibration examines this parameter at larger dimensions.

## Independent small-domain validation

There are sixteen independent replicates in each of eight cells, for 128 trials per method.

| Dimension | Sparsity | Query budget | CS recovery | Prefix recovery |
| --- | --- | --- | --- | --- |
| 8 | 4 | 37 | 16/16 | 16/16 |
| 10 | 4 | 44 | 16/16 | 16/16 |
| 10 | 8 | 70 | 15/16 | 16/16 |
| 10 | 16 | 146 | 16/16 | 16/16 |
| 12 | 4 | 61 | 16/16 | 13/16 |
| 12 | 8 | 90 | 16/16 | 16/16 |
| 12 | 16 | 186 | 16/16 | 16/16 |
| 12 | 32 | 342 | 16/16 | 16/16 |

The mean absolute difference between cell recovery probabilities is 0.03125, or 3.125 percentage points. Aggregate recovery is 127/128 for CS and 125/128 for prefix recovery.

Only sixteen of these 128 prefix trials use multiple rounds; the others initialize on the whole domain. The multiround cell recovers 13/16 instances, versus 16/16 for CS. The sample is small and a gap remains in that cell. These results do not establish a matching full phase boundary or a large speedup. The median paired timing ratio is approximately one in this small-domain search, and four concurrent workers make these timings unsuitable for a publication speed claim.

Use `phase.json` for independent full-grid recovery estimates, `phase_sparse.json` for sparse multiround cells, and `scaling.json` for the asymptotic runtime comparison. Report the round count with any phase or timing conclusion. The plots intentionally retain measured disagreements.

## Supplemental larger-domain calibration

The supplemental search uses a new seed, 20261026, and nine combinations of initialization offsets 6, 8, and 10 with extension multipliers 0.5, 1, and 1.5. Its three cells are `(n, k, Q) = (14, 4, 87)`, `(16, 4, 101)`, and `(18, 8, 216)`. There are two calibration and two independent validation replicates per cell. It selects offset 10 and extension multiplier 1.

Both methods recover all six validation instances. Four prefix validation instances use multiple rounds. With one worker, the median paired CS/prefix timing ratio is approximately 2.60 across the six validation instances. This is a small pilot, not a precise phase-boundary or speedup estimate. The larger initializer increases computation in exchange for observed recovery reliability. It should not be assumed optimal for graph or HPO tasks.

`configs/calibrated_scaling.json` contains this alternative frozen profile. To evaluate it on the main scaling grid:

```bash
shtbench run configs/scaling.json --output outputs/scaling_calibrated --selection configs/calibrated_scaling.json
```

The profile can reach the whole domain at some small dimensions or large sparsities. Assess the recorded round count, actual candidate sizes, and any resource limits before interpreting speed. The supplied selection records preserve the source fingerprints used during calibration; subsequent packaging and formatting do not turn those finite results into claims about the full production grids.

## Reproduce and extend

```bash
shtbench tune configs/tuning.json --output outputs/calibration --workers 4
shtbench tune configs/tuning_scaling.json --output outputs/scaling_calibration --workers 1
```

For final timing evidence, use one worker, independent instances, and enough repetitions for uncertainty estimates. Treat resource-censored baseline runs separately from numerical recovery failures. The primary small-domain calibration used a 2,000-iteration solver cap and `1e-7` internal tolerance; production phase runs use a larger iteration budget and a stricter internal tolerance. Recovery itself is always judged at relative error `1e-6` after the documented common support debiasing step.
