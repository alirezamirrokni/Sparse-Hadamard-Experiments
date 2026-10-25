PHASE_DELTAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
PHASE_RHOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SPARSE_DELTAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
SPARSE_RHOS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
GRAPH_VERTICES = [50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 300, 400, 500, 600, 700, 800]

RUNTIME = {"timing_workers": 1, "blas_threads": 1}
LIMITS = {"max_dense_bytes": 536870912, "max_global_columns": 2000000, "max_local_columns": 2000000}
EXACT = {"max_iterations": 6000, "tolerance": 1e-06, **LIMITS, "threshold": 0.1, **RUNTIME}
NOISY = {"max_iterations": 6000, "tolerance": 1e-08, **LIMITS, **RUNTIME}

SPGL1 = {
    "solver": "spgl1", "sample_factor": 2.4, "bp_tol": 1e-06, "ls_tol": 1e-06, "opt_tol": 0.0001,
    "subspace_min": False, "iter_lim": None, "source": "drrelyea/spgl1", "version": "0.0.3",
}
SPHT_UNRESTRICTED = {
    "solver": "bp", "initial_factor": 2.0, "initial_offset": 8, "extension_factor": 0.5, "sample_factor": 2.4,
    "validated": True, "validation_factor": 0, "validation_min": 0, "attempts": 1, "sampling": "distinct",
    "reuse_samples": True, "threshold": 0.1, "tolerance": 1e-06,
}

CONFIG = {
    "suites": {
        "phase": {
            "seed": 20261030,
            "split": "evaluation-v2",
            "trials": 10,
            "defaults": EXACT,
            "methods": {"cs": {"solver": "bp"}, "prefix": SPHT_UNRESTRICTED},
            "experiments": [
                {"kind": "phase", "sizes": [8, 10], "deltas": PHASE_DELTAS, "rhos": PHASE_RHOS,
                 "distributions": ["rademacher", "gaussian"], "trials": 10},
                {"kind": "phase", "sizes": [12, 14, 16], "deltas": SPARSE_DELTAS, "rhos": SPARSE_RHOS,
                 "distributions": ["rademacher"], "trials": 10,
                 "method_settings": {"prefix": {"initial_factor": 2.0, "initial_offset": 10, "extension_factor": 0.5}}},
            ],
        },
        "graphs": {
            "seed": 20261023,
            "split": "reference-bp-evaluation-v1",
            "trials": 20,
            "defaults": EXACT,
            "methods": {
                "swht": {"implementation": "original", "configuration": "published"},
                "stobbe": {"implementation": "original", "configuration": "published"},
                "prefix_bch": {
                    "solver": "bp", "initial_factor": 1.5, "initial_offset": 0, "extension_factor": 0.75,
                    "sample_factor": 1.5, "validated": True, "validation_factor": 0.25, "validation_min": 8,
                    "attempts": 2, "sampling": "distinct",
                },
                "cs": SPGL1,
            },
            "experiments": [
                {"kind": "graph_edges", "vertices": 100, "edges": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]},
                {"kind": "graph_vertices", "edges": 50, "vertices": GRAPH_VERTICES, "method_max_dimensions": {"stobbe": 210}},
            ],
        },
        "scaling": {
            "seed": 20261030,
            "split": "reference-bp-evaluation-v1",
            "trials": 5,
            "defaults": EXACT,
            "methods": {
                "cs": SPGL1,
                "prefix": SPHT_UNRESTRICTED,
                "prefix_bch": {
                    "solver": "bp", "initial_factor": 2.0, "initial_offset": 1, "extension_factor": 1.0,
                    "sample_factor": 2.4, "validated": True, "validation_factor": 0.25, "validation_min": 8,
                    "attempts": 2, "sampling": "distinct",
                },
            },
            "experiments": [
                {"kind": "scaling", "methods": ["cs", "prefix"],
                 "grid": {"dimension": [12, 14, 16, 18, 20], "sparsity": [4, 8, 16, 32]}},
                {"kind": "scaling", "methods": ["cs", "prefix_bch"],
                 "grid": {"dimension": [30, 60, 100, 200, 400, 800], "sparsity": [8, 16, 32], "degree": [2, 3]}},
            ],
        },
        "robust": {
            "seed": 20261023,
            "split": "evaluation",
            "trials": 5,
            "defaults": NOISY,
            "methods": {
                "cs": {"solver": "fista", "penalty": 0.001},
                "prefix_robust": {
                    "solver": "iht", "initial_factor": 2.0, "extension_factor": 1.0, "retention": 4,
                    "mom_groups": 9, "mom_factor": 8, "sampling": "distinct", "tolerance": 1e-08,
                    "max_dense_bytes": 536870912, "max_local_columns": 2000000, "sample_factor": 8.0,
                    "initial_offset": 1, "max_iterations": 1000,
                },
                "stobbe": {"penalty": 0.001, "baseline_version": "stobbe-robust-v1"},
                "swht": {"C": 1.2, "ratio": 1.3, "baseline_version": "swht-robust-v1"},
            },
            "experiments": [
                {"kind": "robust", "grid": {
                    "dimension": [14], "sparsity": [8, 16, 32], "tail_ratio": [0.001, 0.01, 0.05, 0.1, 0.25],
                    "tail_size": [128], "distribution": ["gaussian", "dynamic"],
                }},
            ],
        },
        "hpo": {
            "seed": 20261023,
            "split": "evaluation",
            "trials": 3,
            "defaults": {**NOISY, "max_dense_bytes": 4294967296},
            "methods": {
                "random": {},
                "gp_ei": {},
                "harmonica": {"solver": "fista", "penalty": 0.01, "max_iterations": 3000},
                "prefix_bch": {
                    "solver": "fista", "initial_factor": 1.5, "initial_offset": 1, "extension_factor": 1.0,
                    "sample_factor": 2.4, "validated": False, "validation_factor": 0.25, "validation_min": 0,
                    "attempts": 2, "sampling": "distinct", "penalty": 0.01,
                },
            },
            "experiments": [
                {"kind": "hpo",
                 "grid": {"task": ["digits"], "dimension": [12, 30, 60, 100], "budget": [160, 320]},
                 "settings": {"stages": 3, "samples_per_stage": 100, "final_samples": 32, "sparsity": 5, "degree": 3}},
                {"kind": "hpo",
                 "grid": {"task": ["sparse", "hierarchical"], "dimension": [30, 60, 100, 200], "budget": [400], "noise": [0.0, 1.0]},
                 "settings": {"stages": 3, "samples_per_stage": 100, "final_samples": 64, "sparsity": 5, "degree": 3}},
            ],
        },
    },
    "evaluation": {
        "seed": 20261024,
        "time_limit_seconds": 1800,
        "max_iterations": 6000,
        "tolerance": 1e-06,
        "max_dense_bytes": 8589934592,
        "max_global_columns": 2000000,
        "max_local_columns": 2000000,
        "bootstrap_draws": 10000,
    },
}
