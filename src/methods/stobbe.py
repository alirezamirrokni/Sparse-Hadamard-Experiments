import math

import numpy as np
from sklearn.linear_model import Lasso

from .. import originals
from ..algebra import bits, top_indices
from ..metrics import sparse_result
from ..operators import CosetOperator, materialize
from ..solvers import debias


def recover_graph(oracle, dimension, edges, settings):
    return (
        originals.load()
        .stobbe(dimension, edges, settings["C"])
        .run(oracle, settings["lambda"])
    )


def stobbe_robust_recover(oracle, dimension, sparsity, rows, settings, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(1 << dimension, min(rows, 1 << dimension), replace=False)
    operator = CosetOperator(indices.astype(np.uint64), [0], 0, dimension)
    observed = oracle(bits(indices, dimension))
    matrix = materialize(operator, settings.get("max_dense_bytes", 536870912))
    matrix = np.asfortranarray(matrix * math.sqrt(len(indices)))
    estimator = Lasso(
        alpha=settings.get("penalty", 0.001),
        fit_intercept=False,
        copy_X=False,
        max_iter=settings.get("max_iterations", 6000),
        tol=settings.get("tolerance", 1e-8),
        selection="cyclic",
    )
    estimator.fit(matrix, observed)
    values = debias(
        operator, observed / math.sqrt(len(indices)), estimator.coef_, sparsity
    )
    selected = top_indices(values, sparsity)
    selected = selected[np.abs(values[selected]) > 1e-8]
    return sparse_result(
        [tuple(np.flatnonzero(row).tolist()) for row in bits(selected, dimension)],
        values[selected],
        status="ok",
        rounds=1,
        solver_report={
            "solver": "lasso-coordinate-descent",
            "iterations": int(estimator.n_iter_),
            "dual_gap": float(estimator.dual_gap_),
            "converged": bool(
                estimator.dual_gap_
                <= settings.get("tolerance", 1e-8) * np.mean(observed**2)
            ),
        },
    )
