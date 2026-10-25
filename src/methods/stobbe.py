import math

import numpy as np
from sklearn.linear_model import Lasso

from .. import original_code
from ..numerics import CosetOperator, bits, debias, sparse_result, top_indices


def recover_graph(oracle, dimension, edges, settings):
    return original_code.load().stobbe(dimension, edges, settings["C"]).run(oracle, settings["lambda"])


def recover(oracle, dimension, sparsity, rows, settings, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(1 << dimension, min(rows, 1 << dimension), replace=False)
    operator = CosetOperator(indices.astype(np.uint64), [0], 0, dimension)
    observed = oracle(bits(indices, dimension))
    if operator.shape[0] * operator.shape[1] * 8 > settings.get("max_dense_bytes", 536870912):
        raise MemoryError("Dense Lasso design exceeds the configured memory limit")
    matrix = np.asfortranarray(operator.columns(np.arange(operator.shape[1])) * math.sqrt(len(indices)))
    tolerance = settings.get("tolerance", 1e-8)
    estimator = Lasso(
        alpha=settings.get("penalty", 0.001), fit_intercept=False, copy_X=False,
        max_iter=settings.get("max_iterations", 6000), tol=tolerance, selection="cyclic",
    )
    estimator.fit(matrix, observed)
    values = debias(operator, observed / math.sqrt(len(indices)), estimator.coef_, sparsity)
    selected = top_indices(values, sparsity)
    selected = selected[np.abs(values[selected]) > 1e-8]
    return sparse_result(
        [tuple(np.flatnonzero(row).tolist()) for row in bits(selected, dimension)], values[selected], status="ok",
        rounds=1, solver_report={
            "solver": "lasso-coordinate-descent",
            "iterations": int(estimator.n_iter_),
            "dual_gap": float(estimator.dual_gap_),
            "converged": bool(estimator.dual_gap_ <= tolerance * np.mean(observed**2)),
        },
    )
