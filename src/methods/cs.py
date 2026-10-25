import math
from functools import lru_cache

import numpy as np

from ..numerics import (
    CosetOperator, DegreeOperator, bits, debias, degree_dictionary, solve, sparse_result, support, top_indices,
)


@lru_cache(maxsize=1)
def spgl1_solver():
    from spgl1 import spg_bp

    return spg_bp


@lru_cache(maxsize=24)
def dictionary(dimension, degree, graph, max_columns):
    if not graph:
        return tuple(degree_dictionary(dimension, degree, max_columns))
    if 1 + dimension * (dimension - 1) // 2 > max_columns:
        raise MemoryError("Graph dictionary exceeds the configured column limit")
    return tuple([()] + [(i, j) for i in range(dimension) for j in range(i + 1, dimension)])


def sample_count(spec, sparsity, factor):
    if spec["kind"].startswith("graph"):
        return math.ceil(factor * sparsity * math.log2(spec["dimension"]))
    degree, dimension = spec.get("degree"), spec["dimension"]
    columns = 1 << dimension if degree is None else sum(math.comb(dimension, d) for d in range(degree + 1))
    return math.ceil(factor * sparsity * math.log(max(math.e, columns / sparsity)))


def recover_spgl1(oracle, dimension, sparsity, rows, settings, seed, degree=None, graph=False):
    rng = np.random.default_rng(seed)
    if degree is None:
        if (1 << dimension) > settings["max_global_columns"]:
            raise MemoryError("Global dictionary exceeds the configured column limit")
        locations = rng.choice(1 << dimension, min(rows, 1 << dimension), replace=False).astype(np.uint64)
        points = bits(locations, dimension)
        operator = CosetOperator(locations, [0], 0, dimension)
        frequencies = None
    else:
        frequencies = dictionary(dimension, degree, graph, settings["max_global_columns"])
        points = rng.integers(0, 2, (rows, dimension), dtype=np.uint8)
        operator = DegreeOperator(points, frequencies, max_cache_bytes=settings["max_dense_bytes"])
    response = oracle(points) / math.sqrt(len(points))
    options = {name: settings[name] for name in ["bp_tol", "ls_tol", "opt_tol", "iter_lim", "subspace_min"] if name in settings}
    values, residual, _, info = spgl1_solver()(operator, response, verbosity=0, **options)
    report = {"solver": "SPGL1", "version": "0.0.3", "objective": "min ||a||_1 subject to Phi*a=b", "options": options}
    for name, value in info.items():
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                continue
            report[name] = value.item() if isinstance(value, np.generic) else value
    report["residual_norm"] = float(np.linalg.norm(residual))
    report["l1_objective"] = float(np.linalg.norm(values, 1))
    values = debias(operator, response, values, sparsity)
    selected = top_indices(values, sparsity)
    selected = selected[np.abs(values[selected]) > 1e-8]
    if frequencies is None:
        recovered = [tuple(np.flatnonzero(row).tolist()) for row in bits(selected, dimension)]
    else:
        recovered = [frequencies[index] for index in selected]
    return sparse_result(recovered, values[selected], status="ok", solver_report=report)


def recover(oracle, dimension, sparsity, rows, settings, store, key, seed, degree=None):
    rng = np.random.default_rng(seed)
    max_columns = settings.get("max_global_columns", 2_000_000)
    if degree is None:
        if 1 << dimension > max_columns:
            raise MemoryError("Global dictionary exceeds configured column limit")
        indices = rng.choice(1 << dimension, min(rows, 1 << dimension), replace=False).astype(np.uint64)
        operator = CosetOperator(indices, [0], 0, dimension)
        points = bits(indices, dimension)
        frequencies = None
    else:
        points = rng.integers(0, 2, (rows, dimension), dtype=np.uint8)
        frequencies = degree_dictionary(dimension, degree, max_columns)
        operator = DegreeOperator(points, frequencies, max_cache_bytes=settings.get("max_dense_bytes", 268_435_456) // 2)
    response = oracle(points) / np.sqrt(len(points))
    values, report = solve(operator, response, settings, store, f"{key}/solver", sparsity)
    if settings.get("debias", True):
        values = debias(operator, response, values, sparsity)
    selected = top_indices(values, sparsity)
    selected = selected[np.abs(values[selected]) > settings.get("coefficient_threshold", 1e-8)]
    output = [support(index, dimension) if frequencies is None else frequencies[index] for index in selected]
    return sparse_result(
        output, values[selected], status="ok", solver_report=report, rounds=1,
        resumed_compute_seconds=report.get("resumed_seconds", 0.0),
    )
