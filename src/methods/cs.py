import math
from functools import lru_cache

import numpy as np

from ..numerics import (
    CosetOperator,
    DegreeOperator,
    bits,
    debias,
    solve,
    sparse_result,
    top_indices,
)


@lru_cache(maxsize=1)
def prepare():
    from spgl1 import spg_bp

    return spg_bp


@lru_cache(maxsize=24)
def dictionary(dimension, degree, graph, max_columns):
    from ..numerics import degree_dictionary

    if graph:
        if 1 + dimension * (dimension - 1) // 2 > max_columns:
            raise MemoryError("Graph dictionary exceeds the configured column limit")
        return tuple(
            [()] + [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
        )
    return tuple(degree_dictionary(dimension, degree, max_columns))


def sample_count(spec, sparsity, factor):
    if spec["kind"].startswith("graph"):
        return math.ceil(factor * sparsity * math.log2(spec["dimension"]))
    degree = spec.get("degree")
    dimension = spec["dimension"]
    columns = (
        1 << dimension
        if degree is None
        else sum(math.comb(dimension, d) for d in range(degree + 1))
    )
    return math.ceil(factor * sparsity * math.log(max(math.e, columns / sparsity)))


def recover(
    oracle,
    dimension,
    sparsity,
    rows,
    settings,
    store,
    key,
    seed,
    degree=None,
    graph=False,
):
    rng = np.random.default_rng(seed)
    if degree is None:
        if (1 << dimension) > settings["max_global_columns"]:
            raise MemoryError("Global dictionary exceeds the configured column limit")
        locations = rng.choice(
            1 << dimension, min(rows, 1 << dimension), replace=False
        ).astype(np.uint64)
        points = bits(locations, dimension)
        operator = CosetOperator(locations, [0], 0, dimension)
        frequencies = None
    else:
        frequencies = dictionary(
            dimension, degree, graph, settings["max_global_columns"]
        )
        points = rng.integers(0, 2, (rows, dimension), dtype=np.uint8)
        operator = DegreeOperator(
            points, frequencies, max_cache_bytes=settings["max_dense_bytes"]
        )
    response = oracle(points) / math.sqrt(len(points))
    options = {
        name: settings[name]
        for name in ["bp_tol", "ls_tol", "opt_tol", "iter_lim", "subspace_min"]
        if name in settings
    }
    values, residual, _gradient, info = prepare()(
        operator, response, verbosity=0, **options
    )
    report = {
        "solver": "SPGL1",
        "version": "0.0.3",
        "objective": "min ||a||_1 subject to Phi*a=b",
        "options": options,
    }
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
    recovered = (
        [tuple(np.flatnonzero(row).tolist()) for row in bits(selected, dimension)]
        if frequencies is None
        else [frequencies[index] for index in selected]
    )
    return sparse_result(recovered, values[selected], status="ok", solver_report=report)


def global_recover(
    oracle, dimension, sparsity, rows, settings, store, key, seed, degree=None
):
    from ..numerics import degree_dictionary

    rng = np.random.default_rng(seed)
    if degree is None:
        if 1 << dimension > settings.get("max_global_columns", 2_000_000):
            raise MemoryError("Global dictionary exceeds configured column limit")
        indices = rng.choice(
            1 << dimension, min(rows, 1 << dimension), replace=False
        ).astype(np.uint64)
        operator = CosetOperator(indices, [0], 0, dimension)
        query_points = bits(indices, dimension)
        frequencies = None
    else:
        query_points = rng.integers(0, 2, (rows, dimension), dtype=np.uint8)
        frequencies = degree_dictionary(
            dimension, degree, settings.get("max_global_columns", 2_000_000)
        )
        operator = DegreeOperator(
            query_points,
            frequencies,
            max_cache_bytes=settings.get("max_dense_bytes", 268_435_456) // 2,
        )
    response = oracle(query_points) / np.sqrt(len(query_points))
    values, report = solve(
        operator, response, settings, store, f"{key}/solver", sparsity
    )
    if settings.get("debias", True):
        values = debias(operator, response, values, sparsity)
    selected = top_indices(values, sparsity)
    selected = selected[
        np.abs(values[selected]) > settings.get("coefficient_threshold", 1e-8)
    ]
    if frequencies is None:
        output = [
            tuple(np.flatnonzero(bits([index], dimension)[0]).tolist())
            for index in selected
        ]
    else:
        output = [frequencies[index] for index in selected]
    return sparse_result(
        output,
        values[selected],
        status="ok",
        solver_report=report,
        rounds=1,
        resumed_compute_seconds=report.get("resumed_seconds", 0.0),
    )
