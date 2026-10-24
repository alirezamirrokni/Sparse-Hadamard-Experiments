import math
import time
from functools import lru_cache

import numpy as np
from numba import njit

from src.algebra import bits, degree_dictionary, top_indices
from src.operators import CosetOperator, DegreeOperator
from src.recovery import sparse_result
from src.solvers import debias, solve


class ExactDegreeOperator(DegreeOperator):
    def __init__(self, points, frequencies, degree, include_linear, max_cache_bytes):
        self.degree = degree
        self.include_linear = include_linear
        super().__init__(points, frequencies, max_cache_bytes=max_cache_bytes)

    def gram_blocks(self):
        signs = 1.0 - 2.0 * self.points.astype(float)
        inner = signs @ signs.T
        dimension = self.points.shape[1]
        gram = np.ones_like(inner)
        if self.include_linear:
            gram += inner
        if self.degree >= 2:
            gram += (inner * inner - dimension) / 2
        if self.degree >= 3:
            gram += (inner**3 - (3 * dimension - 2) * inner) / 6
        gram /= self.shape[0]
        values, vectors = np.linalg.eigh(gram)
        keep = values > max(1e-12, np.max(values) * 1e-12)
        factor = vectors[:, keep] / np.sqrt(values[keep])
        return [(np.arange(self.shape[0]), factor)]


@lru_cache(maxsize=24)
def dictionary(dimension, degree, graph, max_columns):
    if graph:
        return tuple(
            [()] + [(a, b) for a in range(dimension) for b in range(a + 1, dimension)]
        )
    return tuple(degree_dictionary(dimension, degree, max_columns))


@njit(cache=True)
def project_certificate(certificate, columns, projected, sweeps, tolerance):
    candidate = certificate.copy()
    positive = np.zeros(len(columns))
    negative = np.zeros(len(columns))
    squared = np.sum(projected * projected, axis=1)
    for iteration in range(sweeps):
        for j in range(len(columns)):
            if squared[j] <= 1e-20:
                continue
            price = np.dot(columns[j], candidate)
            value = max(0.0, positive[j] + (price - 1.0) / squared[j])
            candidate -= (value - positive[j]) * projected[j]
            positive[j] = value
            price = np.dot(columns[j], candidate)
            value = max(0.0, negative[j] + (-price - 1.0) / squared[j])
            candidate += (value - negative[j]) * projected[j]
            negative[j] = value
        maximum = 0.0
        for j in range(len(columns)):
            maximum = max(maximum, abs(np.dot(columns[j], candidate)))
        if maximum <= 1.0 + tolerance:
            break
    return candidate


@lru_cache(maxsize=1)
def warmup():
    project_certificate(np.zeros(2), np.eye(2), np.eye(2), 1, 1e-10)


def certified_support_solve(operator, response, sparsity, settings):
    started = time.monotonic()
    support = np.empty(0, dtype=int)
    residual = response.copy()
    scale = max(1.0, np.linalg.norm(response))
    tolerance = settings.get("tolerance", 1e-6)
    target = min(1e-9, tolerance * 0.01)
    values = np.empty(0)
    deadline = started + settings.get("certificate_seconds", 60)
    for iteration in range(settings.get("support_iterations", 16)):
        proposed = top_indices(
            operator.rmatvec(residual), min(2 * sparsity, operator.shape[1])
        )
        merged = np.union1d(support, proposed)
        matrix = operator.columns(merged)
        fitted = np.linalg.lstsq(matrix, response, rcond=1e-12)[0]
        selected = top_indices(fitted, sparsity)
        next_support = merged[selected]
        matrix = operator.columns(next_support)
        values = np.linalg.lstsq(matrix, response, rcond=1e-12)[0]
        residual = response - matrix @ values
        unchanged = np.array_equal(support, next_support)
        support = next_support
        if np.linalg.norm(residual) <= target * scale:
            break
        if unchanged or time.monotonic() >= deadline:
            return None
    if np.linalg.norm(residual) > target * scale:
        return None
    nonzero = np.abs(values) > target * max(1.0, np.linalg.norm(values))
    support, values = support[nonzero], values[nonzero]
    matrix = operator.columns(support)
    signs = np.sign(values)
    certificate = np.linalg.lstsq(matrix.T, signs, rcond=1e-12)[0]
    constraints = set()
    for step in range(settings.get("certificate_rounds", 10)):
        prices = operator.rmatvec(certificate)
        maximum = float(np.max(np.abs(prices)))
        active_error = float(np.max(np.abs(prices[support] - signs), initial=0.0))
        primal = float(np.linalg.norm(matrix @ values - response))
        objective = float(np.sum(np.abs(values)))
        dual_scale = max(1.0, maximum)
        dual_objective = float(response @ certificate / dual_scale)
        gap = objective - dual_objective
        if (
            maximum <= 1 + target
            and active_error <= target
            and primal <= target * scale
            and abs(gap) <= tolerance * max(1.0, objective)
        ):
            solution = np.zeros(operator.shape[1])
            solution[support] = values
            return solution, {
                "solver": "bp",
                "backend": "support-identification-dual-certificate",
                "iterations": iteration + 1,
                "certificate_rounds": step + 1,
                "converged": True,
                "feasibility": primal,
                "dual_infinity_norm": maximum,
                "active_dual_error": active_error,
                "primal_objective": objective,
                "dual_objective": dual_objective,
                "duality_gap": gap,
                "seconds": time.monotonic() - started,
                "resumed_seconds": 0.0,
            }
        if time.monotonic() >= deadline:
            return None
        constraints.update(np.flatnonzero(np.abs(prices) > 1 + target).tolist())
        constraints.difference_update(support.tolist())
        if not constraints or len(constraints) > 4 * operator.shape[0]:
            return None
        columns = operator.columns(np.asarray(sorted(constraints), dtype=int))
        try:
            coefficients = np.linalg.solve(matrix.T @ matrix, matrix.T @ columns)
        except np.linalg.LinAlgError:
            return None
        projected = columns - matrix @ coefficients
        certificate = project_certificate(
            certificate,
            np.ascontiguousarray(columns.T),
            np.ascontiguousarray(projected.T),
            1000,
            target * 0.1,
        )

    return None


def local_solve(operator, response, settings, store=None, key=None, sparsity=None):
    if operator.shape[1] <= min(operator.shape[0], 256):
        matrix = operator.columns(np.arange(operator.shape[1]))
        values, _, rank, _ = np.linalg.lstsq(matrix, response, rcond=1e-12)
        residual = float(np.linalg.norm(matrix @ values - response))
        if rank == operator.shape[1] and residual <= 1e-9 * max(
            1.0, np.linalg.norm(response)
        ):
            return values, {
                "solver": "bp",
                "backend": "unique-feasible-linear-system",
                "converged": True,
                "iterations": 1,
                "feasibility": residual,
                "resumed_seconds": 0.0,
            }
    if hasattr(operator, "prefix_bits") and (
        operator.prefix_bits > 0 or operator.shape[1] <= 64 * sparsity
    ):
        return solve(operator, response, settings, store, key, sparsity)
    certified = certified_support_solve(operator, response, sparsity, settings)
    if certified is None:
        values, report = solve(operator, response, settings, store, key, sparsity)
        report["certificate_fallback"] = True
    else:
        values, report = certified
    return values, report


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
        indices = rng.choice(
            1 << dimension, min(rows, 1 << dimension), replace=False
        ).astype(np.uint64)
        points = bits(indices, dimension)
        operator = CosetOperator(indices, [0], 0, dimension)
        frequencies = None
    else:
        frequencies = dictionary(
            dimension, degree, graph, settings["max_global_columns"]
        )
        if len(frequencies) > settings["max_global_columns"]:
            raise MemoryError("CS dictionary exceeds the declared column limit")
        points = rng.integers(0, 2, (rows, dimension), dtype=np.uint8)
        operator = ExactDegreeOperator(
            points, frequencies, degree, not graph, settings["max_dense_bytes"]
        )
    response = oracle(points) / np.sqrt(len(points))
    certified = certified_support_solve(operator, response, sparsity, settings)
    if certified is None:
        values, report = solve(
            operator, response, settings, store, key + "/solver", sparsity
        )
        report["certificate_fallback"] = True
    else:
        values, report = certified
    values = debias(operator, response, values, sparsity)
    selected = top_indices(values, sparsity)
    selected = selected[np.abs(values[selected]) > 1e-8]
    if frequencies is None:
        frequencies = [
            tuple(np.flatnonzero(row).tolist()) for row in bits(selected, dimension)
        ]
    else:
        frequencies = [frequencies[index] for index in selected]
    return sparse_result(
        frequencies, values[selected], status="ok", solver_report=report
    )


def sample_count(spec, sparsity, factor):
    dimension = spec["dimension"]
    if spec["kind"].startswith("graph"):
        return math.ceil(factor * sparsity * math.log2(dimension))
    degree = spec.get("degree")
    columns = (
        1 << dimension
        if degree is None
        else sum(math.comb(dimension, j) for j in range(degree + 1))
    )
    return math.ceil(factor * sparsity * math.log(max(math.e, columns / sparsity)))
