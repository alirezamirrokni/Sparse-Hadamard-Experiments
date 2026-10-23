import logging
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse.linalg import LinearOperator, cg

from .algebra import top_indices
from .operators import materialize


class InfeasibleSystem(RuntimeError):
    pass


def soft(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def debias(operator, response, values, count, tolerance=1e-10):
    indices = top_indices(values, count)
    if not len(indices):
        return np.zeros(operator.shape[1])
    active = operator.columns(indices)
    fitted = np.linalg.lstsq(active, response, rcond=tolerance)[0]
    result = np.zeros(operator.shape[1])
    result[indices] = fitted
    return result


def spectral_norm_squared(operator):
    vector = np.random.default_rng(719).normal(size=operator.shape[1])
    vector /= np.linalg.norm(vector)
    value = 1.0
    for _ in range(25):
        next_vector = operator.rmatvec(operator @ vector)
        value = np.linalg.norm(next_vector)
        if value == 0:
            return 1.0
        vector = next_vector / value
    return max(value * 1.05, 1e-12)


def solve(operator, response, settings, store=None, key=None, sparsity=None):
    method = settings.get("solver", "bp")
    max_bytes = int(settings.get("max_dense_bytes", 268_435_456))
    tolerance = settings.get("tolerance", 1e-8)
    max_iterations = int(settings.get("max_iterations", 6000))
    if method == "lp":
        matrix = materialize(operator, max_bytes // 3)
        result = linprog(
            np.ones(2 * matrix.shape[1]),
            A_eq=np.concatenate([matrix, -matrix], axis=1),
            b_eq=response,
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "time_limit": settings.get("solver_seconds", 1800),
            },
        )
        if not result.success:
            raise RuntimeError(f"Basis pursuit failed: {result.message}")
        return result.x[: matrix.shape[1]] - result.x[matrix.shape[1] :], {
            "solver": "lp",
            "iterations": result.nit,
            "converged": True,
        }
    if method == "iht":
        return iterative(operator, response, settings, store, key, sparsity, True)
    if method == "fista":
        return iterative(operator, response, settings, store, key, sparsity, False)
    if method != "bp":
        raise ValueError(f"Unknown solver {method}")
    m, size = operator.shape
    state = store.load_arrays(key) if store is not None and key else None
    z = np.zeros(size) if state is None else state["z"]
    dual = np.zeros(size) if state is None else state["dual"]
    iteration = 0 if state is None else int(state["iteration"])
    elapsed = 0.0 if state is None else float(state["elapsed"])
    started = time.perf_counter()
    gram = LinearOperator(
        (m, m), matvec=lambda v: operator @ operator.rmatvec(v), dtype=float
    )
    orthogonal = (
        hasattr(operator, "prefixes")
        and len(operator.prefixes) == 1
        and operator.prefix_bits == 0
    )
    unique, inverse, counts = (
        np.unique(operator.points, return_inverse=True, return_counts=True)
        if orthogonal
        else (None, None, None)
    )
    blocks = (
        operator.gram_blocks()
        if hasattr(operator, "gram_blocks") and not orthogonal
        else None
    )
    if blocks is not None:
        projected = np.zeros(m)
        for indices, factor in blocks:
            projected[indices] = factor @ (factor.T @ response[indices])
        mismatch = np.linalg.norm(gram @ projected - response)
        if mismatch > max(1e-9, tolerance * 10) * max(1.0, np.linalg.norm(response)):
            raise InfeasibleSystem(
                "Local equality system is infeasible for the retained prefixes"
            )
    row_norm_squared = size / m
    if (
        hasattr(operator, "suffix")
        and settings.get("solver_backend", "compiled") == "compiled"
    ):
        from .kernels import basis_pursuit_chunk

        indices = [] if blocks is None else [block[0] for block in blocks]
        factors = [] if blocks is None else [block[1] for block in blocks]
        row_order = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
        block_offsets = np.asarray(
            [0] + list(np.cumsum([len(x) for x in indices])), dtype=np.int64
        )
        factor_offsets = np.asarray(
            [0] + list(np.cumsum([x.size for x in factors])), dtype=np.int64
        )
        ranks = np.asarray([x.shape[1] for x in factors], dtype=np.int64)
        packed = (
            np.concatenate([x.ravel() for x in factors]) if factors else np.empty(0)
        )
        inverse = np.empty(0, dtype=np.int64) if inverse is None else inverse
        counts = np.empty(0, dtype=np.int64) if counts is None else counts
        converged = False
        checkpoint_started = time.perf_counter()
        while iteration < max_iterations:
            if time.perf_counter() - started + elapsed > settings.get(
                "solver_seconds", 1800
            ):
                raise TimeoutError("Basis-pursuit solver time limit exceeded")
            stop = min(max_iterations, (iteration // 100 + 1) * 100)
            iteration, converged, feasibility = basis_pursuit_chunk(
                z,
                dual,
                response,
                operator.signs,
                operator.suffix,
                operator.width,
                operator.scale,
                orthogonal,
                inverse,
                counts,
                row_order,
                block_offsets,
                factor_offsets,
                ranks,
                packed,
                iteration,
                stop,
                tolerance,
                settings.get("threshold", 0.1),
            )
            if converged:
                break
            if (
                store is not None
                and key
                and time.perf_counter() - checkpoint_started
                >= settings.get("checkpoint_seconds", 2.0)
            ):
                store.save_arrays(
                    key,
                    z=z,
                    dual=dual,
                    iteration=np.array(iteration),
                    elapsed=np.array(elapsed + time.perf_counter() - started),
                )
                checkpoint_started = time.perf_counter()
            if iteration % 1000 == 0:
                logging.info(
                    "solver %s iteration=%d feasibility=%.3g",
                    key,
                    iteration,
                    feasibility,
                )
        return z, {
            "solver": "bp",
            "backend": "compiled",
            "iterations": iteration,
            "converged": converged,
            "feasibility": float(np.linalg.norm(operator @ z - response)),
            "seconds": elapsed + time.perf_counter() - started,
            "resumed_seconds": elapsed,
        }
    converged = False
    projection_residual = float("inf")
    for iteration in range(iteration, max_iterations):
        if time.perf_counter() - started + elapsed > settings.get(
            "solver_seconds", 1800
        ):
            raise TimeoutError("Basis-pursuit solver time limit exceeded")
        base = z - dual
        residual = response - operator @ base
        if orthogonal:
            sums = np.bincount(inverse, weights=residual)
            correction = (sums / (row_norm_squared * counts * counts))[inverse]
        elif blocks is not None:
            correction = np.zeros(m)
            for indices, factor in blocks:
                correction[indices] = factor @ (factor.T @ residual[indices])
        else:
            correction, info = cg(
                gram,
                residual,
                rtol=min(1e-9, tolerance * 0.1),
                atol=1e-12,
                maxiter=settings.get("projection_iterations", 300),
            )
            projection_residual = np.linalg.norm(gram @ correction - residual)
            if info < 0 or not np.isfinite(projection_residual):
                raise RuntimeError("Affine projection failed")
        x = base + operator.rmatvec(correction)
        old_z = z
        z = soft(x + dual, settings.get("threshold", 0.1))
        dual += x - z
        if iteration % 20 == 0:
            primal = np.linalg.norm(x - z)
            change = np.linalg.norm(z - old_z)
            feasibility = np.linalg.norm(operator @ z - response)
            scale = max(1.0, np.linalg.norm(response), np.linalg.norm(z))
            if max(primal, change, feasibility) <= tolerance * scale:
                converged = True
                break
        if store is not None and key and iteration % 100 == 99:
            store.save_arrays(
                key,
                z=z,
                dual=dual,
                iteration=np.array(iteration + 1),
                elapsed=np.array(elapsed + time.perf_counter() - started),
            )
        if iteration % 1000 == 999:
            logging.info(
                "solver %s iteration=%d feasibility=%.3g",
                key,
                iteration + 1,
                feasibility,
            )
    return z, {
        "solver": "bp",
        "iterations": iteration + 1,
        "converged": converged,
        "feasibility": float(np.linalg.norm(operator @ z - response)),
        "seconds": elapsed + time.perf_counter() - started,
        "resumed_seconds": elapsed,
    }


def iterative(operator, response, settings, store, key, sparsity, hard):
    size = operator.shape[1]
    state = store.load_arrays(key) if store is not None and key else None
    x = np.zeros(size) if state is None else state["x"]
    momentum = x.copy() if state is None else state["momentum"]
    t = 1.0 if state is None else float(state["t"])
    start = 0 if state is None else int(state["iteration"])
    elapsed = 0.0 if state is None else float(state.get("elapsed", 0.0))
    started = time.perf_counter()
    lipschitz = spectral_norm_squared(operator)
    penalty = settings.get("penalty", 1e-4)
    tolerance = settings.get("tolerance", 1e-8)
    converged = False
    for iteration in range(start, settings.get("max_iterations", 3000)):
        if time.perf_counter() - started + elapsed > settings.get(
            "solver_seconds", 1800
        ):
            raise TimeoutError("Iterative solver time limit exceeded")
        proposal = (
            momentum - operator.rmatvec(operator @ momentum - response) / lipschitz
        )
        if hard:
            updated = np.zeros(size)
            selected = top_indices(proposal, sparsity)
            updated[selected] = proposal[selected]
        else:
            updated = soft(proposal, penalty / lipschitz)
        change = np.linalg.norm(updated - x)
        next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = updated if hard else updated + (t - 1) / next_t * (updated - x)
        x = updated
        t = next_t
        if iteration > 20 and change <= tolerance * max(1, np.linalg.norm(x)):
            converged = True
            break
        if store is not None and key and iteration % 100 == 99:
            store.save_arrays(
                key,
                x=x,
                momentum=momentum,
                t=np.array(t),
                iteration=np.array(iteration + 1),
                elapsed=np.array(elapsed + time.perf_counter() - started),
            )
        if iteration % 1000 == 999:
            logging.info(
                "solver %s iteration=%d change=%.3g", key, iteration + 1, change
            )
    return x, {
        "solver": "iht" if hard else "fista",
        "iterations": iteration + 1,
        "converged": converged,
        "feasibility": float(np.linalg.norm(operator @ x - response)),
        "seconds": elapsed + time.perf_counter() - started,
        "resumed_seconds": elapsed,
    }
