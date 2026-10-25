import itertools
import logging
import math
import time
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.sparse.linalg import LinearOperator


@njit(cache=True)
def fwht_blocks(values):
    out = values.copy()
    for row in range(out.shape[0]):
        width = 1
        while width < out.shape[1]:
            for start in range(0, out.shape[1], 2 * width):
                for j in range(width):
                    a = out[row, start + j]
                    b = out[row, start + j + width]
                    out[row, start + j] = a + b
                    out[row, start + j + width] = a - b
            width *= 2
    return out


def fwht(values):
    array = np.asarray(values, dtype=np.float64)
    return fwht_blocks(array.reshape(1, -1)).reshape(array.shape)


@njit(cache=True)
def parity64(x):
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x &= np.uint64(15)
    return (np.uint64(0x6996) >> x) & np.uint64(1)


@njit(cache=True)
def characters(points, frequencies):
    out = np.empty((len(points), len(frequencies)), dtype=np.float64)
    for i in range(len(points)):
        for j in range(len(frequencies)):
            out[i, j] = 1.0 - 2.0 * parity64(points[i] & frequencies[j])
    return out


@njit(cache=True)
def scatter_blocks(signs, suffix, values, width):
    out = np.zeros((signs.shape[1], width))
    for i in range(len(values)):
        for j in range(signs.shape[1]):
            out[j, suffix[i]] += signs[i, j] * values[i]
    return out


@njit(cache=True)
def gather_blocks(signs, suffix, values):
    out = np.zeros(len(suffix))
    for i in range(len(suffix)):
        for j in range(signs.shape[1]):
            out[i] += signs[i, j] * values[j, suffix[i]]
    return out


@njit(cache=True)
def degree_characters(points, frequencies):
    out = np.ones((len(points), len(frequencies)))
    for i in range(len(points)):
        for j in range(len(frequencies)):
            parity = 0
            for coordinate in frequencies[j]:
                if coordinate >= 0:
                    parity ^= points[i, coordinate]
            out[i, j] = 1.0 - 2.0 * parity
    return out


@njit(cache=True)
def evaluate_sparse(points, frequencies, values):
    out = np.zeros(len(points))
    for i in range(len(points)):
        for j in range(len(frequencies)):
            parity = 0
            for coordinate in frequencies[j]:
                if coordinate >= 0:
                    parity ^= points[i, coordinate]
            out[i] += (1.0 - 2.0 * parity) * values[j]
    return out


@njit(cache=True)
def transform_inplace(values, width):
    for start in range(0, len(values), width):
        stride = 1
        while stride < width:
            for block in range(start, start + width, 2 * stride):
                for j in range(stride):
                    a = values[block + j]
                    b = values[block + j + stride]
                    values[block + j] = a + b
                    values[block + j + stride] = a - b
            stride *= 2


@njit(cache=True)
def map_points(indices, matrix):
    out = np.zeros((len(indices), matrix.shape[1]), dtype=np.uint8)
    for i in range(len(indices)):
        value = indices[i]
        for bit in range(matrix.shape[0]):
            if (value >> np.uint64(bit)) & np.uint64(1):
                for j in range(matrix.shape[1]):
                    out[i, j] ^= matrix[bit, j]
    return out


@njit(cache=True)
def basis_pursuit_chunk(
    z, dual, response, signs, suffix, width, scale, orthogonal, inverse, counts, row_order,
    block_offsets, factor_offsets, ranks, factors, first, stop, tolerance, threshold,
):
    size = len(z)
    rows = len(response)
    blocks = signs.shape[1]
    base = np.empty(size)
    transformed = np.empty(size)
    correction_bins = np.empty(size)
    residual = np.empty(rows)
    correction = np.empty(rows)
    sums = np.zeros(len(counts))
    work = np.empty(max(1, np.max(ranks) if len(ranks) else 1))
    feasibility = np.inf
    for iteration in range(first, stop):
        for j in range(size):
            base[j] = z[j] - dual[j]
            transformed[j] = base[j]
        transform_inplace(transformed, width)
        for i in range(rows):
            value = 0.0
            for block in range(blocks):
                value += signs[i, block] * transformed[block * width + suffix[i]]
            residual[i] = response[i] - value / scale
        if orthogonal:
            sums[:] = 0.0
            for i in range(rows):
                sums[inverse[i]] += residual[i]
            row_norm = size / rows
            for i in range(rows):
                count = counts[inverse[i]]
                correction[i] = sums[inverse[i]] / (row_norm * count * count)
        else:
            correction[:] = 0.0
            for block in range(len(ranks)):
                left = block_offsets[block]
                right = block_offsets[block + 1]
                rank = ranks[block]
                offset = factor_offsets[block]
                for j in range(rank):
                    value = 0.0
                    for i in range(left, right):
                        value += factors[offset + (i - left) * rank + j] * residual[row_order[i]]
                    work[j] = value
                for i in range(left, right):
                    value = 0.0
                    for j in range(rank):
                        value += factors[offset + (i - left) * rank + j] * work[j]
                    correction[row_order[i]] = value
        correction_bins[:] = 0.0
        for i in range(rows):
            for block in range(blocks):
                correction_bins[block * width + suffix[i]] += signs[i, block] * correction[i]
        transform_inplace(correction_bins, width)
        primal = 0.0
        change = 0.0
        z_norm = 0.0
        for j in range(size):
            x = base[j] + correction_bins[j] / scale
            old = z[j]
            value = x + dual[j]
            if value > threshold:
                z[j] = value - threshold
            elif value < -threshold:
                z[j] = value + threshold
            else:
                z[j] = 0.0
            dual[j] += x - z[j]
            primal += (x - z[j]) ** 2
            change += (z[j] - old) ** 2
            z_norm += z[j] ** 2
        if iteration % 20 == 0:
            transformed[:] = z
            transform_inplace(transformed, width)
            error = 0.0
            response_norm = 0.0
            for i in range(rows):
                value = 0.0
                for block in range(blocks):
                    value += signs[i, block] * transformed[block * width + suffix[i]]
                error += (value / scale - response[i]) ** 2
                response_norm += response[i] ** 2
            feasibility = np.sqrt(error)
            target = tolerance * max(1.0, np.sqrt(response_norm), np.sqrt(z_norm))
            if max(np.sqrt(primal), np.sqrt(change), feasibility) <= target:
                return iteration + 1, True, feasibility
    return stop, False, feasibility


@lru_cache(maxsize=1)
def warmup():
    empty = np.empty(0, dtype=np.int64)
    basis_pursuit_chunk(
        np.zeros(4), np.zeros(4), np.zeros(2), np.ones((2, 1)), np.array([0, 1]), 4, np.sqrt(2.0), True,
        np.array([0, 1]), np.ones(2, dtype=np.int64), empty, np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64), empty, np.empty(0), 0, 1, 1e-8, 0.1,
    )
    map_points(np.zeros(1, dtype=np.uint64), np.zeros((2, 2), dtype=np.uint8))
    fwht(np.zeros(4))
    characters(np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64))
    scatter_blocks(np.ones((2, 1)), np.zeros(2, dtype=np.int64), np.zeros(2), 2)
    gather_blocks(np.ones((2, 1)), np.zeros(2, dtype=np.int64), np.zeros((1, 2)))
    degree_characters(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 1), dtype=np.int32))
    evaluate_sparse(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 1), dtype=np.int32), np.ones(2))


def bits(indices, dimension):
    indices = np.asarray(indices, dtype=np.uint64)
    return ((indices[:, None] >> np.arange(dimension, dtype=np.uint64)) & 1).astype(np.uint8)


def integers(array):
    array = np.asarray(array, dtype=np.uint8)
    if array.shape[1] > 63:
        raise ValueError("Internal bit indices support at most 63 bits")
    return array.astype(np.uint64) @ (np.uint64(1) << np.arange(array.shape[1], dtype=np.uint64))


def support(index, dimension):
    return tuple(np.flatnonzero(bits([index], dimension)[0]).tolist())


def inverse_binary(matrix):
    matrix = np.asarray(matrix, dtype=np.uint8)
    n = len(matrix)
    augmented = np.concatenate([matrix.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    for j in range(n):
        options = np.flatnonzero(augmented[j:, j])
        if not len(options):
            raise ValueError("Singular binary matrix")
        pivot = j + options[0]
        augmented[[j, pivot]] = augmented[[pivot, j]]
        rows = np.flatnonzero(augmented[:, j])
        rows = rows[rows != j]
        augmented[rows] ^= augmented[j]
    return augmented[:, n:]


def random_invertible(dimension, rng):
    while True:
        matrix = rng.integers(0, 2, (dimension, dimension), dtype=np.uint8)
        try:
            return matrix, inverse_binary(matrix)
        except ValueError:
            continue


def degree_dictionary(dimension, degree, max_columns=2_000_000):
    size = sum(math.comb(dimension, d) for d in range(min(dimension, degree) + 1))
    if size > max_columns:
        raise MemoryError(f"Degree dictionary has {size:,} columns; limit is {max_columns:,}")
    return [tuple(c) for d in range(min(dimension, degree) + 1) for c in itertools.combinations(range(dimension), d)]


def tuple_characters(points, frequencies):
    points = np.asarray(points, dtype=np.uint8)
    out = np.ones((len(points), len(frequencies)))
    for j, frequency in enumerate(frequencies):
        if frequency:
            out[:, j] -= 2 * (points[:, frequency].sum(axis=1) % 2)
    return out


def padded_frequencies(frequencies):
    degree = max((len(f) for f in frequencies), default=0)
    padded = np.full((len(frequencies), degree), -1, dtype=np.int32)
    for j, frequency in enumerate(frequencies):
        padded[j, : len(frequency)] = frequency
    return padded


def top_indices(values, count):
    count = min(len(values), count)
    if count <= 0:
        return np.empty(0, dtype=int)
    if count == len(values):
        return np.arange(len(values))
    cutoff = np.partition(np.abs(values), len(values) - count)[len(values) - count]
    above = np.flatnonzero(np.abs(values) > cutoff)
    tied = np.flatnonzero(np.abs(values) == cutoff)
    return np.sort(np.concatenate([above, tied[: count - len(above)]]))


class CosetOperator(LinearOperator):
    def __init__(self, points, prefixes, prefix_bits, extension_bits):
        self.points = np.asarray(points, dtype=np.uint64)
        self.prefixes = np.asarray(prefixes, dtype=np.uint64)
        self.prefix_bits = prefix_bits
        self.width = 1 << extension_bits
        self.signs = characters(self.points & np.uint64((1 << prefix_bits) - 1), self.prefixes)
        self.suffix = ((self.points >> np.uint64(prefix_bits)) & np.uint64(self.width - 1)).astype(np.int64)
        self.scale = np.sqrt(len(points))
        extensions = np.arange(self.width, dtype=np.uint64)[None, :] << np.uint64(prefix_bits)
        self.frequencies = (self.prefixes[:, None] | extensions).ravel()
        super().__init__(dtype=np.dtype(float), shape=(len(points), len(self.frequencies)))

    def _matvec(self, vector):
        transformed = fwht_blocks(np.asarray(vector, dtype=float).reshape(-1, self.width))
        return gather_blocks(self.signs, self.suffix, transformed) / self.scale

    def _rmatvec(self, vector):
        bins = scatter_blocks(self.signs, self.suffix, np.asarray(vector).ravel(), self.width)
        return fwht_blocks(bins).ravel() / self.scale

    def columns(self, indices):
        return characters(self.points, self.frequencies[indices]) / self.scale

    def gram_blocks(self):
        blocks = []
        for suffix in np.unique(self.suffix):
            indices = np.flatnonzero(self.suffix == suffix)
            signs = self.signs[indices]
            if len(indices) <= signs.shape[1]:
                gram = (self.width / len(self.points)) * (signs @ signs.T)
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                selected = eigenvalues > max(1e-12, np.max(eigenvalues) * 1e-11)
                factor = eigenvectors[:, selected] / np.sqrt(eigenvalues[selected])
            else:
                left, singular, _ = np.linalg.svd(signs, full_matrices=False)
                selected = singular > max(1e-12, np.max(singular) * 1e-11)
                factor = left[:, selected] / singular[selected] * np.sqrt(len(self.points) / self.width)
            blocks.append((indices, factor))
        return blocks


class DegreeOperator(LinearOperator):
    def __init__(self, points, frequencies, chunk=4096, max_cache_bytes=67_108_864):
        self.points = np.asarray(points, dtype=np.uint8)
        self.chunk = chunk
        self.scale = np.sqrt(len(points))
        self.padded = padded_frequencies(frequencies)
        self.cached_matrix = None
        if len(points) * len(frequencies) * 8 <= max_cache_bytes:
            self.cached_matrix = degree_characters(self.points, self.padded)
            self.cached_matrix /= self.scale
        super().__init__(dtype=np.dtype(float), shape=(len(points), len(frequencies)))

    def _matvec(self, vector):
        if self.cached_matrix is not None:
            return self.cached_matrix @ vector
        result = np.zeros(len(self.points))
        for start in range(0, self.shape[1], self.chunk):
            stop = min(start + self.chunk, self.shape[1])
            result += degree_characters(self.points, self.padded[start:stop]) @ vector[start:stop]
        return result / self.scale

    def _rmatvec(self, vector):
        if self.cached_matrix is not None:
            return self.cached_matrix.T @ vector
        result = np.empty(self.shape[1])
        for start in range(0, self.shape[1], self.chunk):
            stop = min(start + self.chunk, self.shape[1])
            result[start:stop] = degree_characters(self.points, self.padded[start:stop]).T @ vector
        return result / self.scale

    def columns(self, indices):
        if self.cached_matrix is not None:
            return self.cached_matrix[:, indices]
        return degree_characters(self.points, self.padded[indices]) / self.scale


class InfeasibleSystem(RuntimeError):
    pass


def soft(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def debias(operator, response, values, count, tolerance=1e-10):
    indices = top_indices(values, count)
    if not len(indices):
        return np.zeros(operator.shape[1])
    fitted = np.linalg.lstsq(operator.columns(indices), response, rcond=tolerance)[0]
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
    if method in {"iht", "fista"}:
        return iterative(operator, response, settings, store, key, sparsity, method == "iht")
    if method != "bp" or not isinstance(operator, CosetOperator):
        raise ValueError(f"Unsupported solver {method} for {type(operator).__name__}")
    return basis_pursuit(operator, response, settings, store, key)


def basis_pursuit(operator, response, settings, store, key):
    tolerance = settings.get("tolerance", 1e-8)
    max_iterations = int(settings.get("max_iterations", 6000))
    time_limit = settings.get("solver_seconds", 1800)
    m, size = operator.shape
    state = store.load_arrays(key) if store is not None and key else None
    z = np.zeros(size) if state is None else state["z"]
    dual = np.zeros(size) if state is None else state["dual"]
    iteration = 0 if state is None else int(state["iteration"])
    elapsed = 0.0 if state is None else float(state["elapsed"])
    started = time.perf_counter()
    orthogonal = len(operator.prefixes) == 1 and operator.prefix_bits == 0
    empty = np.empty(0, dtype=np.int64)
    inverse, counts = empty, empty
    indices, factors = [], []
    if orthogonal:
        _, inverse, counts = np.unique(operator.points, return_inverse=True, return_counts=True)
    else:
        blocks = operator.gram_blocks()
        projected = np.zeros(m)
        for rows, factor in blocks:
            projected[rows] = factor @ (factor.T @ response[rows])
            indices.append(rows)
            factors.append(factor)
        mismatch = np.linalg.norm(operator @ operator.rmatvec(projected) - response)
        if mismatch > max(1e-9, tolerance * 10) * max(1.0, np.linalg.norm(response)):
            raise InfeasibleSystem("Local equality system is infeasible for the retained prefixes")
    row_order = np.concatenate(indices) if indices else empty
    block_offsets = np.asarray([0] + list(np.cumsum([len(x) for x in indices])), dtype=np.int64)
    factor_offsets = np.asarray([0] + list(np.cumsum([x.size for x in factors])), dtype=np.int64)
    ranks = np.asarray([x.shape[1] for x in factors], dtype=np.int64)
    packed = np.concatenate([x.ravel() for x in factors]) if factors else np.empty(0)
    converged = False
    checkpoint_started = time.perf_counter()
    while iteration < max_iterations:
        if time.perf_counter() - started + elapsed > time_limit:
            raise TimeoutError("Basis-pursuit solver time limit exceeded")
        stop = min(max_iterations, (iteration // 100 + 1) * 100)
        iteration, converged, feasibility = basis_pursuit_chunk(
            z, dual, response, operator.signs, operator.suffix, operator.width, operator.scale, orthogonal,
            inverse, counts, row_order, block_offsets, factor_offsets, ranks, packed, iteration, stop,
            tolerance, settings.get("threshold", 0.1),
        )
        if converged:
            break
        if store is not None and key and time.perf_counter() - checkpoint_started >= 2.0:
            store.save_arrays(
                key, z=z, dual=dual, iteration=np.array(iteration),
                elapsed=np.array(elapsed + time.perf_counter() - started),
            )
            checkpoint_started = time.perf_counter()
        if iteration % 1000 == 0:
            logging.info("solver %s iteration=%d feasibility=%.3g", key, iteration, feasibility)
    return z, {
        "solver": "bp",
        "backend": "compiled",
        "iterations": iteration,
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
        if time.perf_counter() - started + elapsed > settings.get("solver_seconds", 1800):
            raise TimeoutError("Iterative solver time limit exceeded")
        proposal = momentum - operator.rmatvec(operator @ momentum - response) / lipschitz
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
                key, x=x, momentum=momentum, t=np.array(t), iteration=np.array(iteration + 1),
                elapsed=np.array(elapsed + time.perf_counter() - started),
            )
        if iteration % 1000 == 999:
            logging.info("solver %s iteration=%d change=%.3g", key, iteration + 1, change)
    return x, {
        "solver": "iht" if hard else "fista",
        "iterations": iteration + 1,
        "converged": converged,
        "feasibility": float(np.linalg.norm(operator @ x - response)),
        "seconds": elapsed + time.perf_counter() - started,
        "resumed_seconds": elapsed,
    }


def sparse_result(frequencies, values, **metadata):
    return {"frequencies": [[int(j) for j in f] for f in frequencies], "values": [float(v) for v in values], **metadata}


def coefficient_error(truth, recovered):
    original = {tuple(f): float(v) for f, v in zip(truth["frequencies"], truth["values"])}
    estimate = {tuple(f): float(v) for f, v in zip(recovered["frequencies"], recovered["values"])}
    error = sum((original.get(f, 0.0) - estimate.get(f, 0.0)) ** 2 for f in original.keys() | estimate.keys())
    norm = sum(v * v for v in original.values())
    return math.sqrt(error / max(norm, 1e-30))
