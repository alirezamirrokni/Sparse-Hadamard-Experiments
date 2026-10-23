import itertools
import math

import numpy as np
from numba import njit


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


def bits(indices, dimension):
    indices = np.asarray(indices, dtype=np.uint64)
    return ((indices[:, None] >> np.arange(dimension, dtype=np.uint64)) & 1).astype(
        np.uint8
    )


def integers(array):
    array = np.asarray(array, dtype=np.uint8)
    if array.shape[1] > 63:
        raise ValueError("Internal bit indices support at most 63 bits")
    return array.astype(np.uint64) @ (
        np.uint64(1) << np.arange(array.shape[1], dtype=np.uint64)
    )


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
        raise MemoryError(
            f"Degree dictionary has {size:,} columns; limit is {max_columns:,}"
        )
    return [
        tuple(c)
        for d in range(min(dimension, degree) + 1)
        for c in itertools.combinations(range(dimension), d)
    ]


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


def warmup():
    fwht(np.zeros(4))
    characters(np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64))
    scatter_blocks(np.ones((2, 1)), np.zeros(2, dtype=np.int64), np.zeros(2), 2)
    gather_blocks(np.ones((2, 1)), np.zeros(2, dtype=np.int64), np.zeros((1, 2)))
    degree_characters(
        np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 1), dtype=np.int32)
    )
    evaluate_sparse(
        np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 1), dtype=np.int32), np.ones(2)
    )
