import numpy as np
from scipy.sparse.linalg import LinearOperator

from .algebra import (
    characters,
    degree_characters,
    fwht_blocks,
    gather_blocks,
    padded_frequencies,
    scatter_blocks,
)


class CosetOperator(LinearOperator):
    def __init__(self, points, prefixes, prefix_bits, extension_bits):
        self.points = np.asarray(points, dtype=np.uint64)
        self.prefixes = np.asarray(prefixes, dtype=np.uint64)
        self.prefix_bits = prefix_bits
        self.extension_bits = extension_bits
        self.width = 1 << extension_bits
        self.signs = characters(
            self.points & np.uint64((1 << prefix_bits) - 1), self.prefixes
        )
        self.suffix = (
            (self.points >> np.uint64(prefix_bits)) & np.uint64(self.width - 1)
        ).astype(np.int64)
        self.scale = np.sqrt(len(points))
        self.frequencies = (
            self.prefixes[:, None]
            | (
                np.arange(self.width, dtype=np.uint64)[None, :]
                << np.uint64(prefix_bits)
            )
        ).ravel()
        super().__init__(
            dtype=np.dtype(float), shape=(len(points), len(self.frequencies))
        )

    def _matvec(self, vector):
        transformed = fwht_blocks(
            np.asarray(vector, dtype=float).reshape(-1, self.width)
        )
        return gather_blocks(self.signs, self.suffix, transformed) / self.scale

    def _rmatvec(self, vector):
        bins = scatter_blocks(
            self.signs, self.suffix, np.asarray(vector).ravel(), self.width
        )
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
                threshold = max(1e-12, np.max(eigenvalues) * 1e-11)
                selected = eigenvalues > threshold
                factor = eigenvectors[:, selected] / np.sqrt(eigenvalues[selected])
            else:
                left, singular, right = np.linalg.svd(signs, full_matrices=False)
                selected = singular > max(1e-12, np.max(singular) * 1e-11)
                factor = (
                    left[:, selected]
                    / singular[selected]
                    * np.sqrt(len(self.points) / self.width)
                )
            blocks.append((indices, factor))
        return blocks


class DegreeOperator(LinearOperator):
    def __init__(self, points, frequencies, chunk=4096, max_cache_bytes=67_108_864):
        self.points = np.asarray(points, dtype=np.uint8)
        self.frequencies = frequencies
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
            result += (
                degree_characters(self.points, self.padded[start:stop])
                @ vector[start:stop]
            )
        return result / self.scale

    def _rmatvec(self, vector):
        if self.cached_matrix is not None:
            return self.cached_matrix.T @ vector
        result = np.empty(self.shape[1])
        for start in range(0, self.shape[1], self.chunk):
            stop = min(start + self.chunk, self.shape[1])
            result[start:stop] = (
                degree_characters(self.points, self.padded[start:stop]).T @ vector
            )
        return result / self.scale

    def columns(self, indices):
        if self.cached_matrix is not None:
            return self.cached_matrix[:, indices]
        return degree_characters(self.points, self.padded[indices]) / self.scale


def materialize(operator, max_bytes):
    needed = operator.shape[0] * operator.shape[1] * 8
    if needed > max_bytes:
        raise MemoryError(
            f"Dense operator needs {needed:,} bytes, limit is {max_bytes:,}"
        )
    if hasattr(operator, "columns"):
        return operator.columns(np.arange(operator.shape[1]))
    return np.column_stack([operator @ v for v in np.eye(operator.shape[1])])
