import math

import numpy as np

from .algebra import bits, evaluate_sparse, padded_frequencies


class SparseSignal:
    def __init__(self, dimension, frequencies, values):
        self.dimension = dimension
        self.frequencies = [tuple(f) for f in frequencies]
        self.values = np.asarray(values, dtype=float)
        self.padded = padded_frequencies(self.frequencies)

    def __call__(self, points):
        return evaluate_sparse(
            np.asarray(points, dtype=np.uint8), self.padded, self.values
        )

    def truth(self):
        return {"frequencies": self.frequencies, "values": self.values.tolist()}


def random_signal(
    dimension,
    sparsity,
    seed,
    distribution="rademacher",
    degree=None,
    tail_ratio=0.0,
    tail_size=0,
):
    rng = np.random.default_rng(seed)
    count = sparsity + tail_size
    if degree is None:
        locations = rng.choice(1 << dimension, count, replace=False)
        frequencies = [
            tuple(np.flatnonzero(row).tolist()) for row in bits(locations, dimension)
        ]
    else:
        maximum = sum(math.comb(dimension, d) for d in range(degree + 1))
        if count > maximum:
            raise ValueError("Sparsity exceeds degree dictionary size")
        frequencies = set()
        weights = np.array(
            [math.comb(dimension, d) for d in range(degree + 1)], dtype=float
        )
        while len(frequencies) < count:
            d = int(rng.choice(degree + 1, p=weights / weights.sum()))
            frequencies.add(
                tuple(sorted(rng.choice(dimension, d, replace=False).tolist()))
            )
        frequencies = sorted(frequencies)
        rng.shuffle(frequencies)
    if distribution == "gaussian":
        values = rng.normal(size=count)
    elif distribution == "positive":
        values = rng.uniform(0.5, 1.5, count)
    elif distribution == "dynamic":
        values = rng.choice([-1.0, 1.0], count) * np.geomspace(1.0, 1e-3, count)
    else:
        values = rng.choice([-1.0, 1.0], count)
    if tail_size:
        values[sparsity:] *= (
            tail_ratio
            * np.linalg.norm(values[:sparsity])
            / max(np.linalg.norm(values[sparsity:]), 1e-30)
        )
    return SparseSignal(dimension, frequencies, values)


def graph_signal(dimension, edges):
    return SparseSignal(
        dimension, [()] + list(edges), [len(edges) / 2] + [-0.5] * len(edges)
    )


def tail_bound(signal, sparsity):
    values = np.sort(np.abs(signal.values))[::-1][sparsity:]
    return float(np.linalg.norm(values) + np.sum(values) / np.sqrt(sparsity))
