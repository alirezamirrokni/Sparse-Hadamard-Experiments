import math


def sparse_result(frequencies, values, **metadata):
    return {
        "frequencies": [[int(j) for j in f] for f in frequencies],
        "values": [float(v) for v in values],
        **metadata,
    }


def coefficient_error(truth, recovered):
    original = {
        tuple(f): float(v) for f, v in zip(truth["frequencies"], truth["values"])
    }
    estimate = {
        tuple(f): float(v)
        for f, v in zip(recovered["frequencies"], recovered["values"])
    }
    error = sum(
        (original.get(f, 0.0) - estimate.get(f, 0.0)) ** 2
        for f in original.keys() | estimate.keys()
    )
    norm = sum(v * v for v in original.values())
    return math.sqrt(error / max(norm, 1e-30))
