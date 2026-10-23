import numpy as np
from numba import njit


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
    z,
    dual,
    response,
    signs,
    suffix,
    width,
    scale,
    orthogonal,
    inverse,
    counts,
    row_order,
    block_offsets,
    factor_offsets,
    ranks,
    factors,
    first,
    stop,
    tolerance,
    threshold,
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
                        value += (
                            factors[offset + (i - left) * rank + j]
                            * residual[row_order[i]]
                        )
                    work[j] = value
                for i in range(left, right):
                    value = 0.0
                    for j in range(rank):
                        value += factors[offset + (i - left) * rank + j] * work[j]
                    correction[row_order[i]] = value
        correction_bins[:] = 0.0
        for i in range(rows):
            for block in range(blocks):
                correction_bins[block * width + suffix[i]] += (
                    signs[i, block] * correction[i]
                )
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


def warmup():
    empty = np.empty(0, dtype=np.int64)
    basis_pursuit_chunk(
        np.zeros(4),
        np.zeros(4),
        np.zeros(2),
        np.ones((2, 1)),
        np.array([0, 1]),
        4,
        np.sqrt(2.0),
        True,
        np.array([0, 1]),
        np.ones(2, dtype=np.int64),
        empty,
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        empty,
        np.empty(0),
        0,
        1,
        1e-8,
        0.1,
    )
    map_points(np.zeros(1, dtype=np.uint64), np.zeros((2, 2), dtype=np.uint8))
