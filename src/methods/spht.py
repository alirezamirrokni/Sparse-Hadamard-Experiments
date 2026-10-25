import functools
import logging
import math
import time

import galois
import numpy as np

from ..numerics import (
    CosetOperator, bits, characters, debias, integers, map_points, random_invertible, solve, sparse_result,
    support, top_indices,
)
from ..store import BudgetExceeded


class BinaryField:
    def __init__(self, degree):
        self.degree = degree
        self.order = (1 << degree) - 1
        self.modulus = int(galois.primitive_poly(2, degree))
        self.exp = [0] * (2 * self.order)
        self.log = [0] * (self.order + 1)
        value = 1
        for i in range(self.order):
            self.exp[i] = value
            self.log[value] = i
            value <<= 1
            if value & (1 << degree):
                value ^= self.modulus
        self.exp[self.order :] = self.exp[: self.order]

    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % self.order]

    def power(self, a, exponent):
        return 0 if a == 0 else self.exp[(self.log[a] * exponent) % self.order]

    def trim(self, polynomial):
        polynomial = list(polynomial)
        while len(polynomial) > 1 and polynomial[-1] == 0:
            polynomial.pop()
        return polynomial

    def add(self, a, b):
        out = [0] * max(len(a), len(b))
        for i, value in enumerate(a):
            out[i] ^= value
        for i, value in enumerate(b):
            out[i] ^= value
        return self.trim(out)

    def multiply(self, a, b):
        out = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] ^= self.mul(x, y)
        return self.trim(out)

    def divide(self, a, b):
        remainder = self.trim(a)
        b = self.trim(b)
        if b == [0]:
            raise ZeroDivisionError
        quotient = [0] * max(1, len(a) - len(b) + 1)
        while remainder != [0] and len(remainder) >= len(b):
            offset = len(remainder) - len(b)
            coefficient = self.div(remainder[-1], b[-1])
            quotient[offset] ^= coefficient
            for j, value in enumerate(b):
                remainder[offset + j] ^= self.mul(coefficient, value)
            remainder = self.trim(remainder)
        return self.trim(quotient), remainder

    def gcd(self, a, b):
        while b != [0]:
            a, b = b, self.divide(a, b)[1]
        return [self.div(value, a[-1]) for value in a]

    def roots(self, polynomial):
        polynomial = self.trim(polynomial)
        if len(polynomial) == 1:
            return []
        if len(polynomial) == 2:
            return [self.div(polynomial[0], polynomial[1])]
        for j in range(self.degree):
            term = [0, 1 << j]
            trace = [0]
            for _ in range(self.degree):
                trace = self.add(trace, term)
                term = self.divide(self.multiply(term, term), polynomial)[1]
            factor = self.gcd(polynomial, trace)
            if 1 < len(factor) < len(polynomial):
                quotient, remainder = self.divide(polynomial, factor)
                if remainder != [0]:
                    raise ValueError("Invalid trace factor")
                return self.roots(factor) + self.roots(quotient)
        raise ValueError("Locator does not split into distinct field roots")

    def berlekamp_massey(self, sequence):
        c = [1] + [0] * len(sequence)
        b = c.copy()
        length = 0
        shift = 1
        discrepancy_old = 1
        for j in range(len(sequence)):
            discrepancy = sequence[j]
            for i in range(1, length + 1):
                discrepancy ^= self.mul(c[i], sequence[j - i])
            if discrepancy == 0:
                shift += 1
                continue
            old = c.copy()
            factor = self.div(discrepancy, discrepancy_old)
            for i in range(len(c) - shift):
                c[i + shift] ^= self.mul(factor, b[i])
            if 2 * length <= j:
                length = j + 1 - length
                b = old
                discrepancy_old = discrepancy
                shift = 1
            else:
                shift += 1
        return self.trim(c[: length + 1])


class BCHMap:
    def __init__(self, dimension, degree):
        self.dimension = dimension
        self.degree = degree
        self.field_bits = dimension.bit_length()
        self.identity = degree * self.field_bits >= dimension
        self.rank = dimension if self.identity else degree * self.field_bits
        if self.rank > 63:
            raise ValueError("BCH syndrome dimension exceeds 63 bits")
        if self.identity:
            self.matrix = np.eye(dimension, dtype=np.uint8)
            self.field = None
            return
        self.field = BinaryField(self.field_bits)
        self.matrix = np.zeros((self.rank, dimension), dtype=np.uint8)
        for coordinate in range(dimension):
            for j in range(degree):
                value = self.field.power(coordinate + 1, 2 * j + 1)
                for bit in range(self.field_bits):
                    self.matrix[j * self.field_bits + bit, coordinate] = (value >> bit) & 1

    def encode(self, locations):
        if self.identity:
            return sum(1 << j for j in locations)
        result = 0
        for j in range(self.degree):
            value = 0
            for location in locations:
                value ^= self.field.power(location + 1, 2 * j + 1)
            result |= value << (j * self.field_bits)
        return result

    def decode(self, syndrome):
        syndrome = int(syndrome)
        if self.identity:
            result = tuple(j for j in range(self.dimension) if (syndrome >> j) & 1)
        else:
            mask = (1 << self.field_bits) - 1
            powers = [0] * (2 * self.degree + 1)
            for j in range(1, len(powers)):
                if j % 2:
                    powers[j] = (syndrome >> ((j // 2) * self.field_bits)) & mask
                else:
                    powers[j] = self.field.mul(powers[j // 2], powers[j // 2])
            roots = self.field.roots(self.field.berlekamp_massey(powers[1:]))
            result = tuple(sorted(self.field.div(1, root) - 1 for root in roots))
        if len(result) > self.degree or any(j < 0 or j >= self.dimension for j in result) or self.encode(result) != syndrome:
            raise ValueError("Syndrome is outside the promised degree ball")
        return result


@functools.lru_cache(maxsize=64)
def bch_map(dimension, degree):
    return BCHMap(dimension, degree)


def fresh_points(oracle, query_map, length, count, rng):
    universe = 1 << length
    count = min(count, universe)
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    if count + oracle.count >= universe // 2:
        candidates = rng.permutation(universe).astype(np.uint64)
        packed = np.packbits(map_points(candidates, query_map), axis=1, bitorder="little")
        chosen = [int(c) for c, row in zip(candidates, packed) if row.tobytes() not in oracle.seen]
        return np.asarray(chosen[:count], dtype=np.uint64)
    chosen = []
    visited = set()
    while len(chosen) < count and len(visited) < universe:
        candidates = rng.integers(0, universe, min(universe, max(32, 2 * (count - len(chosen)))), dtype=np.uint64)
        packed = np.packbits(map_points(candidates, query_map), axis=1, bitorder="little")
        for candidate, row in zip(candidates, packed):
            candidate = int(candidate)
            if candidate not in visited:
                visited.add(candidate)
                if row.tobytes() not in oracle.seen:
                    chosen.append(candidate)
                    if len(chosen) == count:
                        break
    return np.asarray(chosen, dtype=np.uint64)


def draw(oracle, query_map, length, count, rng, distinct):
    if distinct:
        return fresh_points(oracle, query_map, length, count, rng)
    return rng.integers(0, 1 << length, count, dtype=np.uint64)


def robust_refinement(oracle, query_map, operator, response, values, prefixes, old_length, length, sparsity, keep, settings, rng):
    coarse = np.zeros(operator.shape[1])
    selected = top_indices(values, keep)
    coarse[selected] = values[selected]
    group_size = max(8, math.ceil(settings.get("mom_factor", 2.0) * sparsity))
    refinements = []
    for _ in range(settings.get("mom_groups", 5)):
        rows = rng.integers(0, 1 << length, group_size, dtype=np.uint64)
        refined = CosetOperator(rows, prefixes, old_length, length - old_length)
        observed = oracle(map_points(rows, query_map)) / math.sqrt(group_size)
        refinements.append(refined.rmatvec(observed - refined @ coarse))
    values = coarse + np.median(np.asarray(refinements), axis=0)
    return values, top_indices(values, keep), float(np.linalg.norm(operator @ values - response))


def run_stage(
    oracle, query_map, internal, length, prefixes, old_length, sparsity, settings, store, stage_key, stage_seed,
    sample_history, validated, attempts, robust, keep, tail_budget,
):
    evaluation_started = oracle.actual_evaluation_seconds
    rng = np.random.default_rng(stage_seed)
    extension = length - old_length
    columns = len(prefixes) * (1 << extension)
    if columns > settings.get("max_local_columns", 2_000_000):
        raise MemoryError(f"Local dictionary has {columns:,} columns")
    log_ratio = math.log(max(math.e, columns / max(1, sparsity)))
    desired = max(sparsity + 1, math.ceil(settings.get("sample_factor", 2.4) * max(1.0, sparsity * log_ratio)))
    validation_count = 0
    if validated:
        validation_count = max(1, settings.get("validation_min", 8), math.ceil(settings.get("validation_factor", 0.25) * sparsity))
    allocated = None
    if oracle.budget is not None:
        fraction = log_ratio / (log_ratio + (internal - length) * math.log(2))
        allocated = math.floor((oracle.budget - oracle.count) * fraction)
        if allocated < 1:
            raise BudgetExceeded("No remaining stage budget")
        if validated and allocated < 2:
            raise BudgetExceeded("Stage budget cannot reserve independent validation")
        validation_count = min(validation_count, max(int(validated), allocated // 8))
        desired = max(1, allocated - validation_count)
    distinct = settings.get("sampling", "distinct") == "distinct"
    validations = draw(oracle, query_map, length, validation_count, rng, distinct)
    validation_count = len(validations)
    validation_values = oracle(map_points(validations, query_map)) if validation_count else None
    best, best_residual, reports = None, float("inf"), []
    for attempt in range(1 if allocated is not None else attempts):
        rows = draw(oracle, query_map, length, desired, rng, distinct)
        if not len(rows):
            rows = rng.integers(0, 1 << length, min(desired, 1 << length), dtype=np.uint64)
        sampled = oracle(map_points(rows, query_map))
        reused_rows = 0
        fitting_rows, fitting_values = rows, sampled
        if settings.get("reuse_samples", True) and not robust and sample_history:
            reused_rows = len(sample_history)
            fitting_rows = np.concatenate([np.asarray(list(sample_history), dtype=np.uint64), rows])
            fitting_values = np.concatenate([np.asarray(list(sample_history.values())), sampled])
        operator = CosetOperator(fitting_rows, prefixes, old_length, extension)
        response = fitting_values / math.sqrt(len(fitting_rows))
        values, report = solve(operator, response, settings, store, f"{stage_key}/solver/{attempt}", min(sparsity, columns))
        sample_history.update({int(row): float(value) for row, value in zip(rows, sampled)})
        if robust:
            if tail_budget is None or tail_budget <= 0:
                raise ValueError("Robust recovery needs an explicit positive tail budget")
            if allocated is not None:
                raise ValueError("Budgeted robust recovery requires a separate refinement budget")
            values, selected, residual = robust_refinement(
                oracle, query_map, operator, response, values, prefixes, old_length, length, sparsity, keep, settings, rng
            )
        else:
            values = debias(operator, response, values, sparsity)
            selected = top_indices(values, sparsity)
            threshold = settings.get("coefficient_threshold", 1e-8) * max(1.0, np.max(np.abs(values), initial=0))
            selected = selected[np.abs(values[selected]) > threshold]
            if validation_count:
                predicted = characters(validations, operator.frequencies[selected]) @ values[selected]
                residual = float(np.linalg.norm(predicted - validation_values) / max(1.0, np.linalg.norm(validation_values)))
            else:
                residual = float(np.linalg.norm(operator @ values - response) / max(1.0, np.linalg.norm(response)))
        reports.append(report)
        candidate = {
            "retained": operator.frequencies[selected].tolist(),
            "values": values[selected].tolist(),
            "accepted": bool(residual <= settings.get("validation_tolerance", 1e-6) or robust),
            "residual": residual,
            "columns": columns,
            "sampled_rows": len(rows),
            "reused_rows": reused_rows,
            "validation_rows": validation_count,
            "attempts": attempt + 1,
            "solver_reports": reports,
        }
        if residual < best_residual:
            best, best_residual = candidate, residual
        if candidate["accepted"] or not validated:
            break
    best["oracle_state"] = oracle.snapshot()
    observed = [] if validation_values is None else validation_values
    sample_history.update({int(row): float(value) for row, value in zip(validations, observed)})
    best["sample_history"] = sample_history
    best["resumed_seconds"] = sum(report.get("resumed_seconds", 0.0) for report in reports)
    best["evaluation_wall_seconds"] = oracle.actual_evaluation_seconds - evaluation_started
    return best


def prefix_recover(oracle, dimension, sparsity, settings, store, key, seed, degree=None, robust=False, tail_budget=None):
    preprocessing_started = time.perf_counter()
    rng = np.random.default_rng(seed)
    if degree is None:
        if dimension > 63:
            raise ValueError("Use BCH for ambient dimensions above 63")
        reduction, internal = np.eye(dimension, dtype=np.uint8), dimension
        decoder = lambda frequency: support(frequency, dimension)
    else:
        code = bch_map(dimension, degree)
        reduction, internal, decoder = code.matrix, code.rank, code.decode
    mix, inverse = random_invertible(internal, rng)
    query_map = (mix @ reduction) % 2
    initial_factor = settings.get("initial_factor", 2.0)
    initial_bits = min(internal, max(1, math.ceil(initial_factor * math.log2(max(2, sparsity))) + settings.get("initial_offset", 1)))
    width = max(1, math.ceil(settings.get("extension_factor", 1.0) * math.log2(max(2, 2 * sparsity))))
    levels = [initial_bits] + list(range(initial_bits + width, internal, width))
    if levels[-1] != internal:
        levels.append(internal)
    keep = max(sparsity, math.ceil(settings.get("retention", 2.0 if robust else 1.0) * sparsity))
    validated = settings.get("validated", True) and not robust
    attempts = settings.get("attempts", 2) if validated else 1
    mode = "robust" if robust else "validated" if validated else "unvalidated"
    retained = np.array([0], dtype=np.uint64)
    previous = 0
    stages = []
    sample_history = {}
    total_seconds = 0.0
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    for stage, length in enumerate(levels):
        stage_key = f"{key}/stage/{stage}"
        stage_seed = int(rng.integers(0, 2**63 - 1))
        result = store.memo(stage_key, lambda: run_stage(
            oracle, query_map, internal, length, retained, previous, sparsity, settings, store, stage_key, stage_seed,
            sample_history, validated, attempts, robust, keep, tail_budget,
        ))
        oracle.restore(result["oracle_state"])
        sample_history = {int(row): value for row, value in result["sample_history"].items()}
        result = {name: value for name, value in result.items() if name not in {"oracle_state", "sample_history"}}
        logging.info(
            "prefix %s stage=%d/%d bits=%d columns=%d accepted=%s queries=%d",
            mode, stage + 1, len(levels), length, result["columns"], result["accepted"], oracle.count,
        )
        stages.append(result)
        total_seconds += result["seconds"]
        retained = np.asarray(result["retained"], dtype=np.uint64)
        previous = length
        failure = "validation_failed" if validated and not result["accepted"] else "empty_support"
        if failure == "validation_failed" or not len(retained):
            return sparse_result(
                [], [], status=failure, stages=stages, rounds=len(stages), algorithm_seconds=total_seconds + preprocessing_seconds,
                preprocessing_seconds=preprocessing_seconds, mode=mode,
            )
    decoding_started = time.perf_counter()
    values = np.asarray(stages[-1]["values"])
    selected = top_indices(values, sparsity)
    syndromes = integers((bits(retained[selected], internal) @ inverse.T) % 2)
    frequencies, amplitudes, invalid = [], [], 0
    for syndrome, value in zip(syndromes, values[selected]):
        try:
            frequencies.append(decoder(int(syndrome)))
            amplitudes.append(value)
        except ValueError:
            invalid += 1
    decoding_seconds = time.perf_counter() - decoding_started
    return sparse_result(
        frequencies, amplitudes, status="ok" if invalid == 0 else "invalid_syndrome", stages=stages, rounds=len(stages),
        algorithm_seconds=total_seconds + preprocessing_seconds + decoding_seconds, preprocessing_seconds=preprocessing_seconds,
        decoding_seconds=decoding_seconds, mode=mode, internal_dimension=internal, invalid_syndromes=invalid,
        tail_budget=tail_budget,
    )
