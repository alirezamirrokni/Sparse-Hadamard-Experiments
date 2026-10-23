import logging
import math
import time

import numpy as np

from .algebra import bits, characters, integers, random_invertible, top_indices
from .bch import bch_map
from .kernels import map_points
from .operators import CosetOperator, DegreeOperator
from .solvers import debias, solve
from .storage import BudgetExceeded


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


def fresh_points(oracle, query_map, internal, length, count, rng):
    universe = 1 << length
    count = min(count, universe)
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    if count + oracle.count >= universe // 2:
        candidates = rng.permutation(universe).astype(np.uint64)
        points = map_points(candidates, query_map)
        packed = np.packbits(points, axis=1, bitorder="little")
        chosen = [
            int(c)
            for c, row in zip(candidates, packed)
            if row.tobytes() not in oracle.seen
        ]
        return np.asarray(chosen[:count], dtype=np.uint64)
    chosen = []
    visited = set()
    while len(chosen) < count and len(visited) < universe:
        batch = min(universe, max(32, 2 * (count - len(chosen))))
        candidates = rng.integers(0, universe, batch, dtype=np.uint64)
        packed = np.packbits(
            map_points(candidates, query_map), axis=1, bitorder="little"
        )
        for candidate, row in zip(candidates, packed):
            candidate = int(candidate)
            if candidate not in visited:
                visited.add(candidate)
                if row.tobytes() not in oracle.seen:
                    chosen.append(candidate)
                    if len(chosen) == count:
                        break
    return np.asarray(chosen, dtype=np.uint64)


def prefix_recover(
    oracle,
    dimension,
    sparsity,
    settings,
    store,
    key,
    seed,
    degree=None,
    robust=False,
    tail_budget=None,
):
    preprocessing_started = time.perf_counter()
    rng = np.random.default_rng(seed)
    if degree is None:
        if dimension > 63:
            raise ValueError("Use BCH for ambient dimensions above 63")
        reduction = np.eye(dimension, dtype=np.uint8)
        decoder = lambda frequency: tuple(
            np.flatnonzero(bits([frequency], dimension)[0]).tolist()
        )
        internal = dimension
    else:
        code = bch_map(dimension, degree)
        reduction = code.matrix
        decoder = code.decode
        internal = code.rank
    mix, inverse = random_invertible(internal, rng)
    query_map = (mix @ reduction) % 2
    initial_bits = min(
        internal,
        max(
            1,
            math.ceil(settings.get("initial_factor", 2.0) * math.log2(max(2, sparsity)))
            + settings.get("initial_offset", 1),
        ),
    )
    width = max(
        1,
        math.ceil(
            settings.get("extension_factor", 1.0) * math.log2(max(2, 2 * sparsity))
        ),
    )
    levels = [initial_bits] + list(range(initial_bits + width, internal, width))
    if levels[-1] != internal:
        levels.append(internal)
    retained = np.array([0], dtype=np.uint64)
    previous = 0
    stages = []
    sample_history = {}
    total_seconds = 0.0
    keep = max(
        sparsity,
        math.ceil(settings.get("retention", 2.0 if robust else 1.0) * sparsity),
    )
    validated = settings.get("validated", True) and not robust
    attempts = settings.get("attempts", 2) if validated else 1
    mode = "robust" if robust else "validated" if validated else "unvalidated"
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    for stage, length in enumerate(levels):
        extension = length - previous
        stage_key = f"{key}/stage/{stage}"
        stage_seed = int(rng.integers(0, 2**63 - 1))
        current_prefixes = retained.copy()
        old_length = previous

        def run_stage():
            evaluation_started = oracle.actual_evaluation_seconds
            local_rng = np.random.default_rng(stage_seed)
            columns = len(current_prefixes) * (1 << extension)
            if columns > settings.get("max_local_columns", 2_000_000):
                raise MemoryError(f"Local dictionary has {columns:,} columns")
            natural = max(
                1.0, sparsity * math.log(max(math.e, columns / max(1, sparsity)))
            )
            desired = max(
                sparsity + 1, math.ceil(settings.get("sample_factor", 2.4) * natural)
            )
            allocated = None
            if oracle.budget is not None:
                remaining_weight = math.log(max(math.e, columns / max(1, sparsity))) + (
                    internal - length
                ) * math.log(2)
                fraction = (
                    math.log(max(math.e, columns / max(1, sparsity))) / remaining_weight
                )
                allocated = math.floor((oracle.budget - oracle.count) * fraction)
                if allocated < 1:
                    raise BudgetExceeded("No remaining stage budget")
            validation_count = (
                math.ceil(settings.get("validation_factor", 0.25) * sparsity)
                if validated
                else 0
            )
            validation_count = (
                max(1, settings.get("validation_min", 8), validation_count)
                if validated
                else 0
            )
            if allocated is not None:
                if validated and allocated < 2:
                    raise BudgetExceeded(
                        "Stage budget cannot reserve independent validation"
                    )
                validation_count = min(
                    validation_count, max(int(validated), allocated // 8)
                )
                desired = max(1, allocated - validation_count)
            distinct = settings.get("sampling", "distinct") == "distinct"
            validations = (
                fresh_points(
                    oracle, query_map, internal, length, validation_count, local_rng
                )
                if distinct
                else local_rng.integers(
                    0, 1 << length, validation_count, dtype=np.uint64
                )
            )
            validation_count = len(validations)
            validation_values = None
            if validation_count:
                validation_values = oracle(map_points(validations, query_map))
            best = None
            best_residual = float("inf")
            reports = []
            for attempt in range(attempts):
                if allocated is not None and attempt:
                    break
                rows = (
                    fresh_points(
                        oracle, query_map, internal, length, desired, local_rng
                    )
                    if distinct
                    else local_rng.integers(0, 1 << length, desired, dtype=np.uint64)
                )
                if not len(rows):
                    rows = local_rng.integers(
                        0, 1 << length, min(desired, 1 << length), dtype=np.uint64
                    )
                sampled = oracle(map_points(rows, query_map))
                reused_rows = 0
                if (
                    settings.get("reuse_samples", True)
                    and not robust
                    and sample_history
                ):
                    reused_rows = len(sample_history)
                    fitting_rows = np.concatenate(
                        [np.asarray(list(sample_history), dtype=np.uint64), rows]
                    )
                    fitting_values = np.concatenate(
                        [np.asarray(list(sample_history.values())), sampled]
                    )
                else:
                    fitting_rows, fitting_values = rows, sampled
                operator = CosetOperator(
                    fitting_rows, current_prefixes, old_length, extension
                )
                response = fitting_values / math.sqrt(len(fitting_rows))
                values, report = solve(
                    operator,
                    response,
                    settings,
                    store,
                    f"{stage_key}/solver/{attempt}",
                    min(sparsity, columns),
                )
                sample_history.update(
                    {int(row): float(value) for row, value in zip(rows, sampled)}
                )
                if robust:
                    if tail_budget is None or tail_budget <= 0:
                        raise ValueError(
                            "Robust recovery needs an explicit positive tail budget"
                        )
                    coarse = np.zeros(columns)
                    selected = top_indices(values, keep)
                    coarse[selected] = values[selected]
                    groups = settings.get("mom_groups", 5)
                    group_size = max(
                        8, math.ceil(settings.get("mom_factor", 2.0) * sparsity)
                    )
                    if allocated is not None:
                        raise ValueError(
                            "Budgeted robust recovery requires a separate refinement budget"
                        )
                    refinements = []
                    for group in range(groups):
                        refined_rows = local_rng.integers(
                            0, 1 << length, group_size, dtype=np.uint64
                        )
                        refined_operator = CosetOperator(
                            refined_rows, current_prefixes, old_length, extension
                        )
                        observed = oracle(
                            map_points(refined_rows, query_map)
                        ) / math.sqrt(group_size)
                        residual = observed - refined_operator @ coarse
                        refinements.append(refined_operator.rmatvec(residual))
                    values = coarse + np.median(np.asarray(refinements), axis=0)
                    selected = top_indices(values, keep)
                    residual = float(np.linalg.norm(operator @ values - response))
                else:
                    values = debias(operator, response, values, sparsity)
                    selected = top_indices(values, sparsity)
                    threshold = settings.get("coefficient_threshold", 1e-8) * max(
                        1.0, np.max(np.abs(values), initial=0)
                    )
                    selected = selected[np.abs(values[selected]) > threshold]
                    if validation_count:
                        predicted = (
                            characters(validations, operator.frequencies[selected])
                            @ values[selected]
                        )
                        residual = float(
                            np.linalg.norm(predicted - validation_values)
                            / max(1.0, np.linalg.norm(validation_values))
                        )
                    else:
                        residual = float(
                            np.linalg.norm(operator @ values - response)
                            / max(1.0, np.linalg.norm(response))
                        )
                reports.append(report)
                candidate = {
                    "retained": operator.frequencies[selected].tolist(),
                    "values": values[selected].tolist(),
                    "accepted": bool(
                        residual <= settings.get("validation_tolerance", 1e-6) or robust
                    ),
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
            sample_history.update(
                {
                    int(row): float(value)
                    for row, value in zip(
                        validations,
                        [] if validation_values is None else validation_values,
                    )
                }
            )
            best["sample_history"] = sample_history
            best["resumed_seconds"] = sum(
                report.get("resumed_seconds", 0.0) for report in reports
            )
            best["evaluation_wall_seconds"] = (
                oracle.actual_evaluation_seconds - evaluation_started
            )
            return best

        result = store.memo(stage_key, run_stage)
        oracle.restore(result["oracle_state"])
        sample_history = {
            int(row): value for row, value in result["sample_history"].items()
        }
        result = {
            name: value
            for name, value in result.items()
            if name not in {"oracle_state", "sample_history"}
        }
        logging.info(
            "prefix %s stage=%d/%d bits=%d columns=%d accepted=%s queries=%d",
            mode,
            stage + 1,
            len(levels),
            length,
            result["columns"],
            result["accepted"],
            oracle.count,
        )
        stages.append(result)
        total_seconds += result["seconds"]
        retained = np.asarray(result["retained"], dtype=np.uint64)
        previous = length
        if validated and not result["accepted"]:
            return sparse_result(
                [],
                [],
                status="validation_failed",
                stages=stages,
                rounds=len(stages),
                algorithm_seconds=total_seconds + preprocessing_seconds,
                preprocessing_seconds=preprocessing_seconds,
                mode=mode,
            )
        if not len(retained):
            return sparse_result(
                [],
                [],
                status="empty_support",
                stages=stages,
                rounds=len(stages),
                algorithm_seconds=total_seconds + preprocessing_seconds,
                preprocessing_seconds=preprocessing_seconds,
                mode=mode,
            )
    decoding_started = time.perf_counter()
    values = np.asarray(stages[-1]["values"])
    selected = top_indices(values, sparsity)
    unmixed = (bits(retained[selected], internal) @ inverse.T) % 2
    syndromes = integers(unmixed)
    frequencies, amplitudes, invalid = [], [], 0
    for frequency, value in zip(syndromes, values[selected]):
        try:
            frequencies.append(decoder(int(frequency)))
            amplitudes.append(value)
        except ValueError:
            invalid += 1
    decoding_seconds = time.perf_counter() - decoding_started
    return sparse_result(
        frequencies,
        amplitudes,
        status="ok" if invalid == 0 else "invalid_syndrome",
        stages=stages,
        rounds=len(stages),
        algorithm_seconds=total_seconds + preprocessing_seconds + decoding_seconds,
        preprocessing_seconds=preprocessing_seconds,
        decoding_seconds=decoding_seconds,
        mode=mode,
        internal_dimension=internal,
        invalid_syndromes=invalid,
        tail_budget=tail_budget,
    )


def global_recover(
    oracle, dimension, sparsity, rows, settings, store, key, seed, degree=None
):
    from .algebra import degree_dictionary

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
