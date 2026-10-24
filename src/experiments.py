import itertools
import logging
import math
import signal
import time
import traceback

import numpy as np

from . import data as original
from .data import (
    BudgetExceeded,
    Oracle,
    Store,
    digest,
    graph_signal,
    random_signal,
    tail_bound,
)
from .hpo import run_hpo
from .methods import cs, stobbe, swht
from .methods.cs import global_recover
from .methods.spht import bch_map, prefix_recover
from .methods.stobbe import stobbe_robust_recover
from .methods.swht import swht_recover
from .numerics import InfeasibleSystem, coefficient_error, warmup


def expand_grid(grid):
    names = list(grid)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def method_settings(config, experiment, method, parameters=None):
    return {
        **config.get("defaults", {}),
        **config["methods"][method],
        **experiment.get("method_settings", {}).get(method, {}),
        **(parameters or {}),
    }


def jobs(config):
    methods = config["methods"]
    for experiment in config["experiments"]:
        kind = experiment["kind"]
        repetitions = experiment.get("trials", config.get("trials", 20))
        if kind == "phase":
            for size, delta, rho, distribution in itertools.product(
                experiment["sizes"],
                experiment["deltas"],
                experiment["rhos"],
                experiment.get("distributions", ["rademacher"]),
            ):
                base = {
                    "kind": kind,
                    "size": size,
                    "delta": delta,
                    "rho": rho,
                    "distribution": distribution,
                }
                for trial in range(repetitions):
                    seed = int(
                        digest(
                            [
                                config.get("seed", 20261023),
                                config.get("split", "evaluation"),
                                base,
                                trial,
                            ]
                        )[:8],
                        16,
                    )
                    for method in experiment.get("methods", methods):
                        for parameters in expand_grid(
                            experiment.get("parameter_grids", {}).get(method, {})
                        ):
                            yield {
                                "spec": base,
                                "trial": trial,
                                "seed": seed,
                                "method": method,
                                "settings": method_settings(
                                    config, experiment, method, parameters
                                ),
                            }
        elif kind in {"graph_edges", "graph_vertices"}:
            for snapshot in experiment.get("snapshots", [1, 3, 5, 7]):
                if "vertices" in kind:
                    cells = [
                        (n, experiment.get("edges", 50))
                        for n in experiment["vertices"]
                        if snapshot != 5 or n >= 70
                    ]
                else:
                    cells = [
                        (experiment.get("vertices", 100), e)
                        for e in experiment["edges"]
                    ]
                for dimension, edges in cells:
                    base = {
                        "kind": kind,
                        "snapshot": snapshot,
                        "dimension": dimension,
                        "edges": edges,
                    }
                    for trial in range(repetitions):
                        seed = int(
                            digest(
                                [
                                    config.get("seed", 20261023),
                                    config.get("split", "evaluation"),
                                    base,
                                    trial,
                                ]
                            )[:8],
                            16,
                        )
                        for method in experiment.get("methods", methods):
                            if dimension > experiment.get(
                                "method_max_dimensions", {}
                            ).get(method, float("inf")):
                                continue
                            for parameters in expand_grid(
                                experiment.get("parameter_grids", {}).get(method, {})
                            ):
                                yield {
                                    "spec": base,
                                    "trial": trial,
                                    "seed": seed,
                                    "method": method,
                                    "settings": method_settings(
                                        config, experiment, method, parameters
                                    ),
                                }
        elif kind in {"scaling", "robust"}:
            for cell in expand_grid(experiment["grid"]):
                base = {"kind": kind, **cell}
                for trial in range(repetitions):
                    seed = int(
                        digest(
                            [
                                config.get("seed", 20261023),
                                config.get("split", "evaluation"),
                                base,
                                trial,
                            ]
                        )[:8],
                        16,
                    )
                    for method in experiment.get("methods", methods):
                        yield {
                            "spec": base,
                            "trial": trial,
                            "seed": seed,
                            "method": method,
                            "settings": method_settings(config, experiment, method),
                        }
        elif kind == "hpo":
            for cell in expand_grid(experiment["grid"]):
                base = {"kind": kind, **experiment.get("settings", {}), **cell}
                for trial in range(repetitions):
                    seed = int(
                        digest(
                            [
                                config.get("seed", 20261023),
                                config.get("split", "evaluation"),
                                base["task"],
                                trial,
                            ]
                        )[:8],
                        16,
                    )
                    for method in experiment.get("methods", methods):
                        yield {
                            "spec": base,
                            "trial": trial,
                            "seed": seed,
                            "method": method,
                            "settings": method_settings(config, experiment, method),
                        }
        else:
            raise ValueError(f"Unknown experiment kind {kind}")


def execute(job, database, upstream_directory, implementation):
    warmup()
    store = Store(database)
    key = digest({"job": job, "implementation": implementation})
    if store.get(key) is not None:
        store.close()
        return "cached"
    spec, method, settings, seed = (
        job["spec"],
        job["method"],
        job["settings"],
        job["seed"],
    )
    started = time.perf_counter()
    oracle = None
    try:
        if spec["kind"] == "hpo":
            result = run_hpo(spec, method, settings, store, key, seed, implementation)
        else:
            if spec["kind"] == "phase":
                dimension = spec["size"]
                budget = max(1, round(spec["delta"] * (1 << dimension)))
                sparsity = max(1, min(budget, round(spec["rho"] * budget)))
                degree = None
                signal = random_signal(dimension, sparsity, seed, spec["distribution"])
                rows = budget
            else:
                dimension, sparsity = spec["dimension"], spec["sparsity"]
                degree = spec.get("degree")
                signal = random_signal(
                    dimension,
                    sparsity,
                    seed,
                    spec.get("distribution", "gaussian"),
                    degree,
                    spec.get("tail_ratio", 0),
                    spec.get("tail_size", 0),
                )
                budget = None
                dictionary_size = (
                    1 << dimension
                    if degree is None
                    else sum(math.comb(dimension, d) for d in range(degree + 1))
                )
                rows = math.ceil(
                    settings.get("sample_factor", 2.4)
                    * sparsity
                    * math.log(max(math.e, dictionary_size / sparsity))
                )
            namespace = digest(
                {"spec": spec, "seed": seed, "implementation": implementation}
            )
            oracle = Oracle(signal, dimension, store, namespace, budget)
            algorithm_started = time.perf_counter()
            if method.startswith("prefix"):
                robust = "robust" in method
                error_budget = tail_bound(signal, sparsity)
                result = prefix_recover(
                    oracle,
                    dimension,
                    sparsity,
                    settings,
                    store,
                    key,
                    seed + 1709,
                    degree=degree if "bch" in method else None,
                    robust=robust and error_budget > 0,
                    tail_budget=error_budget,
                )
            elif method == "swht":
                result = swht_recover(
                    oracle,
                    dimension,
                    sparsity,
                    settings,
                    upstream_directory,
                    seed,
                )
            elif method == "stobbe" and spec["kind"] == "robust":
                result = stobbe_robust_recover(
                    oracle, dimension, sparsity, rows, settings, seed + 1709
                )
            else:
                result = global_recover(
                    oracle,
                    dimension,
                    sparsity,
                    rows,
                    settings,
                    store,
                    key,
                    seed + 1709,
                    degree=degree,
                )
            duration = time.perf_counter() - algorithm_started
            error = coefficient_error(signal.truth(), result)
            result.update(
                relative_error=error,
                success=bool(error <= 1e-6 and result["status"] == "ok"),
                queries=oracle.count,
                sampled_rows=oracle.requests,
                oracle_seconds=oracle.evaluation_seconds,
                algorithm_seconds=result.get(
                    "algorithm_seconds",
                    duration + result.get("resumed_compute_seconds", 0.0),
                ),
                measured_algorithm_seconds=duration,
                sparsity=sparsity,
                N=1 << dimension,
                degree=degree,
            )
            if spec["kind"] == "robust":
                bound = tail_bound(signal, sparsity)
                norm = np.linalg.norm(signal.values)
                result["error_over_E"] = error * norm / max(bound, 1e-30)
                result["tail_budget"] = bound
            result["wall_seconds"] = time.perf_counter() - started
    except (
        BudgetExceeded,
        MemoryError,
        RuntimeError,
        ValueError,
        AssertionError,
        TimeoutError,
    ) as error:
        result = {
            "status": type(error).__name__,
            "message": str(error),
            "success": False,
            "queries": 0 if oracle is None else oracle.count,
            "sampled_rows": 0 if oracle is None else oracle.requests,
            "wall_seconds": time.perf_counter() - started,
        }
        logging.warning("trial %s %s: %s", method, spec, error)
    result.update(job=job, implementation=implementation)
    store.finish(key, result)
    store.close()
    logging.info(
        "trial %s status=%s success=%s queries=%s seconds=%.3f",
        method,
        result["status"],
        result.get("success"),
        result.get("queries"),
        result.get("wall_seconds", 0),
    )
    return result["status"]


logger = logging.getLogger(__name__)


class TimedStore(Store):
    def __init__(self, path):
        self.io_seconds = 0.0
        self.io_cpu_seconds = 0.0
        super().__init__(path)

    def get(self, key):
        started = time.perf_counter()
        cpu = time.process_time()
        try:
            return super().get(key)
        finally:
            self.io_seconds += time.perf_counter() - started
            self.io_cpu_seconds += time.process_time() - cpu

    def put(self, key, value, kind="checkpoint"):
        started = time.perf_counter()
        cpu = time.process_time()
        try:
            return super().put(key, value, kind)
        finally:
            self.io_seconds += time.perf_counter() - started
            self.io_cpu_seconds += time.process_time() - cpu

    def save_arrays(self, key, **arrays):
        return None


class TimedOracle(Oracle):
    def __init__(self, *args, **kwargs):
        self.call_seconds = 0.0
        self.cpu_seconds = 0.0
        self.locations = set()
        super().__init__(*args, **kwargs)
        self.shape = (2,) * self.dimension

    def __call__(self, points):
        started = time.perf_counter()
        cpu = time.process_time()
        packed = np.packbits(
            np.atleast_2d(np.asarray(points, dtype=np.uint8)), axis=1, bitorder="little"
        )
        self.locations.update(row.tobytes() for row in packed)
        try:
            return super().__call__(points)
        finally:
            self.call_seconds += time.perf_counter() - started
            self.cpu_seconds += time.process_time() - cpu

    def __getitem__(self, point):
        return float(self(np.asarray(point, dtype=np.uint8)[None, :])[0])


class CutFunction:
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, points):
        return np.asarray([self.graph[tuple(int(x) for x in row)] for row in points])


def timeout_handler(signum, frame):
    raise TimeoutError("Common trial wall-time limit exceeded")


def evaluate(job, database, implementation, max_seconds):
    warmup()
    if job["method"] == "cs":
        cs.prepare()
    store = TimedStore(database)
    key = digest({"job": job, "implementation": implementation})
    existing = store.get(key)
    if existing is not None:
        store.close()
        return existing
    with store.db:
        store.db.execute("DELETE FROM records WHERE key LIKE ?", (key + "/%",))
    spec, settings, method, seed = (
        job["spec"],
        job["settings"],
        job["method"],
        job["seed"],
    )
    dimension = spec["dimension"]
    is_graph = spec["kind"].startswith("graph")
    if is_graph:
        modules = original.load()
    if is_graph:
        graph = original.graph(spec)
        truth = graph_signal(dimension, sorted(graph.graph)).truth()
        function = CutFunction(graph)
        sparsity, degree = len(graph.graph) + 1, 2
        modules.wht(np.zeros((2, 2)))
    else:
        sparsity, degree = spec["sparsity"], spec.get("degree")
        function = random_signal(dimension, sparsity, seed, "gaussian", degree)
        truth = function.truth()
    preparation = time.perf_counter()
    if method == "prefix_bch":
        bch_map(dimension, degree)
    if method == "cs" and degree is not None:
        columns = (
            1 + dimension * (dimension - 1) // 2
            if is_graph
            else sum(math.comb(dimension, j) for j in range(degree + 1))
        )
        if columns <= settings["max_global_columns"]:
            cs.dictionary(dimension, degree, is_graph, settings["max_global_columns"])
    bch_precompute_seconds = time.perf_counter() - preparation
    oracle = TimedOracle(function, dimension, store, key)
    rng_state = np.random.get_state()
    np.random.seed(seed % (2**32))
    io_started = store.io_seconds
    io_cpu_started = store.io_cpu_seconds
    started = time.perf_counter()
    cpu_started = time.process_time()
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    result = {}
    try:
        if method == "swht":
            recovered = swht.recover_graph(oracle, dimension, sparsity - 1, settings)
        elif method == "stobbe":
            recovered = stobbe.recover_graph(oracle, dimension, sparsity - 1, settings)
        else:
            if method.startswith("prefix"):
                numerical = prefix_recover(
                    oracle,
                    dimension,
                    sparsity,
                    settings,
                    store,
                    key,
                    seed + 1709,
                    degree=degree if "bch" in method else None,
                )
            elif method == "cs":
                factor = settings["sample_factor"]
                rows = cs.sample_count(spec, sparsity, factor)
                numerical = cs.recover(
                    oracle,
                    dimension,
                    sparsity,
                    rows,
                    settings,
                    store,
                    key,
                    seed + 1709,
                    degree,
                    is_graph,
                )
            else:
                raise ValueError(method)
            result["numerical_status"] = numerical["status"]
            result["solver_report"] = numerical.get("solver_report")
            result["relative_error"] = coefficient_error(truth, numerical)
            if is_graph:
                recovered = {}
                for frequency, value in zip(
                    numerical["frequencies"], numerical["values"]
                ):
                    point = np.zeros(dimension, dtype=np.uint8)
                    point[list(frequency)] = 1
                    recovered[tuple(point.tolist())] = float(value)
            else:
                result["success"] = bool(
                    result["relative_error"] <= 1e-6 and numerical["status"] == "ok"
                )
                result["recovered_frequencies"] = numerical["frequencies"]
                result["recovered_values"] = numerical["values"]
                result["truth_digest"] = digest(truth)
        if is_graph:
            try:
                estimated = modules.graph.create_from_FT(dimension, dict(recovered))
                result["success"] = bool(graph == estimated)
                result["recovered_edges"] = [
                    [int(a), int(b), float(weight)]
                    for (a, b), weight in sorted(estimated.graph.items())
                ]
                result["truth_digest"] = digest(sorted(graph.graph.items()))
                result["edge_error"] = graph.reconstError(estimated) / max(
                    1, len(graph.graph)
                )
            except (AssertionError, ValueError):
                result.update(success=False, edge_error=1.0)
        result["status"] = "ok" if result["success"] else "recovery_failed"
    except (BudgetExceeded, InfeasibleSystem) as error:
        result.update(status="recovery_failed", success=False, message=str(error))
    except TimeoutError as error:
        result.update(status="timeout", success=False, message=str(error))
    except MemoryError as error:
        result.update(status="resource_limit", success=False, message=str(error))
    except Exception as error:
        logger.exception("Trial implementation error")
        result.update(
            status="implementation_error",
            success=False,
            message=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        np.random.set_state(rng_state)
    elapsed = time.perf_counter() - started
    io_seconds = store.io_seconds - io_started
    cpu_elapsed = time.process_time() - cpu_started
    result.update(
        job=job,
        implementation=implementation,
        queries=len(oracle.locations),
        requested_queries=oracle.requests,
        wall_seconds=elapsed,
        process_cpu_seconds=cpu_elapsed,
        bch_precompute_seconds=bch_precompute_seconds,
        oracle_wrapper_seconds=oracle.call_seconds,
        checkpoint_seconds=io_seconds,
        optimizer_wall_seconds=max(1e-9, elapsed - oracle.call_seconds - io_seconds),
        algorithm_seconds=max(
            1e-9,
            cpu_elapsed - oracle.cpu_seconds - (store.io_cpu_seconds - io_cpu_started),
        ),
        runtime_censored=result["status"] in {"timeout", "resource_limit"},
        time_limit=max_seconds,
    )
    store.finish(key, result)
    store.close()
    logger.info(
        "%s %s n=%s k=%s repeat=%d status=%s queries=%d optimizer=%.3fs wall=%.3fs",
        job.get("role", "evaluation"),
        method,
        dimension,
        sparsity,
        job["trial"] + 1,
        result["status"],
        result["queries"],
        result["algorithm_seconds"],
        elapsed,
    )
    if result["status"] == "implementation_error":
        raise RuntimeError(result["traceback"])
    return result
