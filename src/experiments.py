import copy
import itertools
import logging
import math
import signal
import time
import traceback

import numpy as np

from . import original_code
from .config import CONFIG
from .hpo import run_hpo
from .methods import cs, spht, stobbe, swht
from .numerics import InfeasibleSystem, coefficient_error, warmup
from .signals import graph_signal, random_signal, tail_bound
from .store import BudgetExceeded, Oracle, Store, digest

EVALUATED = {"graphs", "scaling"}


def grid(values):
    return [dict(zip(values, cell)) for cell in itertools.product(*values.values())]


def cells(experiment):
    kind = experiment["kind"]
    if kind == "phase":
        for size, delta, rho, distribution in itertools.product(
            experiment["sizes"], experiment["deltas"], experiment["rhos"], experiment["distributions"]
        ):
            yield {"kind": kind, "size": size, "delta": delta, "rho": rho, "distribution": distribution}
    elif kind.startswith("graph"):
        for snapshot in [1, 3, 5, 7]:
            if kind == "graph_vertices":
                pairs = [(n, experiment["edges"]) for n in experiment["vertices"] if snapshot != 5 or n >= 70]
            else:
                pairs = [(experiment["vertices"], e) for e in experiment["edges"]]
            for dimension, edges in pairs:
                yield {"kind": kind, "snapshot": snapshot, "dimension": dimension, "edges": edges}
    else:
        for cell in grid(experiment["grid"]):
            yield {"kind": kind, **experiment.get("settings", {}), **cell}


def jobs(suite):
    config = copy.deepcopy(CONFIG["suites"][suite])
    policy = CONFIG["evaluation"]
    for experiment in config["experiments"]:
        for spec in cells(experiment):
            for trial in range(experiment.get("trials", config["trials"])):
                identity = spec["task"] if spec["kind"] == "hpo" else spec
                seed = int(digest([config["seed"], config["split"], identity, trial])[:8], 16)
                for method in experiment.get("methods", config["methods"]):
                    if spec.get("dimension", 0) > experiment.get("method_max_dimensions", {}).get(method, math.inf):
                        continue
                    settings = {
                        **config["defaults"],
                        **config["methods"][method],
                        **experiment.get("method_settings", {}).get(method, {}),
                    }
                    job = {"spec": spec, "trial": trial, "seed": seed, "method": method, "settings": settings}
                    if suite in EVALUATED:
                        evaluation_settings(job, policy)
                    yield job


def evaluation_settings(job, policy):
    settings = job["settings"]
    settings.update(
        max_dense_bytes=policy["max_dense_bytes"],
        max_global_columns=policy["max_global_columns"],
        max_local_columns=policy["max_local_columns"],
        solver_seconds=policy["time_limit_seconds"],
    )
    if job["method"] == "cs":
        settings.pop("max_iterations", None)
        settings.pop("threshold", None)
    else:
        settings.update(
            max_iterations=min(settings.get("max_iterations", policy["max_iterations"]), policy["max_iterations"]),
            tolerance=policy["tolerance"],
        )
    if job["method"] in {"stobbe", "swht"}:
        published = original_code.published_setting(job["spec"], job["method"])
        settings.update(published["settings"])
        job["published_configuration"] = published
    job["role"] = "evaluation"


def key(job, implementation):
    return digest({"job": job, "implementation": implementation})


def problem(spec, settings, seed):
    if spec["kind"] == "phase":
        dimension = spec["size"]
        budget = max(1, round(spec["delta"] * (1 << dimension)))
        sparsity = max(1, min(budget, round(spec["rho"] * budget)))
        return random_signal(dimension, sparsity, seed, spec["distribution"]), dimension, sparsity, None, budget, budget
    dimension, sparsity, degree = spec["dimension"], spec["sparsity"], spec.get("degree")
    signal = random_signal(
        dimension, sparsity, seed, spec.get("distribution", "gaussian"), degree,
        spec.get("tail_ratio", 0), spec.get("tail_size", 0),
    )
    columns = 1 << dimension if degree is None else sum(math.comb(dimension, d) for d in range(degree + 1))
    rows = math.ceil(settings.get("sample_factor", 2.4) * sparsity * math.log(max(math.e, columns / sparsity)))
    return signal, dimension, sparsity, degree, None, rows


def execute(job, database, implementation):
    warmup()
    store = Store(database)
    trial_key = key(job, implementation)
    if store.get(trial_key) is not None:
        store.close()
        return
    spec, method, settings, seed = job["spec"], job["method"], job["settings"], job["seed"]
    started = time.perf_counter()
    oracle = None
    try:
        if spec["kind"] == "hpo":
            result = run_hpo(spec, method, settings, store, trial_key, seed, implementation)
        else:
            signal, dimension, sparsity, degree, budget, rows = problem(spec, settings, seed)
            namespace = digest({"spec": spec, "seed": seed, "implementation": implementation})
            oracle = Oracle(signal, dimension, store, namespace, budget)
            algorithm_started = time.perf_counter()
            if method.startswith("prefix"):
                error_budget = tail_bound(signal, sparsity)
                result = spht.prefix_recover(
                    oracle, dimension, sparsity, settings, store, trial_key, seed + 1709,
                    degree=degree if "bch" in method else None,
                    robust="robust" in method and error_budget > 0, tail_budget=error_budget,
                )
            elif method == "swht":
                result = swht.recover(oracle, dimension, sparsity, settings, seed)
            elif method == "stobbe" and spec["kind"] == "robust":
                result = stobbe.recover(oracle, dimension, sparsity, rows, settings, seed + 1709)
            else:
                result = cs.recover(oracle, dimension, sparsity, rows, settings, store, trial_key, seed + 1709, degree=degree)
            duration = time.perf_counter() - algorithm_started
            error = coefficient_error(signal.truth(), result)
            result.update(
                relative_error=error,
                success=bool(error <= 1e-6 and result["status"] == "ok"),
                queries=oracle.count,
                sampled_rows=oracle.requests,
                oracle_seconds=oracle.evaluation_seconds,
                algorithm_seconds=result.get("algorithm_seconds", duration + result.get("resumed_compute_seconds", 0.0)),
                measured_algorithm_seconds=duration,
                sparsity=sparsity,
                N=1 << dimension,
                degree=degree,
            )
            if spec["kind"] == "robust":
                bound = tail_bound(signal, sparsity)
                result["error_over_E"] = error * np.linalg.norm(signal.values) / max(bound, 1e-30)
                result["tail_budget"] = bound
            result["wall_seconds"] = time.perf_counter() - started
    except (BudgetExceeded, MemoryError, RuntimeError, ValueError, AssertionError, TimeoutError) as error:
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
    store.finish(trial_key, result)
    store.close()
    logging.info(
        "trial %s status=%s success=%s queries=%s seconds=%.3f",
        method, result["status"], result.get("success"), result.get("queries"), result.get("wall_seconds", 0),
    )


class TimedStore(Store):
    def __init__(self, path):
        self.io_seconds = 0.0
        self.io_cpu_seconds = 0.0
        super().__init__(path)

    def timed(self, function, *args):
        started, cpu = time.perf_counter(), time.process_time()
        try:
            return function(*args)
        finally:
            self.io_seconds += time.perf_counter() - started
            self.io_cpu_seconds += time.process_time() - cpu

    def get(self, key):
        return self.timed(super().get, key)

    def put(self, key, value, kind="checkpoint"):
        return self.timed(super().put, key, value, kind)

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
        started, cpu = time.perf_counter(), time.process_time()
        packed = np.packbits(np.atleast_2d(np.asarray(points, dtype=np.uint8)), axis=1, bitorder="little")
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


def recover_numerically(oracle, spec, method, settings, store, trial_key, seed, dimension, sparsity, degree, is_graph):
    if method.startswith("prefix"):
        return spht.prefix_recover(
            oracle, dimension, sparsity, settings, store, trial_key, seed + 1709, degree=degree if "bch" in method else None
        )
    if method == "cs":
        rows = cs.sample_count(spec, sparsity, settings["sample_factor"])
        return cs.recover_spgl1(oracle, dimension, sparsity, rows, settings, seed + 1709, degree, is_graph)
    raise ValueError(method)


def evaluate(job, database, implementation, max_seconds):
    warmup()
    if job["method"] == "cs":
        cs.spgl1_solver()
    store = TimedStore(database)
    trial_key = key(job, implementation)
    if store.get(trial_key) is not None:
        store.close()
        return
    with store.db:
        store.db.execute("DELETE FROM records WHERE key LIKE ?", (trial_key + "/%",))
    spec, settings, method, seed = job["spec"], job["settings"], job["method"], job["seed"]
    dimension = spec["dimension"]
    is_graph = spec["kind"].startswith("graph")
    if is_graph:
        modules = original_code.load()
        graph = original_code.graph(spec)
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
        spht.bch_map(dimension, degree)
    if method == "cs" and degree is not None:
        if is_graph:
            columns = 1 + dimension * (dimension - 1) // 2
        else:
            columns = sum(math.comb(dimension, j) for j in range(degree + 1))
        if columns <= settings["max_global_columns"]:
            cs.dictionary(dimension, degree, is_graph, settings["max_global_columns"])
    bch_precompute_seconds = time.perf_counter() - preparation
    oracle = TimedOracle(function, dimension, store, trial_key)
    rng_state = np.random.get_state()
    np.random.seed(seed % (2**32))
    io_started, io_cpu_started = store.io_seconds, store.io_cpu_seconds
    started, cpu_started = time.perf_counter(), time.process_time()
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    result = {}
    try:
        if method == "swht":
            recovered = swht.recover_graph(oracle, dimension, sparsity - 1, settings)
        elif method == "stobbe":
            recovered = stobbe.recover_graph(oracle, dimension, sparsity - 1, settings)
        else:
            numerical = recover_numerically(
                oracle, spec, method, settings, store, trial_key, seed, dimension, sparsity, degree, is_graph
            )
            result["numerical_status"] = numerical["status"]
            result["solver_report"] = numerical.get("solver_report")
            result["relative_error"] = coefficient_error(truth, numerical)
            if is_graph:
                recovered = {}
                for frequency, value in zip(numerical["frequencies"], numerical["values"]):
                    point = np.zeros(dimension, dtype=np.uint8)
                    point[list(frequency)] = 1
                    recovered[tuple(point.tolist())] = float(value)
            else:
                result["success"] = bool(result["relative_error"] <= 1e-6 and numerical["status"] == "ok")
                result["recovered_frequencies"] = numerical["frequencies"]
                result["recovered_values"] = numerical["values"]
                result["truth_digest"] = digest(truth)
        if is_graph:
            try:
                estimated = modules.graph.create_from_FT(dimension, dict(recovered))
                result["success"] = bool(graph == estimated)
                result["recovered_edges"] = [[int(a), int(b), float(w)] for (a, b), w in sorted(estimated.graph.items())]
                result["truth_digest"] = digest(sorted(graph.graph.items()))
                result["edge_error"] = graph.reconstError(estimated) / max(1, len(graph.graph))
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
        logging.exception("Trial implementation error")
        result.update(status="implementation_error", success=False, message=str(error), traceback=traceback.format_exc())
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
        algorithm_seconds=max(1e-9, cpu_elapsed - oracle.cpu_seconds - (store.io_cpu_seconds - io_cpu_started)),
        runtime_censored=result["status"] in {"timeout", "resource_limit"},
        time_limit=max_seconds,
    )
    store.finish(trial_key, result)
    store.close()
    logging.info(
        "%s %s n=%s k=%s repeat=%d status=%s queries=%d optimizer=%.3fs wall=%.3fs",
        job["role"], method, dimension, sparsity, job["trial"] + 1, result["status"], result["queries"],
        result["algorithm_seconds"], elapsed,
    )
    if result["status"] == "implementation_error":
        raise RuntimeError(result["traceback"])
