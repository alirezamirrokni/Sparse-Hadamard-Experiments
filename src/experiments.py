import itertools
import logging
import math
import time

import numpy as np

from .algebra import warmup
from .hpo import run_hpo
from .methods.cs import global_recover
from .methods.spht import prefix_recover
from .methods.stobbe import stobbe_robust_recover
from .methods.swht import swht_recover
from .metrics import coefficient_error
from .signals import random_signal, tail_bound
from .storage import BudgetExceeded, Oracle, Store, digest


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
