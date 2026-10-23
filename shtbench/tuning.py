import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .experiments import execute, expand_grid
from .storage import Store, atomic_text, canonical, digest, source_digest


def calibration_jobs(config, settings, repetitions, split):
    for cell in config["cells"]:
        size = 1 << cell["dimension"]
        queries = round(cell["queries"])
        spec = {
            "kind": "phase",
            "size": cell["dimension"],
            "delta": queries / size,
            "rho": cell["sparsity"] / queries,
            "distribution": cell.get("distribution", "rademacher"),
        }
        for trial in range(repetitions):
            seed = int(digest([config["seed"], split, spec, trial])[:8], 16)
            for method in ["cs", "prefix"]:
                method_settings = {
                    **config["defaults"],
                    **(config["baseline"] if method == "cs" else settings),
                }
                yield {
                    "spec": spec,
                    "method": method,
                    "settings": method_settings,
                    "trial": trial,
                    "seed": seed,
                    "split": split,
                }


def task_payload(payload):
    from threadpoolctl import threadpool_limits

    job, database, directory, implementation = payload
    with threadpool_limits(limits=1):
        return execute(job, database, directory, implementation)


def run_jobs(jobs, database, directory, implementation, workers):
    store = Store(database)
    pending = [
        job
        for job in jobs
        if store.get(digest({"job": job, "implementation": implementation})) is None
    ]
    store.close()
    payloads = [(job, database, directory, implementation) for job in pending]
    if workers == 1:
        for payload in payloads:
            task_payload(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for completed in pool.map(task_payload, payloads, chunksize=1):
                pass


def score(config, settings, repetitions, split, store, implementation):
    cells = defaultdict(lambda: {"cs": [], "prefix": []})
    speedups = []
    multiround = 0
    jobs = list(calibration_jobs(config, settings, repetitions, split))
    pairs = defaultdict(dict)
    for job in jobs:
        key = digest({"job": job, "implementation": implementation})
        result = store.get(key)
        if result is None:
            raise RuntimeError("Incomplete calibration results")
        cells[canonical(job["spec"])][job["method"]].append(bool(result.get("success")))
        pairs[canonical([job["spec"], job["trial"]])][job["method"]] = result
        if job["method"] == "prefix":
            multiround += result.get("rounds", 0) > 1
    gap = np.mean(
        [
            abs(np.mean(values["prefix"]) - np.mean(values["cs"]))
            for values in cells.values()
        ]
    )
    deficit = np.mean(
        [
            max(0.0, np.mean(values["cs"]) - np.mean(values["prefix"]))
            for values in cells.values()
        ]
    )
    own_success = np.mean([np.mean(values["prefix"]) for values in cells.values()])
    for pair in pairs.values():
        if pair["cs"].get("success") and pair["prefix"].get("success"):
            speedups.append(
                pair["cs"]["algorithm_seconds"]
                / max(pair["prefix"]["algorithm_seconds"], 1e-12)
            )
    return {
        "settings": settings,
        "mean_absolute_probability_gap": float(gap),
        "mean_success_deficit": float(deficit),
        "prefix_success_rate": float(own_success),
        "median_paired_speedup": None if not speedups else float(np.median(speedups)),
        "multiround_trials": multiround,
        "trials": sum(len(value["prefix"]) for value in cells.values()),
        "per_cell": [
            {
                "spec": json.loads(cell),
                "cs_success": float(np.mean(values["cs"])),
                "prefix_success": float(np.mean(values["prefix"])),
            }
            for cell, values in cells.items()
        ],
    }


def tune(config, output, workers=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "cache.sqlite"
    implementation = source_digest()
    rng = np.random.default_rng(config["seed"])
    candidates = expand_grid(config["grid"])
    maximum = config.get("max_candidates", len(candidates))
    if len(candidates) > maximum:
        indices = rng.choice(len(candidates), maximum, replace=False)
        candidates = [candidates[i] for i in indices]
    archive = []
    initial_count = len(candidates)
    for repetitions in config["rungs"]:
        logging.info(
            "calibration rung repetitions=%d candidates=%d",
            repetitions,
            len(candidates),
        )
        queue = {}
        for candidate in candidates:
            settings = {**config["prefix"], **candidate}
            for job in calibration_jobs(config, settings, repetitions, "calibration"):
                queue[digest(job)] = job
        run_jobs(
            list(queue.values()),
            str(database),
            config.get("upstream_directory", "data/upstream"),
            implementation,
            workers,
        )
        store = Store(database)
        scores = [
            score(
                config,
                {**config["prefix"], **candidate},
                repetitions,
                "calibration",
                store,
                implementation,
            )
            for candidate in candidates
        ]
        store.close()
        scores.sort(
            key=lambda result: (
                result["mean_success_deficit"],
                -result["prefix_success_rate"],
                result["mean_absolute_probability_gap"],
                digest(result["settings"]),
            )
        )
        archive.append({"repetitions": repetitions, "scores": scores})
        atomic_text(
            output / "search.json",
            json.dumps(
                {
                    "implementation": implementation,
                    "candidate_count": initial_count,
                    "rungs": archive,
                },
                indent=2,
            ),
        )
        candidates = [
            {key: result["settings"][key] for key in config["grid"]}
            for result in scores[: max(1, len(scores) // config.get("reduction", 4))]
        ]
    selected = archive[-1]["scores"][0]
    validation_trials = config.get("validation_trials", 16)
    heldout_jobs = list(
        calibration_jobs(config, selected["settings"], validation_trials, "validation")
    )
    run_jobs(
        heldout_jobs,
        str(database),
        config.get("upstream_directory", "data/upstream"),
        implementation,
        workers,
    )
    store = Store(database)
    heldout = score(
        config,
        selected["settings"],
        validation_trials,
        "validation",
        store,
        implementation,
    )
    store.close()
    frozen = {
        "selected_settings": selected["settings"],
        "calibration": selected,
        "held_out_validation": heldout,
        "candidate_count": initial_count,
        "implementation": implementation,
        "config": config,
        "timing_workers": workers,
        "selection_rule": "Minimize recovery deficit, then maximize recovery rate, then minimize absolute probability gap; timing does not select parameters",
    }
    atomic_text(output / "selected.json", json.dumps(frozen, indent=2))
    logging.info(
        "selected calibration gap=%.4f held-out gap=%.4f",
        selected["mean_absolute_probability_gap"],
        heldout["mean_absolute_probability_gap"],
    )
    return frozen
