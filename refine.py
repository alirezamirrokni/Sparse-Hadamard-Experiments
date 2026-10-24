import argparse
import fcntl
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import shutil
import sqlite3
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing

from threadpoolctl import threadpool_limits

import main
from src.experiments import execute, jobs
from src.plotting import aggregate, graph_operating_points, render
from src.storage import Store, atomic_text, digest, provenance

root = main.ROOT
out = root / "results" / "refinement"
logger = logging.getLogger(__name__)


def calibrate_graphs():
    report_path = out / "graph-report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    implementation = main.cache_implementation(root / "results", "graphs")
    config, _ = main.configuration(root / "config.json", "graphs")
    base = {**config["defaults"], **config["methods"]["prefix_bch"]}
    variants = []
    for initial, offset, extension, sample in itertools.product(
        [1.5, 1.75, 2.0], [0, 1, 2], [0.75, 1.0], [0.9, 1.1, 1.3, 1.5]
    ):
        variants.append(
            {
                "initial_factor": initial,
                "initial_offset": offset,
                "extension_factor": extension,
                "sample_factor": sample,
            }
        )
    variants.append(
        {
            "initial_factor": 2.0,
            "initial_offset": 1,
            "extension_factor": 1.0,
            "sample_factor": 2.4,
        }
    )
    database = out / "graph-search.sqlite"
    store = Store(database)

    def evaluate(parameters, cases, seeds, split):
        results = []
        for dimension, edges, snapshot in cases:
            spec = {
                "kind": "graph_vertices" if dimension != 100 else "graph_edges",
                "snapshot": snapshot,
                "dimension": dimension,
                "edges": edges,
            }
            for repeat in range(seeds):
                seed = int(digest(["refinement-graph-v1", split, spec, repeat])[:8], 16)
                settings = {**base, **parameters, "solver_seconds": 30}
                job = {
                    "spec": spec,
                    "trial": repeat,
                    "seed": seed,
                    "method": "prefix_bch",
                    "settings": settings,
                }
                execute(job, str(database), str(root / "data/upstream"), implementation)
                result = store.get(
                    digest({"job": job, "implementation": implementation})
                )
                results.append(result)
        success = sum(bool(r.get("success")) for r in results)
        ratio = statistics.mean(
            r["queries"]
            / (
                2.4
                * (r["job"]["spec"]["edges"] + 1)
                * math.log2(r["job"]["spec"]["dimension"])
            )
            for r in results
        )
        return {
            "parameters": parameters,
            "successes": success,
            "trials": len(results),
            "query_ratio": ratio,
            "mean_seconds": statistics.mean(r["wall_seconds"] for r in results),
        }

    with threadpool_limits(limits=1):
        ranked = []
        for i, p in enumerate(variants):
            record = evaluate(p, [(100, 50, 1), (800, 50, 3)], 1, "screen")
            ranked.append(record)
            atomic_text(out / "graph-screen.json", json.dumps(ranked, indent=2))
            if i % 10 == 0:
                print("screen", i + 1, "/", len(variants), record, flush=True)
        ranked.sort(
            key=lambda r: (-r["successes"], r["query_ratio"], r["mean_seconds"])
        )
        shortlist = []
        for record in ranked[:12]:
            evaluated = evaluate(
                record["parameters"],
                [
                    (50, 50, 1),
                    (100, 20, 3),
                    (100, 50, 3),
                    (100, 70, 7),
                    (200, 50, 7),
                    (800, 50, 1),
                ],
                3,
                "selection",
            )
            shortlist.append(evaluated)
            atomic_text(out / "graph-selection.json", json.dumps(shortlist, indent=2))
            print("selection", evaluated, flush=True)
        shortlist.sort(
            key=lambda r: (
                -(r["successes"] / r["trials"]),
                r["query_ratio"],
                r["mean_seconds"],
            )
        )
        best_rate = shortlist[0]["successes"] / shortlist[0]["trials"]
        eligible = [r for r in shortlist if r["successes"] / r["trials"] == best_rate]
        minimum_queries = min(r["query_ratio"] for r in eligible)
        winner = min(
            (r for r in eligible if r["query_ratio"] <= 1.05 * minimum_queries),
            key=lambda r: r["mean_seconds"],
        )
        validation = evaluate(
            winner["parameters"],
            [
                (50, 50, 1),
                (100, 20, 1),
                (100, 50, 5),
                (100, 70, 5),
                (200, 50, 3),
                (400, 50, 7),
                (800, 50, 7),
            ],
            12,
            "heldout-balanced",
        )
        report = {
            "screen_candidates": len(variants),
            "selection_rule": "minimum runtime within 5% of the lowest training query ratio at the best recovery rate",
            "selected": winner,
            "validation": validation,
        }
        atomic_text(out / "graph-report.json", json.dumps(report, indent=2))
        print("FINAL", json.dumps(report), flush=True)
    store.close()
    return json.loads(report_path.read_text())


def calibrate_phase():
    report_path = out / "phase-report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    implementation = main.cache_implementation(root / "results", "phase")
    config, _ = main.configuration(root / "config.json", "phase")
    base = {**config["defaults"], **config["methods"]["prefix"]}
    database = out / "phase-search.sqlite"
    store = Store(database)

    def evaluate(parameters, cases, repeats, split, method="prefix"):
        results = []
        for size, delta, rho in cases:
            spec = {
                "kind": "phase",
                "size": size,
                "delta": delta,
                "rho": rho,
                "distribution": "rademacher",
            }
            for repeat in range(repeats):
                seed = int(digest(["refinement-phase-v1", split, spec, repeat])[:8], 16)
                settings = {
                    **config["defaults"],
                    **(config["methods"]["cs"] if method == "cs" else base),
                    **parameters,
                    "solver_seconds": 60,
                }
                job = {
                    "spec": spec,
                    "trial": repeat,
                    "seed": seed,
                    "method": method,
                    "settings": settings,
                }
                execute(job, str(database), str(root / "data/upstream"), implementation)
                results.append(
                    store.get(digest({"job": job, "implementation": implementation}))
                )
        return {
            "parameters": parameters,
            "method": method,
            "successes": sum(bool(r.get("success")) for r in results),
            "trials": len(results),
            "mean_seconds": statistics.mean(r["wall_seconds"] for r in results),
            "mean_rounds": statistics.mean(r.get("rounds", 0) for r in results),
        }

    cases = [
        (12, 0.005, 0.025),
        (12, 0.005, 0.1),
        (12, 0.01, 0.05),
        (12, 0.01, 0.125),
        (12, 0.02, 0.025),
        (12, 0.02, 0.15),
        (14, 0.005, 0.025),
        (14, 0.005, 0.1),
    ]
    variants = [
        {
            "initial_factor": initial,
            "initial_offset": offset,
            "extension_factor": extension,
        }
        for initial, offset, extension in itertools.product(
            [1.5, 2.0, 2.5], [4, 6, 8, 10], [0.5, 1.0]
        )
    ]
    with threadpool_limits(limits=1):
        ranked = []
        baseline = evaluate({}, cases, 2, "selection")
        print("BASELINE", baseline, flush=True)
        for i, p in enumerate(variants):
            record = evaluate(p, cases, 2, "selection")
            ranked.append(record)
            atomic_text(out / "phase-selection.json", json.dumps(ranked, indent=2))
            print("selection", i + 1, "/", len(variants), record, flush=True)
        ranked.sort(
            key=lambda r: (-r["successes"], r["mean_seconds"], r["mean_rounds"])
        )
        winner = ranked[0]
        heldout = cases + [
            (14, 0.04, 0.125),
            (14, 0.01, 0.05),
            (16, 0.005, 0.025),
            (16, 0.005, 0.125),
        ]
        current = evaluate({}, heldout, 6, "heldout")
        validation = evaluate(winner["parameters"], heldout, 6, "heldout")
        cs = evaluate({}, heldout, 6, "heldout", "cs")
        report = {
            "screen_candidates": len(variants),
            "baseline_selection": baseline,
            "selected": winner,
            "current_validation": current,
            "validation": validation,
            "cs_validation": cs,
        }
        atomic_text(out / "phase-report.json", json.dumps(report, indent=2))
        print("FINAL", json.dumps(report), flush=True)
    store.close()
    return json.loads(report_path.read_text())


def read_results(directory):
    store = Store(directory / "cache.sqlite")
    selected = []
    for _, row in store.iter_records("result"):
        selected.append(
            {
                key: row[key]
                for key in [
                    "job",
                    "status",
                    "success",
                    "queries",
                    "wall_seconds",
                    "algorithm_seconds",
                    "implementation",
                ]
                if key in row
            }
        )
    store.close()
    return selected


def resolved_config(bundle, suite):
    config = json.loads(json.dumps(bundle["suites"][suite]))
    config.setdefault("defaults", {}).update(timing_workers=1, blas_threads=1)
    return config


def candidate_config(bundle, suite, report, fallback=False):
    bundle = json.loads(json.dumps(bundle))
    config = bundle["suites"][suite]
    if suite == "graphs":
        validation = report["validation"]
        if (
            validation["successes"] != validation["trials"]
            or validation["query_ratio"] >= 1
        ):
            raise RuntimeError(
                "Graph calibration did not preserve recovery and reduce query cost"
            )
        config["methods"]["prefix_bch"].update(report["selected"]["parameters"])
        for experiment in config["experiments"]:
            experiment.get("parameter_grids", {}).pop("prefix_bch", None)
    else:
        for experiment in config["experiments"]:
            if max(experiment["deltas"]) > 0.08:
                continue
            experiment["deltas"] = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
            experiment["rhos"] = [round(i * 0.025, 3) for i in range(1, 13)]
            better = (
                report["validation"]["successes"]
                >= report["current_validation"]["successes"] + 2
            )
            if better and not fallback:
                experiment.setdefault("method_settings", {})["prefix"] = report[
                    "selected"
                ]["parameters"]
    return bundle


def prepare(suite, report, fallback=False):
    stage = out / suite
    stage.mkdir(parents=True, exist_ok=True)
    state_path = stage / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("prepared"):
            return state
    original = json.loads((root / "config.json").read_text())
    bundle = candidate_config(original, suite, report, fallback)
    source = root / "results" / suite
    implementation = main.cache_implementation(root / "results", suite)
    old_config = resolved_config(original, suite)
    config = resolved_config(bundle, suite)
    old_jobs = list(jobs(old_config))
    expected = {
        digest({"job": job, "implementation": implementation}) for job in jobs(config)
    }
    with closing(
        sqlite3.connect(f"file:{source / 'cache.sqlite'}?mode=ro", uri=True)
    ) as db:
        completed = {
            row[0] for row in db.execute("SELECT key FROM records WHERE kind='result'")
        }
        if not all(
            digest({"job": job, "implementation": implementation}) in completed
            for job in old_jobs
        ):
            raise RuntimeError(
                f"The existing {suite} run must finish before replacement evaluation"
            )
        candidate = stage / suite
        if candidate.exists():
            shutil.rmtree(candidate)
        candidate.mkdir()
        with closing(sqlite3.connect(candidate / "cache.sqlite")) as target:
            db.backup(target)
    with closing(sqlite3.connect(candidate / "cache.sqlite")) as db:
        result_keys = [
            row[0] for row in db.execute("SELECT key FROM records WHERE kind='result'")
        ]
        obsolete = [key for key in result_keys if key not in expected]
        db.executemany("DELETE FROM records WHERE key=?", [(key,) for key in obsolete])
        db.execute("DELETE FROM records WHERE kind!='result'")
        db.execute("DELETE FROM queries")
        db.commit()
        db.execute("VACUUM")
    identity = {
        "source_sha256": implementation,
        "numerical_source_sha256": main.numerical_source_digest(),
        "config_sha256": digest(config),
        "entrypoint_sha256": hashlib.sha256(
            (root / "main.py").read_bytes()
        ).hexdigest(),
        "timing_workers": 1,
    }
    atomic_text(
        candidate / "manifest.json",
        json.dumps({"identity": identity, "run": provenance(config)}, indent=2),
    )
    atomic_text(stage / "config.json", json.dumps(bundle, indent=2) + "\n")
    state = {
        "prepared": True,
        "suite": suite,
        "implementation": implementation,
        "old_config_sha256": digest(old_config),
        "new_config_sha256": digest(config),
        "expected_trials": len(expected),
        "obsolete_trials_removed_from_candidate": len(obsolete),
        "fallback": fallback,
        "promoted": False,
    }
    atomic_text(state_path, json.dumps(state, indent=2))
    return state


def initialize_worker(cpu_queue):
    os.sched_setaffinity(0, {cpu_queue.get()})


def run_trial(job, database, implementation):
    with threadpool_limits(limits=1):
        return execute(job, database, str(root / "data/upstream"), implementation)


def evaluate_candidate(suite, state, workers):
    stage = out / suite
    bundle = json.loads((stage / "config.json").read_text())
    config = resolved_config(bundle, suite)
    directory = stage / suite
    with closing(sqlite3.connect(directory / "cache.sqlite")) as db:
        completed = {
            row[0] for row in db.execute("SELECT key FROM records WHERE kind='result'")
        }
    pending = [
        job
        for job in jobs(config)
        if digest({"job": job, "implementation": state["implementation"]})
        not in completed
    ]
    pending.sort(
        key=lambda job: (
            job["trial"],
            job["spec"].get("size", job["spec"].get("dimension", 0)),
        )
    )
    logger.info(
        "Replacement %s: %d cached, %d pending", suite, len(completed), len(pending)
    )
    started = time.perf_counter()
    if workers == 1:
        for index, job in enumerate(pending, 1):
            run_trial(job, str(directory / "cache.sqlite"), state["implementation"])
            logger.info(
                "Replacement %s progress=%d/%d elapsed=%.1fs",
                suite,
                len(completed) + index,
                state["expected_trials"],
                time.perf_counter() - started,
            )
    elif pending:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        available = sorted(os.sched_getaffinity(0))
        selected = [available[-1], available[-2]][:workers]
        for cpu in selected:
            queue.put(cpu)
        with ProcessPoolExecutor(
            max_workers=len(selected),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(queue,),
        ) as pool:
            futures = [
                pool.submit(
                    run_trial,
                    job,
                    str(directory / "cache.sqlite"),
                    state["implementation"],
                )
                for job in pending
            ]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                logger.info(
                    "Replacement %s progress=%d/%d elapsed=%.1fs",
                    suite,
                    len(completed) + index,
                    state["expected_trials"],
                    time.perf_counter() - started,
                )
    with closing(sqlite3.connect(directory / "cache.sqlite")) as db:
        actual = db.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]
    if actual != state["expected_trials"]:
        raise RuntimeError(f"{suite}: incomplete replacement cache")
    render(stage, suite, config, state["implementation"])


def replacement_quality(suite):
    stage = out / suite
    before = read_results(root / "results" / suite)
    after = read_results(stage / suite)
    if suite == "graphs":
        new = [row for row in after if row["job"]["method"] == "prefix_bch"]
        selected = graph_operating_points(aggregate(before))
        old_queries = {
            (r["kind"], r["dimension"], r["edges"], r["snapshot"]): r["queries_median"]
            for r in selected
            if r["method"] == "prefix_bch"
        }
        ratios = [
            row["queries"]
            / old_queries[
                tuple(
                    row["job"]["spec"][k]
                    for k in ["kind", "dimension", "edges", "snapshot"]
                )
            ]
            for row in new
        ]
        cs_queries = {
            (
                row["job"]["spec"]["kind"],
                row["job"]["spec"]["dimension"],
                row["job"]["spec"]["edges"],
                row["job"]["spec"]["snapshot"],
                row["job"]["trial"],
            ): row["queries"]
            for row in after
            if row["job"]["method"] == "cs"
        }
        vs_cs = [
            row["queries"]
            / cs_queries[
                tuple(
                    row["job"]["spec"][k]
                    for k in ["kind", "dimension", "edges", "snapshot"]
                )
                + (row["job"]["trial"],)
            ]
            for row in new
        ]
        old_times = {
            (r["kind"], r["dimension"], r["edges"], r["snapshot"]): r[
                "algorithm_seconds_median"
            ]
            for r in selected
            if r["method"] == "prefix_bch"
        }
        time_ratios = [
            row["algorithm_seconds"]
            / old_times[
                tuple(
                    row["job"]["spec"][k]
                    for k in ["kind", "dimension", "edges", "snapshot"]
                )
            ]
            for row in new
        ]
        query_ratio = statistics.mean(ratios)
        runtime_ratio = statistics.mean(time_ratios)
        improved = query_ratio < 0.95 or (query_ratio <= 1.10 and runtime_ratio <= 0.5)
        successes = sum(bool(row.get("success")) for row in new)
        return {
            "accepted": successes == len(new)
            and statistics.mean(vs_cs) < 1
            and improved,
            "successes": successes,
            "trials": len(new),
            "mean_query_ratio_to_previous_plot": query_ratio,
            "mean_query_ratio_to_cs": statistics.mean(vs_cs),
            "mean_runtime_ratio_to_previous_plot": runtime_ratio,
        }

    def signature(row):
        job = row["job"]
        return digest([job["spec"], job["seed"], job["trial"], job["method"]])

    old = {
        signature(r): r
        for r in before
        if r["job"]["method"] == "prefix" and r["job"]["spec"]["size"] >= 12
    }
    matched = {signature(r): r for r in after if signature(r) in old}
    old_successes = sum(bool(r.get("success")) for r in old.values())
    new_successes = sum(bool(r.get("success")) for r in matched.values())
    return {
        "accepted": len(matched) == len(old) and new_successes >= old_successes,
        "matched_trials": len(matched),
        "old_successes": old_successes,
        "new_successes": new_successes,
        "extended_grid_trials": sum(r["job"]["spec"]["size"] >= 12 for r in after),
    }


def promote(suite, state, quality):
    stage = out / suite
    figures = (
        ["graphs", "graph_recovery"]
        if suite == "graphs"
        else ["phase_diagrams", "phase_sparse"]
    )
    for name in figures:
        for suffix in ["png", "pdf"]:
            if (stage / "figures" / f"{name}.{suffix}").stat().st_size < 1000:
                raise RuntimeError("Replacement figures were not generated")
    with (out / "promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads((root / "config.json").read_text())
        bundle = json.loads((stage / "config.json").read_text())
        target = root / "results" / suite
        candidate = stage / suite
        backup = stage / "superseded"
        actual = (
            json.loads((target / "manifest.json").read_text())["identity"][
                "config_sha256"
            ]
            if target.exists()
            else None
        )
        if actual == state["old_config_sha256"] and not backup.exists():
            if digest(resolved_config(current, suite)) != state["old_config_sha256"]:
                raise RuntimeError("Live configuration changed during refinement")
            target.rename(backup)
        if not target.exists() and backup.exists() and candidate.exists():
            candidate.rename(target)
        actual = json.loads((target / "manifest.json").read_text())["identity"][
            "config_sha256"
        ]
        if actual != state["new_config_sha256"]:
            raise RuntimeError(
                "Unexpected replacement manifest; previous data preserved"
            )
        current["suites"][suite] = bundle["suites"][suite]
        atomic_text(root / "config.json", json.dumps(current, indent=2) + "\n")
        for name in figures:
            for suffix in ["png", "pdf"]:
                destination = root / "results" / "figures" / f"{name}.{suffix}"
                temporary = destination.with_suffix(destination.suffix + ".refine.tmp")
                shutil.copyfile(stage / "figures" / destination.name, temporary)
                temporary.replace(destination)
        state.update(promoted=True, quality=quality)
        atomic_text(stage / "state.json", json.dumps(state, indent=2))
        if backup.exists():
            shutil.rmtree(backup)
        shutil.rmtree(stage / "figures")
        logger.info(
            "Promoted %s replacement and removed superseded results: %s", suite, quality
        )


def run(suite, workers):
    state_path = out / suite / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state.get("promoted"):
        backup = out / suite / "superseded"
        if backup.exists():
            shutil.rmtree(backup)
        figures = out / suite / "figures"
        if figures.exists():
            shutil.rmtree(figures)
        logger.info("%s replacement already complete", suite)
        return
    if state and (out / suite / "superseded").exists():
        quality = json.loads((out / suite / "quality.json").read_text())
        if not quality["accepted"]:
            raise RuntimeError("Cannot resume an unvalidated promotion")
        promote(suite, state, quality)
        return
    report = calibrate_graphs() if suite == "graphs" else calibrate_phase()
    state = prepare(suite, report)
    logging.getLogger().setLevel(logging.INFO)
    evaluate_candidate(suite, state, 1 if suite == "graphs" else workers)
    quality = replacement_quality(suite)
    atomic_text(out / suite / "quality.json", json.dumps(quality, indent=2))
    if not quality["accepted"]:
        if suite == "phase" and not state["fallback"]:
            logger.warning(
                "Phase parameter change did not improve matched evaluation; retaining original parameters for the extended grid"
            )
            shutil.rmtree(out / suite)
            state = prepare(suite, report, fallback=True)
            evaluate_candidate(suite, state, workers)
            quality = replacement_quality(suite)
            atomic_text(out / suite / "quality.json", json.dumps(quality, indent=2))
        if not quality["accepted"]:
            raise RuntimeError(
                f"{suite} replacement failed acceptance; live results were retained"
            )
    promote(suite, state, quality)


def cli():
    parser = argparse.ArgumentParser(
        description="Calibrate SPHT parameters, evaluate complete replacements, and promote only validated improvements"
    )
    parser.add_argument("--suite", choices=["graphs", "phase", "all"], default="all")
    parser.add_argument("--phase-workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This refinement is already running") from error
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(out / f"{args.suite}-pipeline.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        for suite in ["graphs", "phase"] if args.suite == "all" else [args.suite]:
            run(suite, args.phase_workers)


if __name__ == "__main__":
    cli()
