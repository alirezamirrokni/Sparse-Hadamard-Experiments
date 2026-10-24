import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.experiments import jobs as legacy_jobs
from src.storage import atomic_text, digest

from . import original
from .scheduling import timing_slot
from .trials import evaluate

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".paper-venv" / "bin" / "python"
logger = logging.getLogger(__name__)


def invoke(args, suite, action="run"):
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        command = [
            str(PYTHON),
            str(ROOT / "main.py"),
            "paper",
            "--suite",
            suite,
            "--paper-stage",
            action,
            "--config",
            str(args.config),
            "--output",
            str(args.output),
        ]
        subprocess.run(command, check=True)
        return
    if action == "run":
        run(args, suite)
    elif action == "prepare":
        original.prepare()
    elif action == "status":
        print_status(args, suite)
    elif action == "plot":
        from .plots import render

        state = initialize(args, suite)
        render(state["workspace"], suite, state["bundle"], state["implementation"])


def configuration(args, suite):
    bundle = json.loads(Path(args.config).read_text())
    config = copy.deepcopy(bundle["suites"][suite])
    config.setdefault("defaults", {}).update(timing_workers=1, blas_threads=1)
    return bundle, config


def target(args, suite):
    _, config = configuration(args, suite)
    return sum(1 for _ in legacy_jobs(config))


def location(output, suite):
    workspace = Path(output) / "paper" / suite
    if (workspace / "cache.sqlite").exists():
        return workspace
    live = Path(output) / suite
    manifest = live / "manifest.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text()).get("protocol") == "paper-v1"
    ):
        return live
    return workspace


def completed(output, suite):
    path = location(output, suite) / "cache.sqlite"
    if not path.exists():
        return 0
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as database:
        return database.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]


def print_status(args, suite):
    total = target(args, suite)
    count = completed(args.output, suite)
    path = Path(args.output) / "paper" / f"{suite}-state.json"
    stage = (
        json.loads(path.read_text()).get("stage", "not started")
        if path.exists()
        else "not started"
    )
    print(
        f"{suite}: {count:,}/{total:,} paper trials ({100 * count / max(1, total):.1f}%) {stage}"
    )


def initialize(args, suite):
    bundle, config = configuration(args, suite)
    protocol = bundle["paper"]
    files = list((ROOT / "src").glob("*.py")) + list((ROOT / "src/paper").glob("*.py"))
    files = [path for path in files if path.name not in {"plotting.py", "plots.py"}]
    files += [
        ROOT / "src/paper/original_manifest.json",
        ROOT / "requirements-paper.lock",
    ]
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    environment = {
        package: version(package)
        for package in [
            "numpy",
            "scipy",
            "numba",
            "galois",
            "matplotlib",
            "seaborn",
            "scikit-learn",
        ]
    }
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    identity = {
        "suite": config,
        "protocol": protocol,
        "sources": sources,
        "python": platform.python_version(),
        "packages": environment,
        "cpu_model": model,
    }
    implementation = digest(identity)
    workspace = Path(args.output) / "paper" / suite
    live = location(args.output, suite)
    if live != workspace:
        meta = json.loads((live / "manifest.json").read_text())
        if meta.get("implementation") == implementation:
            state_path = Path(args.output) / "paper" / f"{suite}-state.json"
            stage = (
                json.loads(state_path.read_text()).get("stage")
                if state_path.exists()
                else None
            )
            recovery = (
                Path(args.output) / "paper" / f"{suite}-superseded"
            ).exists() or stage == "promoting"
            return {
                "workspace": live,
                "bundle": bundle,
                "config": config,
                "implementation": implementation,
                "promoted": not recovery,
                "recover_promotion": recovery,
                "output": Path(args.output),
                "suite": suite,
            }
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = workspace / "manifest.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text())["implementation"] != implementation
    ):
        raise RuntimeError(
            f"Paper protocol inputs changed for {suite}; use a fresh output directory"
        )
    if not manifest.exists():
        atomic_text(
            manifest,
            json.dumps(
                {
                    "protocol": "paper-v1",
                    "implementation": implementation,
                    "identity": identity,
                    "created_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "timing": "single-thread process CPU time excluding oracle wrapper and checkpoint I/O; offline BCH maps excluded; wall times also retained",
                    "original_sources": json.loads(original.MANIFEST.read_text()),
                },
                indent=2,
            ),
        )
    return {
        "workspace": workspace,
        "bundle": bundle,
        "config": config,
        "implementation": implementation,
        "promoted": False,
        "output": Path(args.output),
        "suite": suite,
    }


def settings(state, method, factor=None):
    config = state["config"]
    policy = state["bundle"]["paper"]
    result = {**config.get("defaults", {}), **config["methods"][method]}
    result.update(
        max_iterations=policy["max_iterations"],
        tolerance=policy["tolerance"],
        max_dense_bytes=policy["max_dense_bytes"],
        max_global_columns=policy["max_global_columns"],
        max_local_columns=policy["max_local_columns"],
        solver_seconds=policy["time_limit_seconds"],
    )
    if method == "cs" or method.startswith("prefix"):
        result.update(
            {
                name: policy[name]
                for name in [
                    "support_iterations",
                    "certificate_rounds",
                    "certificate_seconds",
                ]
            }
        )
    if method == "cs":
        result["threshold"] = state.get(
            "cs_threshold", state["bundle"]["paper"]["cs_thresholds"][0]
        )
        result.update(
            solver="bp", sample_factor=float(factor if factor is not None else 2.4)
        )
    return result


def trial(state, job, calibration=False):
    directory = state["workspace"]
    policy = state["bundle"]["paper"]
    filename = "calibration.sqlite" if calibration else "cache.sqlite"
    with timing_slot(state["output"] / "paper"), threadpool_limits(limits=1):
        limit = policy["time_limit_seconds"]
        if calibration:
            limit = min(limit, policy["calibration_trial_seconds"])
        return evaluate(
            job,
            str(directory / filename),
            state["implementation"],
            limit,
        )


def calibration_cases(suite):
    if suite == "graphs":
        return [
            {
                "kind": "graph_edges" if n == 100 else "graph_vertices",
                "dimension": n,
                "edges": e,
                "snapshot": s,
            }
            for n, e, s in [
                (100, 20, 1),
                (100, 50, 3),
                (100, 70, 7),
                (50, 50, 1),
                (210, 50, 3),
                (800, 50, 7),
            ]
        ]
    return [
        {"kind": "scaling", "dimension": 12, "sparsity": 8},
        {"kind": "scaling", "dimension": 18, "sparsity": 16},
        {"kind": "scaling", "dimension": 20, "sparsity": 4},
        {"kind": "scaling", "dimension": 100, "sparsity": 16, "degree": 2},
        {"kind": "scaling", "dimension": 400, "sparsity": 16, "degree": 2},
        {"kind": "scaling", "dimension": 60, "sparsity": 16, "degree": 3},
        {"kind": "scaling", "dimension": 100, "sparsity": 16, "degree": 3},
    ]


def calibrate(state):
    path = state["workspace"] / "calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    policy = state["bundle"]["paper"]
    suite = state["suite"]
    repetitions = policy["calibration_repetitions"][suite]
    required = math.ceil(repetitions * policy["target_recovery"])
    candidates = []
    spent_seconds = 0.0
    exhausted = False
    for factor in policy["cs_sample_factors"]:
        for threshold in policy["cs_thresholds"]:
            successes = [0] * len(calibration_cases(suite))
            attempts = [0] * len(successes)
            times = []
            rejected = False
            for index, spec in enumerate(calibration_cases(suite)):
                for repeat in range(repetitions):
                    if spent_seconds >= policy["calibration_budget_seconds"]:
                        exhausted = True
                        rejected = True
                        break
                    seed = int(
                        digest(
                            [
                                policy["seed"],
                                "paper-calibration-v1",
                                suite,
                                spec,
                                repeat,
                            ]
                        )[:8],
                        16,
                    )
                    parameters = settings(state, "cs", factor)
                    parameters["threshold"] = threshold
                    job = {
                        "spec": spec,
                        "method": "cs",
                        "settings": parameters,
                        "seed": seed,
                        "trial": repeat,
                        "role": "calibration",
                    }
                    result = trial(state, job, calibration=True)
                    spent_seconds += result["wall_seconds"]
                    attempts[index] += 1
                    successes[index] += bool(result["success"])
                    times.append(result["algorithm_seconds"])
                    if attempts[index] - successes[index] > repetitions - required:
                        rejected = True
                        break
                if rejected:
                    break
            item = {
                "factor": factor,
                "threshold": threshold,
                "successes": successes,
                "attempts": attempts,
                "mean_seconds": sum(times) / len(times) if times else None,
                "target_met": not rejected and min(successes) >= required,
            }
            candidates.append(item)
            atomic_text(
                state["workspace"] / "calibration-progress.json",
                json.dumps(candidates, indent=2),
            )
            logger.info(
                "%s CS calibration factor=%s threshold=%s successes=%s attempts=%s eligible=%s",
                suite,
                factor,
                threshold,
                successes,
                attempts,
                item["target_met"],
            )
            if exhausted:
                break
        if exhausted:
            break
        if any(item["target_met"] and item["factor"] == factor for item in candidates):
            break
    eligible = [item for item in candidates if item["target_met"]]
    selected = (
        min(eligible, key=lambda item: (item["factor"], item["mean_seconds"]))
        if eligible
        else {
            "factor": policy["cs_sample_factors"][-1],
            "threshold": policy["cs_thresholds"][0],
            "target_met": False,
        }
    )
    report = {
        "factor": selected["factor"],
        "threshold": selected["threshold"],
        "target_met": selected["target_met"],
        "target": policy["target_recovery"],
        "repetitions_per_case": repetitions,
        "cases": calibration_cases(suite),
        "candidates": candidates,
        "budget_exhausted": exhausted,
        "screening_compute_seconds": spent_seconds,
        "maximum_screening_trials": len(policy["cs_sample_factors"])
        * len(policy["cs_thresholds"])
        * len(calibration_cases(suite))
        * repetitions,
        "seed_role": "paper-calibration-v1",
        "selection": "bounded independent pilot: minimum declared factor passing every screening case; early rejection, per-trial timeout and total compute budget; largest declared factor if none passes; empirical screening is not a recovery guarantee",
    }
    atomic_text(path, json.dumps(report, indent=2))
    return report


def evaluation_jobs(state, factor):
    for job in legacy_jobs(state["config"]):
        method = job["method"]
        inherited = job["settings"]
        baseline = {
            **state["config"].get("defaults", {}),
            **state["config"]["methods"][method],
        }
        overrides = {
            name: value
            for name, value in inherited.items()
            if name not in baseline or value != baseline[name]
        }
        job["settings"] = {**settings(state, method, factor), **overrides}
        job["role"] = "evaluation"
        if method in {"swht", "stobbe"}:
            published = original.published_setting(job["spec"], method)
            job["settings"].update(published["settings"])
            job["published_configuration"] = published
        yield job


def save_state(state, stage, **extra):
    atomic_text(
        state["output"] / "paper" / f"{state['suite']}-state.json",
        json.dumps(
            {"stage": stage, "implementation": state["implementation"], **extra},
            indent=2,
        ),
    )


def run(args, suite):
    output = Path(args.output)
    directory = output / "paper"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{suite}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("%s paper run already active", suite)
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(directory / f"{suite}.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[-min(4, len(available))]})
        state = initialize(args, suite)
        if state["promoted"]:
            print_status(args, suite)
            return
        if state.get("recover_promotion"):
            promote(state)
            save_state(state, "complete", expected_trials=target(args, suite))
            return
        if suite == "graphs":
            original.prepare()
        save_state(state, "calibrating")
        calibration = calibrate(state)
        factor = calibration["factor"]
        state["cs_threshold"] = calibration["threshold"]
        all_jobs = list(evaluation_jobs(state, factor))
        rows = [
            {
                "kind": job["spec"]["kind"],
                "snapshot": job["spec"].get("snapshot"),
                "dimension": job["spec"]["dimension"],
                "edges": job["spec"].get("edges"),
                "method": job["method"],
                **job.get("published_configuration", {}).get("settings", {}),
            }
            for job in all_jobs
            if job["trial"] == 0 and job["method"] in {"swht", "stobbe"}
        ]
        atomic_text(
            state["workspace"] / "published-configurations.json",
            json.dumps(rows, indent=2),
        )
        save_state(state, "evaluating", expected_trials=len(all_jobs), cs_factor=factor)
        database = state["workspace"] / "cache.sqlite"
        if database.exists():
            with closing(sqlite3.connect(database)) as connection:
                finished = {
                    key
                    for (key,) in connection.execute(
                        "SELECT key FROM records WHERE kind='result'"
                    )
                }
        else:
            finished = set()
        pending = [
            job
            for job in all_jobs
            if digest({"job": job, "implementation": state["implementation"]})
            not in finished
        ]
        for index, job in enumerate(pending, len(finished) + 1):
            trial(state, job)
            logger.info("%s paper progress=%d/%d", suite, index, len(all_jobs))
            if index % state["bundle"]["paper"]["plot_every"][suite] == 0:
                from .plots import render

                render(
                    state["workspace"], suite, state["bundle"], state["implementation"]
                )
        from .plots import audit, render

        audit(state["workspace"], suite, all_jobs, state["implementation"])
        render(state["workspace"], suite, state["bundle"], state["implementation"])
        promote(state)
        save_state(state, "complete", expected_trials=len(all_jobs), cs_factor=factor)


def promote(state):
    suite = state["suite"]
    workspace = state["workspace"]
    output = state["output"]
    names = (
        ["graphs", "graph_recovery"]
        if suite == "graphs"
        else ["scaling_runtime", "scaling_queries"]
    )
    for name in names:
        for suffix in ["png", "pdf"]:
            if (workspace / "figures" / f"{name}.{suffix}").stat().st_size < 1000:
                raise RuntimeError("Missing final paper figure")
    (output / "figures").mkdir(parents=True, exist_ok=True)
    with (output / "paper" / "promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = output / suite
        backup = output / "paper" / f"{suite}-superseded"
        save_state(state, "promoting")
        meta = (
            json.loads((target / "manifest.json").read_text())
            if target.exists()
            else {}
        )
        already_moved = meta.get("implementation") == state["implementation"]
        if not already_moved:
            if target.exists():
                if backup.exists():
                    raise RuntimeError(
                        "Conflicting promotion data; previous results preserved"
                    )
                target.rename(backup)
            workspace.rename(target)
        workspace = target
        for name in names:
            for suffix in ["png", "pdf"]:
                destination = output / "figures" / f"{name}.{suffix}"
                temporary = destination.with_suffix(destination.suffix + ".paper.tmp")
                shutil.copyfile(target / "figures" / destination.name, temporary)
                temporary.replace(destination)
        if backup.exists():
            shutil.rmtree(backup)
        logger.info("Completed and published %s paper figures", suite)
