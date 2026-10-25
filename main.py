import os

for variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"


import argparse
import copy
import fcntl
import logging
import platform
import shlex
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from contextlib import closing, nullcontext
from importlib.metadata import version
from pathlib import Path

from threadpoolctl import threadpool_limits

from src import data as originals
from src.config import CONFIG
from src.data import Store, digest, source_digest, timing_slot
from src.experiments import evaluate, execute, jobs
from src.plotting import render

ROOT = Path(__file__).resolve().parent
SUITES = ("phase", "graphs", "scaling", "robust", "hpo")
ACTIVE = ("phase", "graphs", "scaling")
BENCHMARK_PYTHON = ROOT / ".venv-bench/bin/python"
PHASE_PYTHON = ROOT / ".venv/bin/python"
logger = logging.getLogger(__name__)


def configuration(suite):
    config = copy.deepcopy(CONFIG["suites"][suite])
    config.setdefault("defaults", {}).update(timing_workers=1, blas_threads=1)
    return config


def planned_jobs(suite):
    config = configuration(suite)
    for job in jobs(config):
        if suite in {"graphs", "scaling"}:
            policy = CONFIG["evaluation"]
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
                    max_iterations=policy["max_iterations"],
                    tolerance=policy["tolerance"],
                )
            if job["method"] in {"stobbe", "swht"}:
                published = originals.published_setting(job["spec"], job["method"])
                settings.update(published["settings"])
                job["published_configuration"] = published
            job["role"] = "evaluation"
        yield job


def identity(suite):
    packages = ["numpy", "scipy", "numba", "galois"]
    if suite in {"graphs", "scaling"}:
        packages.append("spgl1")
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return {
        "configuration": configuration(suite),
        "evaluation": CONFIG["evaluation"] if suite in {"graphs", "scaling"} else None,
        "numerical_source": source_digest(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "cpu": model,
    }


def completed(output, suite):
    path = Path(output) / suite / "cache.sqlite"
    if not path.exists():
        return 0
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return db.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]


def initialize(output, suite):
    directory = Path(output) / suite
    directory.mkdir(parents=True, exist_ok=True)
    store = Store(directory / "cache.sqlite")
    expected = identity(suite)
    metadata = store.get("__run__")
    if metadata is None:
        if store.db.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]:
            raise RuntimeError(
                "Unidentified results require an explicit cache migration"
            )
        metadata = {
            "identity": expected,
            "implementation": digest(expected),
            "complete": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suite": suite,
        }
        store.put("__run__", metadata, kind="metadata")
    elif metadata["identity"] != expected:
        raise RuntimeError(
            f"Numerical inputs changed for {suite}; choose a fresh output directory"
        )
    store.close()
    return metadata


def audit(output, suite, planned, metadata):
    store = Store(Path(output) / suite / "cache.sqlite")
    expected = {
        digest({"job": job, "implementation": metadata["implementation"]})
        for job in planned
    }
    actual = {
        key
        for (key,) in store.db.execute("SELECT key FROM records WHERE kind='result'")
    }
    if actual != expected:
        store.close()
        raise RuntimeError("Completion audit failed: unexpected or missing results")
    counts = Counter()
    statuses = Counter()
    for _, row in store.iter_records("result"):
        counts[digest([row["job"]["spec"], row["job"]["method"]])] += 1
        statuses[row["status"]] += 1
    if statuses.get("implementation_error"):
        raise RuntimeError("Implementation errors must be resolved before publication")
    expected_counts = Counter(digest([job["spec"], job["method"]]) for job in planned)
    if counts != expected_counts:
        raise RuntimeError("Completion audit failed: incorrect repetition counts")
    metadata.update(
        complete=True,
        stage="complete",
        completed_trials=len(actual),
        audit={
            "expected_trials": len(expected),
            "configuration_count": len(counts),
            "statuses": dict(statuses),
            "passed": True,
        },
    )
    store.put("__run__", metadata, kind="metadata")
    store.close()


def run(output, suite):
    output = Path(output)
    directory = output / suite
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("%s is already running", suite)
            return
        logs = output / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(logs / f"{suite}.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        cpus = sorted(os.sched_getaffinity(0))
        cpu = cpus[-1] if suite == "phase" else cpus[-min(4, len(cpus))]
        os.sched_setaffinity(0, {cpu})
        logger.info(
            "%s assigned CPU %d; timing queue %s",
            suite,
            cpu,
            "bypassed" if suite == "phase" else "shared",
        )
        if suite == "graphs":
            originals.prepare()
        metadata = initialize(output, suite)
        implementation = metadata["implementation"]
        planned = list(planned_jobs(suite))
        store = Store(directory / "cache.sqlite")
        finished = {
            key
            for (key,) in store.db.execute(
                "SELECT key FROM records WHERE kind='result'"
            )
        }
        pending = [
            job
            for job in planned
            if digest({"job": job, "implementation": implementation}) not in finished
        ]
        if suite not in {"graphs", "scaling"}:
            pending.sort(key=lambda job: job["trial"])
        metadata.update(
            stage="running",
            expected_trials=len(planned),
            completed_trials=len(finished),
            execution={
                "cpu": cpu,
                "timing_queue": "independent" if suite == "phase" else "shared",
            },
        )
        store.put("__run__", metadata, kind="metadata")
        store.close()
        logger.info("%s: %d completed, %d pending", suite, len(finished), len(pending))
        for index, job in enumerate(pending, len(finished) + 1):
            with (
                nullcontext() if suite == "phase" else timing_slot(output / ".runtime"),
                threadpool_limits(limits=1),
            ):
                if suite in {"graphs", "scaling"}:
                    evaluate(
                        job,
                        directory / "cache.sqlite",
                        implementation,
                        CONFIG["evaluation"]["time_limit_seconds"],
                    )
                else:
                    execute(
                        job,
                        directory / "cache.sqlite",
                        ROOT / "data/originals",
                        implementation,
                    )
            logger.info("%s progress=%d/%d", suite, index, len(planned))
        audit(output, suite, planned, metadata)
        with nullcontext() if suite == "phase" else timing_slot(output / ".runtime"):
            render(output, suite, configuration(suite), implementation)
        logger.info("%s complete; final PNG and PDF figures saved", suite)


def session(suite):
    return f"sht-{suite}"


def alive(suite):
    result = subprocess.run(
        [
            "tmux",
            "display-message",
            "-p",
            "-t",
            f"={session(suite)}:0.0",
            "#{pane_dead}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "0"


def start(output, suites):
    for suite in suites:
        if alive(suite):
            print(f"{session(suite)}: already running")
            continue
        subprocess.run(
            ["tmux", "kill-session", "-t", "=" + session(suite)],
            capture_output=True,
            check=False,
        )
        python = BENCHMARK_PYTHON if suite in {"graphs", "scaling"} else PHASE_PYTHON
        command = shlex.join(
            [
                str(python),
                str(ROOT / "main.py"),
                "run",
                "--suite",
                suite,
                "--output",
                str(output),
            ]
        )
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session(suite),
                "-c",
                str(ROOT),
                command,
            ],
            check=True,
        )
        subprocess.run(
            ["tmux", "set-option", "-w", "-t", session(suite), "remain-on-exit", "on"],
            check=True,
        )
        print(f"{session(suite)}: started")


def status(output, suites):
    for suite in suites:
        expected = sum(1 for _ in jobs(configuration(suite)))
        count = completed(output, suite)
        state = (
            "complete"
            if count == expected
            else "running"
            if alive(suite)
            else "not running"
        )
        print(
            f"{suite}: {count:,}/{expected:,} ({100 * count / max(1, expected):.1f}%) {state}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run and resume sparse Hadamard experiments"
    )
    parser.add_argument(
        "command",
        choices=["run", "start", "stop", "status", "plot", "plan", "fetch"],
        nargs="?",
        default="run",
    )
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    suites = ACTIVE if args.suite == "all" else (args.suite,)
    if args.command == "start":
        start(output, suites)
    elif args.command == "stop":
        for suite in SUITES if args.suite == "all" else suites:
            subprocess.run(
                ["tmux", "kill-session", "-t", "=" + session(suite)],
                capture_output=True,
                check=False,
            )
    elif args.command == "status":
        status(output, SUITES if args.suite == "all" else suites)
    elif args.command == "plan":
        for suite in suites:
            print(f"{suite}: {sum(1 for _ in jobs(configuration(suite))):,} trials")
    elif args.command == "fetch":
        originals.prepare()
    elif args.command == "plot":
        for suite in suites:
            path = output / suite / "cache.sqlite"
            if not path.exists():
                print(f"{suite}: no completed dataset")
                continue
            store = Store(path)
            metadata = store.get("__run__") or {}
            store.close()
            if not metadata.get("complete"):
                print(f"{suite}: incomplete; no partial figure generated")
                continue
            with (
                nullcontext() if suite == "phase" else timing_slot(output / ".runtime")
            ):
                render(output, suite, configuration(suite), metadata["implementation"])
    else:
        for suite in suites:
            python = (
                BENCHMARK_PYTHON if suite in {"graphs", "scaling"} else PHASE_PYTHON
            )
            if Path(sys.prefix).resolve() != python.parent.parent.resolve():
                subprocess.run(
                    [
                        str(python),
                        str(ROOT / "main.py"),
                        "run",
                        "--suite",
                        suite,
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
            else:
                run(output, suite)


if __name__ == "__main__":
    main()
