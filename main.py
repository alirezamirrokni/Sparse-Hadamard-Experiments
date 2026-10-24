import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

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

from threadpoolctl import threadpool_limits

from src.algebra import warmup
from src.experiments import execute, jobs
from src.plotting import render
from src.storage import Store, atomic_text, digest, provenance
from src.upstream import fetch_sources


ROOT = Path(__file__).resolve().parent
SUITES = ("phase", "graphs", "scaling", "robust", "hpo")


def numerical_source_digest():
    value = hashlib.sha256()
    files = [file for file in (ROOT / "src").glob("*.py") if file.name != "plotting.py"]
    files += list((ROOT / "src").glob("upstream_manifest.json"))
    for file in sorted(files):
        value.update(file.name.encode())
        value.update(file.read_bytes())
    return value.hexdigest()


def cache_implementation(output, suite):
    manifest = output / suite / "manifest.json"
    if manifest.exists():
        identity = json.loads(manifest.read_text())["identity"]
        if identity.get("numerical_source_sha256") != numerical_source_digest():
            raise RuntimeError(
                f"Numerical inputs changed for {suite}; choose a fresh --output directory"
            )
        return identity["source_sha256"]
    return numerical_source_digest()


def configuration(path, suite):
    bundle = json.loads(path.read_text())
    config = bundle["suites"][suite]
    config.setdefault("defaults", {}).update(timing_workers=1, blas_threads=1)
    return config, bundle["runtime"]


def setup_logging(output, name):
    directory = output / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(directory / f"{name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def count_completed(output, suite):
    database = output / suite / "cache.sqlite"
    if not database.exists():
        return 0
    with sqlite3.connect(
        f"file:{database}?mode=ro", uri=True, timeout=120
    ) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]


def run_suite(args, suite, limit=None):
    config, runtime = configuration(args.config, suite)
    directory = args.output / suite
    directory.mkdir(parents=True, exist_ok=True)
    implementation = cache_implementation(args.output, suite)
    identity = {
        "source_sha256": implementation,
        "numerical_source_sha256": numerical_source_digest(),
        "config_sha256": digest(config),
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_workers": 1,
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["identity"] != identity:
            raise RuntimeError(
                f"Inputs changed for {suite}; choose a fresh --output directory"
            )
    else:
        if (directory / "cache.sqlite").exists():
            raise RuntimeError(f"An unidentified cache already exists for {suite}")
        atomic_text(
            manifest_path,
            json.dumps({"identity": identity, "run": provenance(config)}, indent=2),
        )
    store = Store(directory / "cache.sqlite")
    finished = {
        row[0]
        for row in store.db.execute("SELECT key FROM records WHERE kind='result'")
    }
    store.close()
    expanded = list(jobs(config))
    if args.methods:
        expanded = [job for job in expanded if job["method"] in args.methods]
    pending = [
        job
        for job in expanded
        if digest({"job": job, "implementation": implementation}) not in finished
    ]
    families = {
        kind: index
        for index, kind in enumerate(
            dict.fromkeys(job["spec"]["kind"] for job in expanded)
        )
    }
    pending.sort(key=lambda job: (families[job["spec"]["kind"]], job["trial"]))
    done = len(expanded) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    logging.info(
        "suite=%s planned=%d completed=%d batch=%d",
        suite,
        len(expanded),
        done,
        len(pending),
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        warmup()
        for index, job in enumerate(pending, 1):
            execute(
                job,
                str(directory / "cache.sqlite"),
                str(ROOT / "data/upstream"),
                implementation,
            )
            if args.methods and index % min(runtime["batch_size"][suite], 20) == 0:
                render(args.output, suite, config, implementation)
            elapsed = time.perf_counter() - started
            logging.info(
                "suite=%s progress=%d/%d batch=%d/%d elapsed=%.1fs estimated_remaining=%.1fs",
                suite,
                done + index,
                len(expanded),
                index,
                len(pending),
                elapsed,
                elapsed * (len(expanded) - done - index) / index,
            )
    render(args.output, suite, config, implementation)
    return done + len(pending), len(expanded)


def worker(args):
    suite = args.suite
    if suite == "all":
        raise ValueError("worker requires one --suite")
    setup_logging(args.output, suite)
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        width = min(len(cpus), len(SUITES))
        os.sched_setaffinity(0, {cpus[-width + SUITES.index(suite) % width]})
    os.nice(5)
    config, runtime = configuration(args.config, suite)
    total = sum(1 for _ in jobs(config))
    batch = runtime["batch_size"][suite]
    if count_completed(args.output, suite) >= total:
        render(args.output, suite, config, cache_implementation(args.output, suite))
        logging.info("suite=%s complete", suite)
        return
    while count_completed(args.output, suite) < total:
        if shutil.disk_usage(args.output.parent).free < 5 * 1024**3:
            raise RuntimeError("Fewer than 5 GiB remain; free space and resume")
        done, total = run_suite(args, suite, batch)
        logging.info("suite=%s checkpointed=%d/%d", suite, done, total)
    logging.info("suite=%s complete", suite)


def session_name(suite):
    return f"sht-{suite}"


def start(args):
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required for persistent runs")
    fetch_sources(ROOT / "data/upstream")
    selected = SUITES if args.suite == "all" else (args.suite,)
    for suite in selected:
        name = session_name(suite)
        existing = subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"], capture_output=True, check=False
        )
        if existing.returncode == 0:
            dead = subprocess.check_output(
                ["tmux", "display-message", "-p", "-t", f"{name}:0.0", "#{pane_dead}"],
                text=True,
            ).strip()
            if dead != "1":
                print(f"{name}: already running")
                continue
            subprocess.run(["tmux", "kill-session", "-t", f"={name}"], check=True)
        command = shlex.join(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "worker",
                "--suite",
                suite,
                "--config",
                str(args.config),
                "--output",
                str(args.output),
            ]
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", name, "-c", str(ROOT), command],
            check=True,
        )
        subprocess.run(
            ["tmux", "set-option", "-w", "-t", f"{name}:0", "remain-on-exit", "on"],
            check=True,
        )
        print(f"{name}: started")


def suite_sessions(suite):
    if not shutil.which("tmux"):
        return []
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    base = session_name(suite)
    return [
        name
        for name in result.stdout.splitlines()
        if name == base or name.startswith(base + "-")
    ]


def status(args):
    selected = SUITES if args.suite == "all" else (args.suite,)
    for suite in selected:
        config, runtime = configuration(args.config, suite)
        total = sum(1 for _ in jobs(config))
        done = count_completed(args.output, suite)
        state = "complete" if done >= total else "not running"
        for name in suite_sessions(suite):
            pane = subprocess.run(
                ["tmux", "display-message", "-p", "-t", f"{name}:0.0", "#{pane_dead}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if pane.returncode == 0 and pane.stdout.strip() == "0":
                state = "running" if done < total else "finishing plots"
        print(
            f"{suite}: {done:,}/{total:,} ({100 * done / max(total, 1):.1f}%) {state}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable sparse Hadamard comparisons and generate eight consolidated figures."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "start", "status", "stop", "plot", "plan", "fetch", "worker"],
    )
    parser.add_argument("--suite", default="all", choices=["all", *SUITES])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    if args.methods and (args.command != "run" or args.suite == "all"):
        parser.error("--methods requires run and one --suite")
    if args.methods:
        available = configuration(args.config, args.suite)[0]["methods"]
        if not set(args.methods) <= set(available):
            parser.error("Unknown method in --methods")
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    selected = SUITES if args.suite == "all" else (args.suite,)
    if args.command == "start":
        start(args)
    elif args.command == "status":
        status(args)
    elif args.command == "stop":
        for suite in selected:
            for name in suite_sessions(suite):
                subprocess.run(
                    ["tmux", "kill-session", "-t", f"={name}"],
                    capture_output=True,
                    check=False,
                )
    elif args.command == "fetch":
        fetch_sources(ROOT / "data/upstream")
    elif args.command == "plan":
        counts = {
            suite: sum(1 for _ in jobs(configuration(args.config, suite)[0]))
            for suite in selected
        }
        print(
            json.dumps(
                {"trials": counts, "total": sum(counts.values()), "maximum_figures": 8},
                indent=2,
            )
        )
    elif args.command == "worker":
        worker(args)
    elif args.command == "plot":
        for suite in selected:
            render(
                args.output,
                suite,
                configuration(args.config, suite)[0],
                cache_implementation(args.output, suite),
            )
    else:
        if args.limit is not None and (args.suite == "all" or args.limit < 1):
            parser.error("--limit requires one --suite and a positive number")
        setup_logging(
            args.output,
            args.suite + "-" + "-".join(args.methods) if args.methods else "main",
        )
        if "graphs" in selected:
            fetch_sources(ROOT / "data/upstream")
        for suite in selected:
            run_suite(args, suite, args.limit)


if __name__ == "__main__":
    main()
