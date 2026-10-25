import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import fcntl
import logging
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from threadpoolctl import threadpool_limits

from src import original_code
from src.config import CONFIG
from src.experiments import EVALUATED, evaluate, execute, jobs, key
from src.plotting import render
from src.store import Store, digest

ROOT = Path(__file__).resolve().parent
SUITES = ("phase", "graphs", "scaling", "robust", "hpo")
ACTIVE = ("phase", "graphs", "scaling")
CPU_INDEX = {"phase": -1, "graphs": -2}


def interpreter(suite):
    return ROOT / (".venv-bench" if suite in EVALUATED else ".venv") / "bin" / "python"


def identity(suite):
    packages = ["numpy", "scipy", "numba", "galois"] + (["spgl1"] if suite in EVALUATED else [])
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), platform.processor())
    return {
        "configuration": CONFIG["suites"][suite],
        "evaluation": CONFIG["evaluation"] if suite in EVALUATED else None,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "cpu": cpu,
    }


def open_run(directory, suite):
    store = Store(directory / "cache.sqlite")
    expected = identity(suite)
    metadata = store.get("__run__")
    if metadata is None:
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata = {"identity": expected, "implementation": digest(expected), "complete": False, "created_utc": created, "suite": suite}
    elif any(metadata["identity"].get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"Settings or environment changed for {suite}; choose a fresh --output directory")
    return store, metadata


def audit(directory, planned, metadata):
    store = Store(directory / "cache.sqlite")
    if store.keys() != {key(job, metadata["implementation"]) for job in planned}:
        raise RuntimeError("Completion audit failed: unexpected or missing results")
    statuses = Counter(row["status"] for _, row in store.iter_records("result"))
    if statuses["implementation_error"]:
        raise RuntimeError("Implementation errors must be resolved before publication")
    metadata.update(
        complete=True, stage="complete", completed_trials=len(planned),
        audit={"expected_trials": len(planned), "statuses": dict(statuses), "passed": True},
    )
    store.put("__run__", metadata, kind="metadata")
    store.close()


def run(output, suite):
    directory = output / suite
    directory.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
        handlers=[logging.FileHandler(output / "logs" / f"{suite}.log"), logging.StreamHandler()],
    )
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.info("%s is already running", suite)
            return
        cpus = sorted(os.sched_getaffinity(0))
        cpu = cpus[max(CPU_INDEX.get(suite, -4), -len(cpus))]
        os.sched_setaffinity(0, {cpu})
        with (output / f"cpu{cpu}.lock").open("a") as cpu_lock:
            logging.info("%s waiting for exclusive use of CPU %d", suite, cpu)
            fcntl.flock(cpu_lock, fcntl.LOCK_EX)
            if suite == "graphs":
                original_code.prepare()
            store, metadata = open_run(directory, suite)
            implementation = metadata["implementation"]
            planned = list(jobs(suite))
            finished = store.keys()
            pending = [job for job in planned if key(job, implementation) not in finished]
            if suite not in EVALUATED:
                pending.sort(key=lambda job: job["trial"])
            metadata.update(stage="running", expected_trials=len(planned), completed_trials=len(finished), execution={"cpu": cpu})
            store.put("__run__", metadata, kind="metadata")
            store.close()
            logging.info("%s: %d completed, %d pending on CPU %d", suite, len(finished), len(pending), cpu)
            for index, job in enumerate(pending, len(finished) + 1):
                with threadpool_limits(limits=1):
                    if suite in EVALUATED:
                        evaluate(job, directory / "cache.sqlite", implementation, CONFIG["evaluation"]["time_limit_seconds"])
                    else:
                        execute(job, directory / "cache.sqlite", implementation)
                logging.info("%s progress=%d/%d", suite, index, len(planned))
            audit(directory, planned, metadata)
            render(output, suite, implementation)
        logging.info("%s complete; final PNG and PDF figures saved", suite)


def session(suite):
    return f"sht-{suite}"


def alive(suite):
    command = ["tmux", "display-message", "-p", "-t", f"={session(suite)}:0.0", "#{pane_dead}"]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "0"


def start(output, suite):
    if alive(suite):
        print(f"{session(suite)}: already running")
        return
    subprocess.run(["tmux", "kill-session", "-t", "=" + session(suite)], capture_output=True)
    command = shlex.join([str(interpreter(suite)), str(ROOT / "main.py"), "run", "--suite", suite, "--output", str(output)])
    subprocess.run(["tmux", "new-session", "-d", "-s", session(suite), "-c", str(ROOT), command], check=True)
    subprocess.run(["tmux", "set-option", "-w", "-t", session(suite), "remain-on-exit", "on"], check=True)
    print(f"{session(suite)}: started")


def cache_state(output, suite):
    path = output / suite / "cache.sqlite"
    if not path.exists():
        return None, 0
    store = Store(path)
    found, count = store.get("__run__"), len(store.keys())
    store.close()
    return found, count


def main():
    parser = argparse.ArgumentParser(description="Run and resume sparse Hadamard experiments")
    parser.add_argument("command", choices=["run", "start", "stop", "status", "plot", "fetch"], nargs="?", default="run")
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "fetch":
        return original_code.prepare()
    suites = (ACTIVE if args.command in {"run", "start", "plot"} else SUITES) if args.suite == "all" else (args.suite,)
    for suite in suites:
        if args.command == "start":
            start(output, suite)
        elif args.command == "stop":
            subprocess.run(["tmux", "kill-session", "-t", "=" + session(suite)], capture_output=True)
        elif args.command == "status":
            expected = sum(1 for _ in jobs(suite))
            count = cache_state(output, suite)[1]
            state = "complete" if count == expected else "running" if alive(suite) else "not running"
            print(f"{suite}: {count:,}/{expected:,} ({100 * count / expected:.1f}%) {state}")
        elif Path(sys.prefix).resolve() != interpreter(suite).parent.parent.resolve():
            command = [str(interpreter(suite)), str(ROOT / "main.py"), args.command, "--suite", suite, "--output", str(output)]
            subprocess.run(command, check=True)
        elif args.command == "plot":
            found = cache_state(output, suite)[0]
            if found and found.get("complete"):
                render(output, suite, found["implementation"])
            else:
                print(f"{suite}: incomplete; no figure generated")
        else:
            run(output, suite)


if __name__ == "__main__":
    main()
