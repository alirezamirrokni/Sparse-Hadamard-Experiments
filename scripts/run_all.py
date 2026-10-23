import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
SESSIONS = ROOT / "outputs/sessions"
LANES = {
    "phase": ["phase", "phase_sparse"],
    "graphs": ["graphs"],
    "hpo": ["hpo"],
    "other": ["figure3", "robust", "scaling"],
}
BATCHES = {"hpo": 8, "robust": 100, "scaling": 20}


def timestamp():
    return datetime.datetime.now(datetime.UTC).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(suite):
    value = hashlib.sha256()
    paths = [ROOT / "configs" / f"{suite}.json", ROOT / "requirements.lock"]
    paths.extend(sorted((ROOT / "shtbench").glob("*.py")))
    paths.append(ROOT / "shtbench/upstream_manifest.json")
    for path in paths:
        value.update(str(path.relative_to(ROOT)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def command(*arguments):
    return [str(PYTHON), "-m", "shtbench.cli", *map(str, arguments)]


def logged(arguments, log):
    print(timestamp(), shlex.join(arguments), flush=True)
    with subprocess.Popen(
        arguments,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Command exited with status {process.returncode}")


def completed(directory):
    path = directory / "cache.sqlite"
    if not path.exists():
        return 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=120) as database:
        return database.execute(
            "SELECT COUNT(*) FROM records WHERE kind='result'"
        ).fetchone()[0]


def worker(lane):
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        cpu = cpus[
            -min(len(cpus), len(LANES))
            + list(LANES).index(lane) % min(len(cpus), len(LANES))
        ]
        os.sched_setaffinity(0, {cpu})
    os.nice(5)
    state = {"lane": lane, "started_at": timestamp(), "status": "running"}
    state_path = SESSIONS / f"{lane}.json"
    save(state_path, state)
    try:
        with (SESSIONS / f"{lane}.log").open("a", buffering=1) as log:
            for suite in LANES[lane]:
                directory = ROOT / "outputs" / suite
                directory.mkdir(parents=True, exist_ok=True)
                config = ROOT / "configs" / f"{suite}.json"
                identity = {"fingerprint": fingerprint(suite), "workers": 1}
                identity_path = directory / "launch.json"
                if identity_path.exists():
                    if json.loads(identity_path.read_text()) != identity:
                        raise RuntimeError(
                            f"Inputs changed for {suite}; use a new output directory"
                        )
                else:
                    if (directory / "cache.sqlite").exists():
                        raise RuntimeError(f"Unidentified existing cache for {suite}")
                    save(identity_path, identity)
                plan = json.loads(
                    subprocess.check_output(
                        command("plan", config), cwd=ROOT, text=True
                    )
                )
                total = plan["total_trials"]
                while True:
                    done = completed(directory)
                    state.update(
                        suite=suite, completed=done, total=total, updated_at=timestamp()
                    )
                    save(state_path, state)
                    print(timestamp(), f"{suite}: {done}/{total} completed", flush=True)
                    if fingerprint(suite) != identity["fingerprint"]:
                        raise RuntimeError(f"Inputs changed while running {suite}")
                    if done >= total:
                        logged(command("plot", directory), log)
                        break
                    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                        raise RuntimeError(
                            "Fewer than 5 GiB remain; free space and restart this session"
                        )
                    logged(
                        command(
                            "run",
                            config,
                            "--output",
                            directory,
                            "--workers",
                            1,
                            "--limit",
                            BATCHES.get(suite, 1000),
                            "--no-plots",
                        ),
                        log,
                    )
                    after = completed(directory)
                    if after <= done:
                        raise RuntimeError(f"No new completed results for {suite}")
                    logged(command("plot", directory), log)
        state.update(status="complete", finished_at=timestamp())
        save(state_path, state)
    except BaseException as error:
        state.update(status="stopped", error=str(error), updated_at=timestamp())
        save(state_path, state)
        raise


def start():
    if not PYTHON.exists():
        raise RuntimeError("Install the project into .venv before starting")
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required")
    SESSIONS.mkdir(parents=True, exist_ok=True)
    for lane, suites in LANES.items():
        name = f"sht-{lane}"
        if (
            subprocess.run(
                ["tmux", "has-session", "-t", f"={name}"],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            dead = subprocess.check_output(
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    f"={name}",
                    "#{pane_dead}",
                ],
                text=True,
            ).strip()
            if dead != "1":
                print(f"{name}: already running")
                continue
            subprocess.run(["tmux", "kill-session", "-t", f"={name}"], check=True)
        arguments = [str(PYTHON), str(Path(__file__).resolve()), "worker", lane]
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "-c",
                str(ROOT),
                shlex.join(arguments),
            ],
            check=True,
        )
        subprocess.run(
            ["tmux", "set-option", "-t", f"={name}", "remain-on-exit", "on"], check=True
        )
        print(f"{name}: started {' -> '.join(suites)}")


def status():
    for lane in LANES:
        path = SESSIONS / f"{lane}.json"
        if not path.exists():
            print(f"sht-{lane}: not started")
            continue
        state = json.loads(path.read_text())
        suite = state.get("suite")
        done = completed(ROOT / "outputs" / suite) if suite else 0
        print(
            f"sht-{lane}: {state['status']} {suite or ''} {done}/{state.get('total', 0)}"
        )
        if state.get("error"):
            print(f"  {state['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "worker"])
    parser.add_argument("lane", nargs="?", choices=list(LANES))
    args = parser.parse_args()
    if args.action == "worker":
        if args.lane is None:
            parser.error("worker requires a lane")
        worker(args.lane)
    elif args.action == "start":
        start()
    else:
        status()


if __name__ == "__main__":
    main()
