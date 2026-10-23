import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from threadpoolctl import threadpool_limits

from .algebra import warmup
from .experiments import jobs
from .plotting import render
from .storage import (
    Store,
    atomic_text,
    configure_logging,
    digest,
    provenance,
    source_digest,
)
from .tuning import task_payload, tune
from .upstream import fetch_sources


def parser():
    command = argparse.ArgumentParser(prog="shtbench")
    sub = command.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--directory", default="data/upstream")
    for name in ["run", "plan", "tune"]:
        item = sub.add_parser(name)
        item.add_argument("config", type=Path)
        item.add_argument("--output", type=Path, default=Path("outputs") / name)
        item.add_argument("--workers", type=int, default=1)
        if name != "tune":
            item.add_argument("--selection", type=Path)
        if name == "run":
            item.add_argument("--upstream", default="data/upstream")
            item.add_argument("--limit", type=int)
            item.add_argument("--no-plots", action="store_true")
            item.add_argument("--shard", type=int, default=0)
            item.add_argument("--shards", type=int, default=1)
            item.add_argument("--retry-errors", action="store_true")
    plot = sub.add_parser("plot")
    plot.add_argument("directory", type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("databases", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    return command


def main():
    args = parser().parse_args()
    if args.command == "fetch":
        fetch_sources(args.directory)
        print(f"Verified upstream sources and datasets: {args.directory}")
        return
    if args.command == "plot":
        configure_logging(args.directory)
        active = args.directory / "active_run.json"
        identity = json.loads(active.read_text()) if active.exists() else {}
        render(
            args.directory / "cache.sqlite",
            args.directory,
            identity.get("source_sha256"),
            identity.get("timing_workers"),
        )
        return
    if args.command == "merge":
        args.output.mkdir(parents=True, exist_ok=True)
        target = Store(args.output / "cache.sqlite")
        for database in args.databases:
            source = Store(database)
            for key, value in source.records("result"):
                target.put(key, value, "result")
            source.close()
        target.close()
        render(args.output / "cache.sqlite", args.output)
        return
    config = json.loads(args.config.read_text())
    config.setdefault("defaults", {})["timing_workers"] = args.workers
    config["defaults"]["blas_threads"] = 1
    if args.command != "tune" and args.selection:
        frozen = json.loads(args.selection.read_text())
        selected = frozen["selected_settings"]
        groups = [config["methods"]] + [
            experiment["method_settings"]
            for experiment in config["experiments"]
            if "method_settings" in experiment
        ]
        for group in groups:
            for method in group:
                if method.startswith("prefix") and "robust" not in method:
                    original = group[method]
                    group[method] = {**original, **selected}
                    for setting in ["solver", "validated", "penalty", "sample_factor"]:
                        if setting in original:
                            group[method][setting] = original[setting]
        config["selection_sha256"] = digest(frozen)
    if args.command == "tune":
        configure_logging(args.output)
        with threadpool_limits(limits=1):
            warmup()
            tune(config, args.output, args.workers)
        return
    expanded = list(jobs(config))
    if args.command == "plan":
        counts = {}
        for job in expanded:
            name = f"{job['spec']['kind']}/{job['method']}"
            counts[name] = counts.get(name, 0) + 1
        print(
            json.dumps(
                {"total_trials": len(expanded), "by_family_and_method": counts},
                indent=2,
            )
        )
        return
    if args.workers < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid worker or shard count")
    configure_logging(args.output)
    implementation = source_digest()
    database = args.output / "cache.sqlite"
    store = Store(database)
    expected = {
        digest({"job": job, "implementation": implementation})
        for index, job in enumerate(expanded)
        if index % args.shards == args.shard
    }
    if args.retry_errors:
        for key, value in store.records("result"):
            if key in expected and value["status"] in {
                "RuntimeError",
                "ValueError",
                "MemoryError",
                "AssertionError",
                "TimeoutError",
            }:
                with store.db:
                    store.db.execute("DELETE FROM records WHERE key=?", (key,))
    pending = [
        job
        for index, job in enumerate(expanded)
        if index % args.shards == args.shard
        and store.get(digest({"job": job, "implementation": implementation})) is None
    ]
    completed = len(expected) - len(pending)
    store.close()
    if args.limit is not None:
        pending = pending[: args.limit]
    atomic_text(
        args.output / f"manifest_{digest(config)[:12]}.json",
        json.dumps(provenance(config), indent=2),
    )
    atomic_text(
        args.output / "active_run.json",
        json.dumps(
            {
                "source_sha256": implementation,
                "config_sha256": digest(config),
                "timing_workers": args.workers,
            },
            indent=2,
        ),
    )
    logging.info(
        "planned=%d previously_completed=%d pending_this_invocation=%d workers=%d",
        len(expected),
        completed,
        len(pending),
        args.workers,
    )
    payloads = [(job, str(database), args.upstream, implementation) for job in pending]
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        warmup()
        if args.workers == 1:
            for index, payload in enumerate(payloads, 1):
                task_payload(payload)
                elapsed = time.perf_counter() - started
                logging.info(
                    "progress=%d/%d elapsed=%.1fs estimated_remaining=%.1fs",
                    index,
                    len(payloads),
                    elapsed,
                    elapsed * (len(payloads) - index) / index,
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for index, value in enumerate(
                    pool.map(task_payload, payloads, chunksize=1), 1
                ):
                    logging.info(
                        "progress=%d/%d elapsed=%.1fs",
                        index,
                        len(payloads),
                        time.perf_counter() - started,
                    )
    if not args.no_plots:
        render(database, args.output, implementation, args.workers)
    logging.info("finished output=%s", args.output)


if __name__ == "__main__":
    main()
