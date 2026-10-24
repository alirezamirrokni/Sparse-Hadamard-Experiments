import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from src.plotting import (
    COLORS,
    STYLES,
    comparison_legend,
    comparison_style,
    matched_successes,
    wilson,
)
from src.storage import Store, atomic_text, digest

from .scheduling import timing_slot


def records(directory, implementation):
    store = Store(Path(directory) / "cache.sqlite")
    rows = [
        {k: v for k, v in row.items() if k not in {"frequencies", "values", "stages"}}
        for _, row in store.iter_records("result")
        if row.get("implementation") == implementation
    ]
    store.close()
    return rows


def csv_file(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=sorted(set().union(*(row.keys() for row in rows)))
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def interval(rows, metric, draws, seed, stratify=False):
    values = [row[metric] for row in rows if metric in row and np.isfinite(row[metric])]
    if not values:
        return None
    center = float(np.median(values))
    if len(values) < 2:
        return center, None, None
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            groups[row["job"]["spec"].get("snapshot", 0) if stratify else 0].append(
                value
            )
    samples = []
    for group in groups.values():
        group = np.asarray(group, float)
        samples.append(group[rng.integers(0, len(group), (draws, len(group)))])
    medians = np.median(np.concatenate(samples, axis=1), axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return center, float(low), float(high)


def error_curve(axis, points, metric, method, draws, seed, style="-", stratify=False):
    xs = sorted(points)
    centers = []
    lower = []
    upper = []
    valid = []
    for x in xs:
        estimate = interval(
            points[x],
            metric,
            draws,
            int(digest([seed, method, metric, x])[:8], 16),
            stratify,
        )
        if estimate is None:
            continue
        center, low, high = estimate
        valid.append(x)
        centers.append(center)
        lower.append(max(0.0, center - low) if low is not None else np.nan)
        upper.append(max(0.0, high - center) if high is not None else np.nan)
    if valid:
        axis.errorbar(
            valid,
            centers,
            yerr=np.asarray([lower, upper]),
            fmt=style + "o",
            color=COLORS[method],
            lw=2.2,
            ms=5,
            capsize=3,
            elinewidth=1.2,
            capthick=1.2,
        )


def format_axes(axes):
    for axis in np.asarray(axes).ravel():
        axis.yaxis.set_minor_formatter(NullFormatter())
        if axis.get_yscale() == "log" and axis.get_ylim()[1] / axis.get_ylim()[0] < 100:
            axis.yaxis.set_major_locator(
                LogLocator(base=10, subs=(1, 2, 5), numticks=7)
            )
            axis.yaxis.set_major_formatter(ScalarFormatter())
        if (
            axis.get_xscale() == "log"
            and axis.lines
            and axis.get_xlim()[1] / axis.get_xlim()[0] < 10
        ):
            xs = sorted({float(x) for line in axis.lines for x in line.get_xdata()})
            axis.set_xticks(xs if len(xs) <= 3 else [xs[0], xs[len(xs) // 2], xs[-1]])
            axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())


def graph_plots(rows, directory, policy):
    methods = ("cs", "stobbe", "swht", "prefix_bch")
    panels = [
        ("graph_edges", "edges", "algorithm_seconds", "Edges", "CPU runtime (s)"),
        ("graph_edges", "edges", "queries", "Edges", "Unique queries"),
        (
            "graph_vertices",
            "dimension",
            "algorithm_seconds",
            "Vertices",
            "CPU runtime (s)",
        ),
        ("graph_vertices", "dimension", "queries", "Vertices", "Unique queries"),
    ]
    with comparison_style():
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.8))
        for axis, (kind, xkey, metric, xlabel, ylabel) in zip(axes, panels):
            for method in methods:
                points = defaultdict(list)
                for row in rows:
                    if (
                        row["job"]["method"] == method
                        and row["job"]["spec"]["kind"] == kind
                        and row["status"] != "resource_limit"
                    ):
                        points[row["job"]["spec"][xkey]].append(row)
                error_curve(
                    axis,
                    points,
                    metric,
                    method,
                    policy["bootstrap_draws"],
                    policy["seed"],
                    stratify=True,
                )
            axis.set(xlabel=xlabel, ylabel=ylabel, yscale="log")
        format_axes(axes)
        comparison_legend(
            figure,
            axes,
            directory,
            "graphs",
            headings=[(0, 1, "Vertices = 100"), (2, 3, "Edges = 50")],
        )
        figure, axis = plt.subplots(figsize=(8, 4.2))
        blocks = defaultdict(dict)
        for row in rows:
            spec = row["job"]["spec"]
            if spec["kind"] == "graph_vertices" and spec["dimension"] > 210:
                continue
            key = digest([spec, row["job"]["trial"], row["job"]["seed"]])
            blocks[key][row["job"]["method"]] = row
        common = [
            row
            for group in blocks.values()
            if all(method in group for method in methods)
            for row in group.values()
        ]
        for index, method in enumerate(methods):
            centers = []
            lower = []
            upper = []
            for kind in ["graph_edges", "graph_vertices"]:
                selected = [
                    r
                    for r in common
                    if r["job"]["method"] == method and r["job"]["spec"]["kind"] == kind
                ]
                n = len(selected)
                successes = sum(bool(r["success"]) for r in selected)
                center = successes / n if n else np.nan
                low, high = wilson(successes, n)
                centers.append(center)
                lower.append(max(0.0, center - low) if low is not None else np.nan)
                upper.append(max(0.0, high - center) if high is not None else np.nan)
            axis.barh(
                np.arange(2) + (index - 1.5) * 0.19,
                centers,
                height=0.18,
                color=COLORS[method],
                xerr=np.asarray([lower, upper]),
                error_kw={"capsize": 3, "elinewidth": 1.2, "ecolor": "#333333"},
            )
        axis.set(
            yticks=np.arange(2),
            yticklabels=["Edges", r"Vertices ($n\leq210$)"],
            xlim=(0, 1),
            xlabel="Exact recovery on shared graph configurations",
        )
        comparison_legend(figure, [axis], directory, "graph_recovery")


def scaling_plots(rows, directory, policy):
    paired = matched_successes(rows)
    sparsities = sorted({row["job"]["spec"]["sparsity"] for row in rows})
    styles = dict(zip(sparsities, STYLES))
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color="#596574", lw=2.2, linestyle=styles[k], label=rf"$k = {k}$"
        )
        for k in sparsities
    ]
    with comparison_style():
        for metric, ylabel, name in [
            ("algorithm_seconds", "CPU runtime (s)", "scaling_runtime"),
            ("queries", "Unique queries", "scaling_queries"),
        ]:
            figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            for axis, degree, title in zip(
                axes,
                [None, 2, 3],
                ["Jointly successful instances", "Degree 2", "Degree 3"],
            ):
                for method in ["cs", "prefix" if degree is None else "prefix_bch"]:
                    for sparsity in sparsities:
                        points = defaultdict(list)
                        for row in paired:
                            spec = row["job"]["spec"]
                            if (
                                row["job"]["method"] == method
                                and spec.get("degree") == degree
                                and spec["sparsity"] == sparsity
                            ):
                                points[
                                    2 ** spec["dimension"]
                                    if degree is None
                                    else spec["dimension"]
                                ].append(row)
                        error_curve(
                            axis,
                            points,
                            metric,
                            method,
                            policy["bootstrap_draws"],
                            policy["seed"],
                            styles[sparsity],
                        )
                axis.set(
                    xlabel=r"$N = 2^n$" if degree is None else "Boolean variables",
                    ylabel=ylabel,
                    title=title,
                    xscale="log",
                    yscale="log",
                )
                if not axis.lines:
                    axis.text(
                        0.5,
                        0.5,
                        "No jointly successful trials yet",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=11,
                    )
            format_axes(axes)
            comparison_legend(
                figure,
                axes,
                directory,
                name,
                methods=("cs", "prefix"),
                extra_handles=handles,
            )


def paired_effects(rows, policy):
    blocks = defaultdict(dict)
    for row in rows:
        job = row["job"]
        blocks[digest([job["spec"], job["seed"], job["trial"]])][job["method"]] = row
    groups = defaultdict(list)
    for block in blocks.values():
        spht = block.get("prefix_bch", block.get("prefix"))
        cs = block.get("cs")
        if cs is not None and spht is not None and cs["success"] and spht["success"]:
            groups[digest(cs["job"]["spec"])].append((cs, spht))
    output = []
    for key, pairs in groups.items():
        item = {**pairs[0][0]["job"]["spec"], "jointly_successful_pairs": len(pairs)}
        rng = np.random.default_rng(
            int(digest([policy["seed"], "paired-effects", key])[:8], 16)
        )
        indices = (
            rng.integers(0, len(pairs), (policy["bootstrap_draws"], len(pairs)))
            if len(pairs) > 1
            else None
        )
        for name, ratios in [
            (
                "cpu_speedup_cs_over_spht",
                [a["algorithm_seconds"] / b["algorithm_seconds"] for a, b in pairs],
            ),
            (
                "query_ratio_spht_over_cs",
                [b["queries"] / a["queries"] for a, b in pairs],
            ),
        ]:
            logs = np.log(np.asarray(ratios))
            item[name] = float(np.exp(np.mean(logs)))
            if indices is not None:
                low, high = np.quantile(
                    np.exp(np.mean(logs[indices], axis=1)), [0.025, 0.975]
                )
                item[name + "_low"] = float(low)
                item[name + "_high"] = float(high)
        output.append(item)
    return output


def render(directory, suite, bundle, implementation):
    directory = Path(directory)
    if directory.parent.name == "paper":
        lock_directory = directory.parent
    else:
        lock_directory = directory.parent / "paper"
    lock_directory.mkdir(parents=True, exist_ok=True)
    with timing_slot(lock_directory):
        _render(directory, suite, bundle, implementation)


def _render(directory, suite, bundle, implementation):
    directory = Path(directory)
    if not (directory / "cache.sqlite").exists():
        return
    rows = records(directory, implementation)
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if suite == "graphs":
        graph_plots(rows, directory / "figures", bundle["paper"])
    else:
        scaling_plots(rows, directory / "figures", bundle["paper"])
    groups = defaultdict(list)
    for row in rows:
        groups[digest([row["job"]["spec"], row["job"]["method"]])].append(row)
    summary = []
    for values in groups.values():
        spec = values[0]["job"]["spec"]
        method = values[0]["job"]["method"]
        n = len(values)
        successes = sum(bool(r["success"]) for r in values)
        low, high = wilson(successes, n)
        item = {
            **spec,
            "method": method,
            "completed": n,
            "successes": successes,
            "recovery": successes / n,
            "recovery_low": low,
            "recovery_high": high,
            "timeouts": sum(r["status"] == "timeout" for r in values),
            "resource_limits": sum(r["status"] == "resource_limit" for r in values),
        }
        for metric in [
            "algorithm_seconds",
            "optimizer_wall_seconds",
            "wall_seconds",
            "queries",
        ]:
            estimate = interval(
                values,
                metric,
                bundle["paper"]["bootstrap_draws"],
                int(digest([spec, method, metric])[:8], 16),
            )
            if estimate is not None:
                (
                    item[metric + "_median"],
                    item[metric + "_low"],
                    item[metric + "_high"],
                ) = estimate
        summary.append(item)
    csv_file(directory / "summary.csv", summary)
    if suite == "scaling":
        paired = matched_successes(rows)
        counts = Counter(digest(row["job"]["spec"]) for row in paired)
        for item in summary:
            spec = {
                k: v
                for k, v in item.items()
                if k in {"kind", "dimension", "degree", "sparsity"}
            }
            item["jointly_successful_pairs"] = counts[digest(spec)] // 2
        csv_file(directory / "coverage.csv", summary)
    csv_file(
        directory / "paired-comparisons.csv", paired_effects(rows, bundle["paper"])
    )
    statuses = Counter(row["status"] for row in rows)
    atomic_text(
        directory / "status.json",
        json.dumps(
            {
                "completed_trials": len(rows),
                "statuses": dict(statuses),
                "publication_ready": (directory / "audit.json").exists(),
            },
            indent=2,
        ),
    )

    state_path = directory.parent / "paper" / f"{suite}-state.json"
    if (
        state_path.exists()
        and json.loads(state_path.read_text()).get("implementation") == implementation
    ):
        destination = directory.parent / "figures"
        destination.mkdir(exist_ok=True)
        for path in (directory / "figures").glob("*"):
            if path.suffix in {".png", ".pdf"}:
                temporary = destination / (path.name + ".paper.tmp")
                shutil.copyfile(path, temporary)
                temporary.replace(destination / path.name)


def audit(directory, suite, jobs, implementation):
    rows = records(directory, implementation)
    expected = {digest({"job": job, "implementation": implementation}) for job in jobs}
    with closing(sqlite3.connect(Path(directory) / "cache.sqlite")) as database:
        actual = {
            key
            for (key,) in database.execute(
                "SELECT key FROM records WHERE kind='result'"
            )
        }
    if expected != actual:
        raise RuntimeError("Publication audit failed: incomplete or unexpected trials")
    if any(row["status"] == "implementation_error" for row in rows):
        raise RuntimeError(
            "Publication audit failed: an implementation error is recorded"
        )
    groups = Counter(digest([row["job"]["spec"], row["job"]["method"]]) for row in rows)
    required = 20 if suite == "graphs" else 5
    if any(count != required for count in groups.values()):
        raise RuntimeError("Publication audit failed: incorrect repetition count")
    compiler = subprocess.run(
        ["g++", "--version"], capture_output=True, text=True, check=False
    )
    native = (
        Path(__file__).resolve().parents[2]
        / "data/originals/code/fastwht/python"
        / ("_hadamardKernel" + sysconfig.get_config_var("EXT_SUFFIX"))
    )
    context = {
        "python": sys.version,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "compiler": compiler.stdout.splitlines()[0] if compiler.stdout else None,
        "original_native_sha256": hashlib.sha256(native.read_bytes()).hexdigest()
        if native.exists()
        else None,
    }
    atomic_text(
        Path(directory) / "audit.json",
        json.dumps(
            {
                "runtime_context": context,
                "publication_ready": True,
                "completed_trials": len(rows),
                "repetitions_per_configuration": required,
                "configuration_count": len(groups),
                "implementation": implementation,
                "resource_limits_and_failures_included": True,
            },
            indent=2,
        ),
    )
