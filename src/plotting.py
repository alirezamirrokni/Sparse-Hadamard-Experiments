import csv
import io
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from .config import CONFIG
from .store import Store, canonical, digest

COLORS = {
    "cs": "#596574",
    "prefix": "#2563A6",
    "prefix_bch": "#2563A6",
    "prefix_robust": "#2563A6",
    "stobbe": "#1B9E77",
    "swht": "#D98324",
    "random": "#777777",
    "gp_ei": "#9467bd",
    "harmonica": "#e69f00",
}
LABELS = {
    "cs": "CS",
    "prefix": "SPHT (Ours)",
    "prefix_bch": "SPHT (Ours)",
    "prefix_robust": "SPHT (Ours)",
    "swht": "SWHT",
    "stobbe": "Stobbe",
    "random": "Random",
    "gp_ei": "GP-EI",
    "harmonica": "Harmonica",
}
ASSESSED = {"ok", "validation_failed", "empty_support", "invalid_syndrome", "InfeasibleSystem", "BudgetExceeded"}
STYLES = ["-", "--", ":", "-."]
METRICS = [
    "queries", "sampled_rows", "algorithm_seconds", "wall_seconds", "relative_error", "edge_error", "error_over_E",
    "best_validation", "test_error", "fit_seconds", "optimization_seconds", "oracle_seconds",
]


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        destination = directory / f"{name}.{extension}"
        temporary = destination.with_name(destination.stem + f".{os.getpid()}.tmp." + extension)
        figure.savefig(temporary, dpi=240, bbox_inches="tight", format=extension)
        temporary.replace(destination)
    plt.close(figure)


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
    writer.writeheader()
    writer.writerows(rows)
    temporary = Path(path).with_name(Path(path).name + f".{os.getpid()}.tmp")
    temporary.write_text(stream.getvalue())
    os.replace(temporary, path)


def wilson(successes, trials):
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return center - radius, center + radius


def dt_curve():
    threshold = np.linspace(0.0001, 4.5, 1500)
    density = norm.pdf(threshold)
    a = density - threshold * norm.sf(threshold)
    return 2 * density / (threshold + 2 * a), a / density


def aggregate(results):
    groups = defaultdict(list)
    for row in results:
        job = row["job"]
        groups[canonical([job["spec"], job["method"], job["settings"]])].append(row)
    summaries = []
    for key, values in groups.items():
        spec, method, settings = json.loads(key)
        eligible = [row for row in values if row["status"] in ASSESSED]
        successes = sum(
            row.get("error_over_E", float("inf")) <= 4 if spec["kind"] == "robust" else bool(row.get("success", False))
            for row in eligible
        )
        statuses = {status: sum(row["status"] == status for row in values) for status in sorted({row["status"] for row in values})}
        summary = {
            **spec,
            "method": method,
            "settings": canonical(settings),
            "trials": len(values),
            "assessed_trials": len(eligible),
            "resource_or_configuration_errors": len(values) - len(eligible),
            "statuses": canonical(statuses),
        }
        if spec["kind"] != "hpo":
            low, high = wilson(successes, len(eligible))
            summary.update(
                successes=successes,
                success_rate=successes / len(eligible) if eligible else None,
                success_low=low,
                success_high=high,
            )
        for metric in METRICS:
            present = [row[metric] for row in values if metric in row and np.isfinite(row[metric])]
            if present:
                summary[metric] = float(np.mean(present))
                summary[f"{metric}_median"] = float(np.median(present))
                summary[f"{metric}_std"] = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
        summaries.append(summary)
    return summaries


def curve(axis, rows, xvalue, metric, method, label=None, style="-", probability=False, color=None):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if row["status"] in ASSESSED and value is not None and np.isfinite(value):
            groups[xvalue(row["job"]["spec"])].append(float(value))
    if not groups:
        return
    xs = sorted(groups)
    centers, lower, upper = [], [], []
    for x in xs:
        values = groups[x]
        if probability:
            centers.append(float(np.mean(values)))
            lo, hi = wilson(sum(values), len(values))
        else:
            centers.append(float(np.median(values)))
            lo, hi = np.quantile(values, [0.25, 0.75])
        lower.append(lo)
        upper.append(hi)
    color = color or COLORS[method]
    axis.plot(xs, centers, style, marker="o", ms=6, lw=2.4, color=color, label=label or LABELS[method])
    if any(len(groups[x]) > 1 for x in xs):
        axis.fill_between(xs, lower, upper, color=color, alpha=0.08)


def phase_surface(deltas, rhos, matrix, resolution=501):
    if min(len(deltas), len(rhos)) < 2:
        return np.asarray(deltas), np.asarray(rhos), matrix.copy()
    xs = np.linspace(min(deltas), max(deltas), resolution)
    ys = np.linspace(min(rhos), max(rhos), resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((yy.ravel(), xx.ravel()))
    finite = np.isfinite(matrix)
    values = RegularGridInterpolator((rhos, deltas), np.where(finite, matrix, 0.0))(points).reshape(xx.shape)
    coverage = RegularGridInterpolator((rhos, deltas), finite.astype(float))(points).reshape(xx.shape)
    values[coverage < 1 - 1e-12] = np.nan
    return xs, ys, values


def phase_matrix(results, method, size, distribution, deltas, rhos):
    groups = defaultdict(list)
    for record in results:
        spec = record["job"]["spec"]
        if record["job"]["method"] == method and spec["size"] == size and spec["distribution"] == distribution:
            if record["status"] in ASSESSED:
                groups[(spec["delta"], spec["rho"])].append(bool(record.get("success")))
    matrix = np.full((len(rhos), len(deltas)), np.nan)
    for (delta, rho), values in groups.items():
        if delta in deltas and rho in rhos:
            matrix[rhos.index(rho), deltas.index(delta)] = np.mean(values)
    return matrix


def plot_phase(results, config, directory):
    for sparse, filename in [(False, "phase_diagrams"), (True, "phase_sparse")]:
        columns = [
            (size, distribution, experiment)
            for experiment in config["experiments"]
            if (max(experiment["deltas"]) <= 0.08) == sparse
            for size in experiment["sizes"]
            for distribution in experiment.get("distributions", ["rademacher"])
        ]
        if not columns:
            continue
        figure, axes = plt.subplots(2, len(columns), figsize=(16, 7.4), squeeze=False, layout="constrained")
        cmap = plt.get_cmap("jet_r").resampled(1024)
        cmap.set_bad("#e9e9e9")
        for column, (size, distribution, experiment) in enumerate(columns):
            deltas, rhos = experiment["deltas"], experiment["rhos"]
            for row_index, method in enumerate(["cs", "prefix"]):
                axis = axes[row_index, column]
                xs, ys, surface = phase_surface(deltas, rhos, phase_matrix(results, method, size, distribution, deltas, rhos))
                if min(len(xs), len(ys)) >= 2:
                    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
                    mesh = axis.imshow(
                        np.ma.masked_invalid(surface), origin="lower",
                        extent=(xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2),
                        aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
                    )
                else:
                    mesh = axis.pcolormesh(xs, ys, surface, shading="nearest", cmap=cmap, vmin=0, vmax=1, rasterized=True)
                finite = surface[np.isfinite(surface)]
                if len(finite) and np.min(finite) < 0.5 < np.max(finite):
                    axis.contour(
                        xs, ys, surface, levels=[0.5], colors="black" if sparse else "white", linewidths=2.2 if sparse else 2.4
                    )
                if not sparse:
                    axis.plot(*dt_curve(), color="black", lw=2.2)
                axis.set(
                    xlim=(min(deltas), max(deltas)),
                    ylim=(min(rhos), max(rhos)),
                    xlabel=r"Query fraction $Q/N$" if row_index else "",
                )
                if column == 0:
                    axis.set_ylabel(LABELS[method] + "\n" + r"Sparsity fraction $k/Q$", labelpad=12)
                axis.set_xticks([0.01, 0.04, 0.08] if sparse else [0.2, 0.5, 0.8])
                if sparse:
                    axis.set_yticks([0.1, 0.2, 0.3] if max(rhos) > 0.15 else [0.05, 0.1, 0.15])
                else:
                    axis.set_yticks([0.2, 0.5, 0.8])
                if column > 0:
                    axis.tick_params(labelleft=False)
                if row_index == 0:
                    axis.set_title(f"N = {2**size:,}" if sparse else f"N = {2**size:,}\n{distribution.title()}")
        figure.colorbar(mesh, ax=axes.ravel().tolist(), label="Recovery probability", shrink=0.9, pad=0.02)
        save_figure(figure, directory, filename)


def comparison_style():
    return plt.rc_context(
        {**sns.axes_style("darkgrid"), "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False}
    )


def comparison_legend(figure, axes, directory, name, methods=("cs", "prefix_bch", "stobbe", "swht"), headings=(), extra_handles=()):
    methods = sorted(methods, key=lambda method: LABELS[method].startswith("SPHT (Ours)"))
    handles = [Line2D([0], [0], color=COLORS[m], lw=2.4, marker="o", label=LABELS[m]) for m in methods]
    if extra_handles:
        handles[-1:-1] = extra_handles
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0.13, 1, 0.9 if headings else 1), pad=1.0, w_pad=1.0)
    for first, last, text in headings:
        center = (axes[first].get_position().x0 + axes[last].get_position().x1) / 2
        figure.text(center, 0.97, text, ha="center", va="top", fontsize=14)
    save_figure(figure, directory, name)


def blocks(rows):
    groups = defaultdict(dict)
    for row in rows:
        job = row["job"]
        groups[digest([job["spec"], job["seed"], job["trial"]])][job["method"]] = row
    return groups.values()


def matched_successes(results):
    selected = []
    for pair in blocks(results):
        other = "prefix_bch" if "prefix_bch" in pair else "prefix"
        if "cs" in pair and other in pair and pair["cs"].get("success") and pair[other].get("success"):
            selected.extend([pair["cs"], pair[other]])
    return selected


def interval(rows, metric, draws, seed, stratify=False):
    values = [row[metric] for row in rows if metric in row and np.isfinite(row[metric])]
    if not values:
        return None
    center = float(np.median(values))
    if len(values) < 2:
        return center, None, None
    groups = defaultdict(list)
    for row in rows:
        if metric in row and np.isfinite(row[metric]):
            groups[row["job"]["spec"].get("snapshot", 0) if stratify else 0].append(row[metric])
    rng = np.random.default_rng(seed)
    samples = [np.asarray(g, float)[rng.integers(0, len(g), (draws, len(g)))] for g in groups.values()]
    low, high = np.quantile(np.median(np.concatenate(samples, axis=1), axis=1), [0.025, 0.975])
    return center, float(low), float(high)


def error_curve(axis, points, metric, method, draws, seed, style="-", stratify=False):
    valid, centers, lower, upper = [], [], [], []
    for x in sorted(points):
        estimate = interval(points[x], metric, draws, int(digest([seed, method, metric, x])[:8], 16), stratify)
        if estimate is None:
            continue
        center, low, high = estimate
        valid.append(x)
        centers.append(center)
        lower.append(max(0.0, center - low) if low is not None else np.nan)
        upper.append(max(0.0, high - center) if high is not None else np.nan)
    if valid:
        axis.errorbar(
            valid, centers, yerr=np.asarray([lower, upper]), fmt=style + "o", color=COLORS[method],
            lw=2.2, ms=5, capsize=3, elinewidth=1.2, capthick=1.2,
        )


def format_axes(axes):
    for axis in np.asarray(axes).ravel():
        axis.yaxis.set_minor_formatter(NullFormatter())
        if axis.get_yscale() == "log" and axis.get_ylim()[1] / axis.get_ylim()[0] < 100:
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=7))
            axis.yaxis.set_major_formatter(ScalarFormatter())
        if axis.get_xscale() == "log" and axis.lines and axis.get_xlim()[1] / axis.get_xlim()[0] < 10:
            xs = sorted({float(x) for line in axis.lines for x in line.get_xdata()})
            axis.set_xticks(xs if len(xs) <= 3 else [xs[0], xs[len(xs) // 2], xs[-1]])
            axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())


def placeholder(axis, text, x=0.5, y=0.5, **options):
    axis.text(x, y, text, ha="center", va="center", fontsize=11, **options)


def plot_graphs(rows, directory, policy):
    methods = ("cs", "stobbe", "swht", "prefix_bch")
    panels = [
        ("graph_edges", "edges", "algorithm_seconds", "Edges", "CPU runtime (s)"),
        ("graph_edges", "edges", "queries", "Edges", "Unique queries"),
        ("graph_vertices", "dimension", "algorithm_seconds", "Vertices", "CPU runtime (s)"),
        ("graph_vertices", "dimension", "queries", "Vertices", "Unique queries"),
    ]
    with comparison_style():
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.8))
        for axis, (kind, xkey, metric, xlabel, ylabel) in zip(axes, panels):
            for method in methods:
                points = defaultdict(list)
                for row in rows:
                    job = row["job"]
                    if job["method"] == method and job["spec"]["kind"] == kind and row["status"] != "resource_limit":
                        points[job["spec"][xkey]].append(row)
                error_curve(axis, points, metric, method, policy["bootstrap_draws"], policy["seed"], stratify=True)
            axis.set(xlabel=xlabel, ylabel=ylabel, yscale="log")
            if not axis.lines:
                placeholder(axis, "Pending measurements", transform=axis.transAxes)
        format_axes(axes)
        comparison_legend(figure, axes, directory, "graphs", headings=[(0, 1, "Vertices = 100"), (2, 3, "Edges = 50")])
        figure, axis = plt.subplots(figsize=(8, 4.2))
        eligible = [r for r in rows if r["job"]["spec"]["kind"] != "graph_vertices" or r["job"]["spec"]["dimension"] <= 210]
        common = [row for group in blocks(eligible) if all(m in group for m in methods) for row in group.values()]
        for index, method in enumerate(methods):
            centers, lower, upper = [], [], []
            for kind in ["graph_edges", "graph_vertices"]:
                selected = [r for r in common if r["job"]["method"] == method and r["job"]["spec"]["kind"] == kind]
                successes = sum(bool(r["success"]) for r in selected)
                center = successes / len(selected) if selected else np.nan
                low, high = wilson(successes, len(selected))
                centers.append(center)
                lower.append(max(0.0, center - low) if low is not None else np.nan)
                upper.append(max(0.0, high - center) if high is not None else np.nan)
            axis.barh(
                np.arange(2) + (index - 1.5) * 0.19, centers, height=0.18, color=COLORS[method],
                xerr=np.asarray([lower, upper]), error_kw={"capsize": 3, "elinewidth": 1.2, "ecolor": "#333333"},
            )
        for position, kind in enumerate(["graph_edges", "graph_vertices"]):
            if not any(row["job"]["spec"]["kind"] == kind for row in common):
                placeholder(axis, "Pending measurements", y=position)
        axis.set(
            yticks=np.arange(2),
            yticklabels=["Edges", r"Vertices ($n\leq210$)"],
            xlim=(0, 1),
            xlabel="Exact recovery on shared graph configurations",
        )
        comparison_legend(figure, [axis], directory, "graph_recovery")


def plot_scaling(rows, directory, policy):
    paired = matched_successes(rows)
    sparsities = sorted({row["job"]["spec"]["sparsity"] for row in rows})
    styles = dict(zip(sparsities, STYLES))
    handles = [Line2D([0], [0], color="#596574", lw=2.2, linestyle=styles[k], label=rf"$k = {k}$") for k in sparsities]
    with comparison_style():
        for metric, ylabel, name in [
            ("algorithm_seconds", "CPU runtime (s)", "scaling_runtime"),
            ("queries", "Unique queries", "scaling_queries"),
        ]:
            figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            for axis, degree, title in zip(axes, [None, 2, 3], ["Jointly successful instances", "Degree 2", "Degree 3"]):
                for method in ["cs", "prefix" if degree is None else "prefix_bch"]:
                    for sparsity in sparsities:
                        points = defaultdict(list)
                        for row in paired:
                            spec = row["job"]["spec"]
                            if row["job"]["method"] == method and spec.get("degree") == degree and spec["sparsity"] == sparsity:
                                points[2 ** spec["dimension"] if degree is None else spec["dimension"]].append(row)
                        error_curve(axis, points, metric, method, policy["bootstrap_draws"], policy["seed"], styles[sparsity])
                axis.set(
                    xlabel=r"$N = 2^n$" if degree is None else "Boolean variables",
                    ylabel=ylabel, title=title, xscale="log", yscale="log",
                )
                if not axis.lines:
                    placeholder(axis, "No jointly successful trials yet", transform=axis.transAxes)
            format_axes(axes)
            comparison_legend(figure, axes, directory, name, methods=("cs", "prefix"), extra_handles=handles)


def plot_robust(results, directory):
    methods = ("cs", "prefix_robust", "stobbe", "swht")
    with comparison_style():
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        groups = defaultdict(list)
        for record in results:
            if record.get("error_over_E", 0) > 0:
                job = record["job"]
                groups[(job["method"], job["spec"]["sparsity"], job["spec"]["tail_ratio"])].append(record["error_over_E"])
        intervals = [np.quantile(values, [0.25, 0.75]) for values in groups.values()]
        limits = (min(lo for lo, _ in intervals) / 1.08, max(hi for _, hi in intervals) * 1.08) if intervals else (0.1, 1)
        for axis, sparsity in zip(axes, [8, 16, 32]):
            for method in methods:
                rows = [r for r in results if r["job"]["method"] == method and r["job"]["spec"]["sparsity"] == sparsity]
                curve(axis, rows, lambda spec: spec["tail_ratio"], "error_over_E", method)
            axis.set(
                xlabel="Tail / head norm ratio",
                ylabel="Recovery error / E" if sparsity == 8 else "",
                xscale="log", yscale="log", ylim=limits, title=f"k = {sparsity}",
            )
            if limits[1] / limits[0] >= 10:
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                axis.yaxis.set_major_formatter(LogFormatterMathtext())
            else:
                axis.yaxis.set_major_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 8)))
                axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(NullFormatter())
        comparison_legend(figure, axes, directory, "robustness_error", methods)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for method in methods:
            rows = [r for r in results if r["job"]["method"] == method]
            assessed = [{**r, "within_bound": r["error_over_E"] <= 4} for r in rows if "error_over_E" in r]
            curve(axes[0], assessed, lambda spec: spec["tail_ratio"], "within_bound", method, probability=True)
            curve(axes[1], rows, lambda spec: spec["sparsity"], "queries", method)
            curve(axes[2], rows, lambda spec: spec["sparsity"], "algorithm_seconds", method)
        axes[0].set(
            xlabel="Tail / head norm ratio", ylabel="Fraction with error ≤ 4E", xscale="log", ylim=(0, 1.03),
            title="Recovery within 4E",
        )
        axes[1].set(xlabel="k", ylabel="Oracle queries", yscale="log", title="Sampling cost")
        axes[2].set(xlabel="k", ylabel="Runtime (s)", yscale="log", title="Computational cost")
        axes[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        axes[1].yaxis.set_minor_formatter(NullFormatter())
        for axis in axes[1:]:
            axis.set_xticks([8, 16, 32])
        comparison_legend(figure, axes, directory, "robustness_cost", methods)


def plot_trace(axis, rows, method, time_axis=False):
    rows = [row for row in rows if row.get("trace")]
    if not rows:
        return
    column = 2 if time_axis else 0
    end = min(row["trace"][-1][column] for row in rows)
    start = max(row["trace"][0][column] for row in rows)
    if end < start:
        return
    grid = np.linspace(start, end, 150)
    values = []
    for row in rows:
        trace = np.asarray(row["trace"], dtype=float)
        values.append(trace[np.maximum(0, np.searchsorted(trace[:, column], grid, side="right") - 1), 1])
    array = np.asarray(values)
    axis.plot(grid, np.median(array, axis=0), color=COLORS[method], label=LABELS[method])
    if len(rows) > 1:
        axis.fill_between(
            grid, np.quantile(array, 0.25, axis=0), np.quantile(array, 0.75, axis=0), color=COLORS[method], alpha=0.1
        )


def finish_hpo(figure, axes, directory):
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
        handles, labels = axis.get_legend_handles_labels()
        if not (axis.lines or axis.collections or axis.patches):
            axis.text(0.5, 0.5, "Awaiting trials", ha="center", va="center", transform=axis.transAxes, color="#777777")
        elif handles:
            handles, labels = zip(*sorted(zip(handles, labels), key=lambda entry: entry[1].startswith("SPHT (Ours)")))
            if len(handles) > 4:
                axis.legend(
                    handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=10.5,
                    frameon=False, handlelength=1.5, handletextpad=0.4, labelspacing=0.2,
                )
            else:
                axis.legend(handles, labels, loc="best", fontsize=11, frameon=False, handlelength=1.5)
    for axis in axes.ravel():
        axis.set_title(textwrap.fill(axis.get_title(), 38))
        if "$" not in axis.get_ylabel():
            axis.set_ylabel(textwrap.fill(axis.get_ylabel(), 36))
        if "$" not in axis.get_xlabel():
            axis.set_xlabel(textwrap.fill(axis.get_xlabel(), 38))
        if axis.get_xscale() == "linear":
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        elif axis.get_xlim()[1] / axis.get_xlim()[0] < 10:
            xs = sorted({float(x) for line in axis.lines for x in line.get_xdata() if x > 0})
            if xs:
                axis.set_xticks(xs if len(xs) <= 3 else [xs[0], xs[len(xs) // 2], xs[-1]])
                axis.xaxis.set_major_formatter(ScalarFormatter())
                axis.xaxis.set_minor_formatter(NullFormatter())
        if axis.get_yscale() == "log":
            axis.yaxis.set_minor_formatter(NullFormatter())
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.95, hspace=0.85, wspace=0.45)
    save_figure(figure, directory, "hpo")


def plot_hpo(results, directory):
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    results = [r for r in results if r["status"] == "ok"]
    maximum_budget = max((r["job"]["spec"]["budget"] for r in results if r["job"]["spec"]["task"] == "digits"), default=320)
    dimension = lambda spec: spec["dimension"]
    for method in ["random", "gp_ei", "harmonica", "prefix_bch"]:
        mine = [r for r in results if r["job"]["method"] == method]
        digits = [r for r in mine if r["job"]["spec"]["task"] == "digits" and r["job"]["spec"]["budget"] == maximum_budget]
        base = [r for r in digits if r["job"]["spec"]["dimension"] == 12]
        plot_trace(axes[0, 0], base, method)
        plot_trace(axes[0, 1], base, method, True)
        curve(axes[0, 2], digits, dimension, "test_error", method)
        for column, task in enumerate(["sparse", "hierarchical"]):
            for index, noise in enumerate([0, 1]):
                rows = [r for r in mine if r["job"]["spec"]["task"] == task and r["job"]["spec"].get("noise", 0) == noise]
                curve(axes[1, column], rows, dimension, "best_validation", method, f"{LABELS[method]}, noise={noise}", STYLES[index])
        overhead = [{**r, "overhead": r.get("fit_seconds", 0) + r.get("optimization_seconds", 0)} for r in digits]
        curve(axes[1, 2], overhead, dimension, "overhead", method)
    axes[0, 0].set(xlabel="Unique objective evaluations", ylabel="Best validation error", title="Digits: 12 effective variables")
    axes[0, 1].set(
        xlabel="Cold-oracle cost + optimizer overhead (s)", ylabel="Best validation error",
        title="Digits: quality versus total cost",
    )
    axes[0, 2].set(
        xlabel="Binary variables including dummies", ylabel="Final test error",
        title="Digits: selected models on held-out test data",
    )
    axes[1, 0].set(xlabel="Binary variables", ylabel="Final objective", title="Sparse polynomial")
    axes[1, 1].set(xlabel="Binary variables", ylabel="Final objective", title="Hierarchical objective")
    axes[1, 2].set(
        xlabel="Binary variables including dummies", ylabel="Optimizer overhead (s)", yscale="log",
        title="Digits: computational scaling",
    )
    finish_hpo(figure, axes, directory)


def paired_effects(rows, policy):
    groups = defaultdict(list)
    for block in blocks(rows):
        spht = block.get("prefix_bch", block.get("prefix"))
        cs = block.get("cs")
        if cs is not None and spht is not None and cs["success"] and spht["success"]:
            groups[digest(cs["job"]["spec"])].append((cs, spht))
    output = []
    for key, pairs in groups.items():
        item = {**pairs[0][0]["job"]["spec"], "jointly_successful_pairs": len(pairs)}
        rng = np.random.default_rng(int(digest([policy["seed"], "paired-effects", key])[:8], 16))
        indices = rng.integers(0, len(pairs), (policy["bootstrap_draws"], len(pairs))) if len(pairs) > 1 else None
        for name, ratios in [
            ("cpu_speedup_cs_over_spht", [a["algorithm_seconds"] / b["algorithm_seconds"] for a, b in pairs]),
            ("query_ratio_spht_over_cs", [b["queries"] / a["queries"] for a, b in pairs]),
        ]:
            logs = np.log(np.asarray(ratios))
            item[name] = float(np.exp(np.mean(logs)))
            if indices is not None:
                low, high = np.quantile(np.exp(np.mean(logs[indices], axis=1)), [0.025, 0.975])
                item[name + "_low"] = float(low)
                item[name + "_high"] = float(high)
        output.append(item)
    return output


def coverage(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[canonical([row["job"]["spec"], row["job"]["method"]])].append(row)
    pair_counts = Counter(canonical(row["job"]["spec"]) for row in matched_successes(rows))
    output = []
    for values in groups.values():
        spec = values[0]["job"]["spec"]
        successes = sum(bool(row["success"]) for row in values)
        low, high = wilson(successes, len(values))
        output.append({
            **spec,
            "method": values[0]["job"]["method"],
            "completed": len(values),
            "successes": successes,
            "recovery": successes / len(values),
            "recovery_low": low,
            "recovery_high": high,
            "jointly_successful_pairs": pair_counts[canonical(spec)] // 2,
            "timeouts": sum(row["status"] == "timeout" for row in values),
            "resource_limits": sum(row["status"] == "resource_limit" for row in values),
        })
    return output


def render(output, suite, implementation):
    directory = Path(output) / suite
    store = Store(directory / "cache.sqlite")
    rows = [row for _, row in store.iter_records("result") if row.get("implementation") == implementation]
    store.close()
    plt.rcParams.update({
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
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figures = Path(output) / "figures"
    policy = CONFIG["evaluation"]
    if suite == "graphs":
        plot_graphs(rows, figures, policy)
    elif suite == "scaling":
        plot_scaling(rows, figures, policy)
    elif suite == "phase":
        plot_phase(rows, CONFIG["suites"]["phase"], figures)
    elif suite == "robust":
        plot_robust(rows, figures)
    else:
        plot_hpo(rows, figures)
    write_csv(directory / "summary.csv", aggregate(rows))
    if suite in {"graphs", "scaling"}:
        write_csv(directory / "paired-comparisons.csv", paired_effects(rows, policy))
        write_csv(directory / "coverage.csv", coverage(rows))
