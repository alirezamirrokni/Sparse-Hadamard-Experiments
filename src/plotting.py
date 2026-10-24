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
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from .config import CONFIG
from .storage import Store, atomic_text, canonical, digest


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        destination = directory / f"{name}.{extension}"
        temporary = destination.with_name(
            destination.stem + f".{os.getpid()}.tmp." + extension
        )
        figure.savefig(temporary, dpi=240, bbox_inches="tight", format=extension)
        temporary.replace(destination)
    plt.close(figure)


def wilson(successes, trials):
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def dt_curve():
    threshold = np.linspace(0.0001, 4.5, 1500)
    density = norm.pdf(threshold)
    tail = norm.sf(threshold)
    a = density - threshold * tail
    delta = 2 * density / (threshold + 2 * a)
    rho = a / density
    return delta, rho


def aggregate(results):
    groups = defaultdict(list)
    for row in results:
        job = row["job"]
        groups[canonical([job["spec"], job["method"], job["settings"]])].append(row)
    summaries = []
    for key, values in groups.items():
        spec, method, settings = json.loads(key)
        eligible = [
            row
            for row in values
            if row["status"]
            in {
                "ok",
                "validation_failed",
                "empty_support",
                "invalid_syndrome",
                "InfeasibleSystem",
                "BudgetExceeded",
            }
        ]
        successes = sum(
            row.get("error_over_E", float("inf")) <= 4
            if spec["kind"] == "robust"
            else bool(row.get("success", False))
            for row in eligible
        )
        low, high = wilson(successes, len(eligible))
        summary = {
            **spec,
            "method": method,
            "settings": canonical(settings),
            "trials": len(values),
            "assessed_trials": len(eligible),
            "resource_or_configuration_errors": len(values) - len(eligible),
            "statuses": canonical(
                {
                    status: sum(row["status"] == status for row in values)
                    for status in sorted({row["status"] for row in values})
                }
            ),
        }
        if spec["kind"] != "hpo":
            summary.update(
                successes=successes,
                success_rate=successes / len(eligible) if eligible else None,
                success_low=low,
                success_high=high,
            )
        for metric in [
            "queries",
            "sampled_rows",
            "algorithm_seconds",
            "wall_seconds",
            "relative_error",
            "edge_error",
            "error_over_E",
            "best_validation",
            "test_error",
            "fit_seconds",
            "optimization_seconds",
            "oracle_seconds",
        ]:
            present = [
                row[metric]
                for row in values
                if metric in row and np.isfinite(row[metric])
            ]
            if present:
                summary[metric] = float(np.mean(present))
                summary[f"{metric}_median"] = float(np.median(present))
                summary[f"{metric}_std"] = (
                    float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
                )
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


COMPARISON_METHODS = ("cs", "prefix_bch", "stobbe", "swht")


COMPARISON_PALETTE = {
    "cs": "#596574",
    "prefix": "#2563A6",
    "prefix_bch": "#2563A6",
    "prefix_robust": "#2563A6",
    "stobbe": "#1B9E77",
    "swht": "#D98324",
}


COLORS = {
    **COMPARISON_PALETTE,
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


ASSESSED = {
    "ok",
    "validation_failed",
    "empty_support",
    "invalid_syndrome",
    "InfeasibleSystem",
    "BudgetExceeded",
}


STYLES = ["-", "--", ":", "-."]


def finish(figure, axes, title, directory, name):
    for axis in np.asarray(axes).ravel():
        axis.grid(alpha=0.18)
        if axis.lines or axis.collections or axis.patches:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                handles, labels = zip(
                    *sorted(
                        zip(handles, labels),
                        key=lambda entry: entry[1].startswith("SPHT (Ours)"),
                    )
                )
                if len(handles) > 4:
                    axis.legend(
                        handles,
                        labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.34),
                        ncol=2,
                        fontsize=10.5,
                        frameon=False,
                        handlelength=1.5,
                        handletextpad=0.4,
                        labelspacing=0.2,
                    )
                else:
                    axis.legend(
                        handles,
                        labels,
                        loc="best",
                        fontsize=11,
                        frameon=False,
                        handlelength=1.5,
                    )
        else:
            axis.text(
                0.5,
                0.5,
                "Awaiting trials",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#777777",
            )
    for axis in np.asarray(axes).ravel():
        axis.set_title(textwrap.fill(axis.get_title(), 38))
        if "$" not in axis.get_ylabel():
            axis.set_ylabel(textwrap.fill(axis.get_ylabel(), 36))
        if "$" not in axis.get_xlabel():
            axis.set_xlabel(textwrap.fill(axis.get_xlabel(), 38))
        if axis.get_xscale() == "linear":
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        elif axis.get_xlim()[1] / axis.get_xlim()[0] < 10:
            xs = sorted(
                {float(x) for line in axis.lines for x in line.get_xdata() if x > 0}
            )
            if xs:
                ticks = xs if len(xs) <= 3 else [xs[0], xs[len(xs) // 2], xs[-1]]
                axis.set_xticks(ticks)
                axis.xaxis.set_major_formatter(ScalarFormatter())
                axis.xaxis.set_minor_formatter(NullFormatter())
        if axis.get_yscale() == "log":
            axis.yaxis.set_minor_formatter(NullFormatter())
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.15, top=0.95, hspace=0.85, wspace=0.45
    )
    save_figure(figure, directory, name)


def curve(
    axis,
    rows,
    xvalue,
    metric,
    method,
    label=None,
    style="-",
    probability=False,
    color=None,
):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] not in ASSESSED:
            continue
        value = row.get(metric)
        if value is not None and np.isfinite(value):
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
    axis.plot(
        xs,
        centers,
        style,
        marker="o",
        ms=6,
        lw=2.4,
        color=color,
        label=label or LABELS[method],
    )
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
    values = RegularGridInterpolator((rhos, deltas), np.where(finite, matrix, 0.0))(
        points
    ).reshape(xx.shape)
    coverage = RegularGridInterpolator((rhos, deltas), finite.astype(float))(
        points
    ).reshape(xx.shape)
    values[coverage < 1 - 1e-12] = np.nan
    return xs, ys, values


def plot_phase_comparisons(results, config, directory):
    for sparse, filename in [(False, "phase_diagrams"), (True, "phase_sparse")]:
        columns = []
        for experiment in config["experiments"]:
            if (max(experiment["deltas"]) <= 0.08) != sparse:
                continue
            for size in experiment["sizes"]:
                for distribution in experiment.get("distributions", ["rademacher"]):
                    columns.append((size, distribution, experiment))
        if not columns:
            continue
        figure, axes = plt.subplots(
            2,
            len(columns),
            figsize=(16, 7.4),
            squeeze=False,
            layout="constrained",
        )
        cmap = plt.get_cmap("jet_r").resampled(1024)
        cmap.set_bad("#e9e9e9")
        for column, (size, distribution, experiment) in enumerate(columns):
            deltas, rhos = experiment["deltas"], experiment["rhos"]
            for row_index, method in enumerate(["cs", "prefix"]):
                groups = defaultdict(list)
                for record in results:
                    spec = record["job"]["spec"]
                    if (
                        record["job"]["method"] != method
                        or spec["size"] != size
                        or spec["distribution"] != distribution
                    ):
                        continue
                    if record["status"] in ASSESSED:
                        groups[(spec["delta"], spec["rho"])].append(
                            bool(record.get("success"))
                        )
                matrix = np.full((len(rhos), len(deltas)), np.nan)
                for (delta, rho), values in groups.items():
                    if delta in deltas and rho in rhos:
                        matrix[rhos.index(rho), deltas.index(delta)] = np.mean(values)
                axis = axes[row_index, column]
                xs, ys, surface = phase_surface(deltas, rhos, matrix)
                if min(len(xs), len(ys)) >= 2:
                    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
                    mesh = axis.imshow(
                        np.ma.masked_invalid(surface),
                        origin="lower",
                        extent=(
                            xs[0] - dx / 2,
                            xs[-1] + dx / 2,
                            ys[0] - dy / 2,
                            ys[-1] + dy / 2,
                        ),
                        aspect="auto",
                        interpolation="nearest",
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                    )
                else:
                    mesh = axis.pcolormesh(
                        xs,
                        ys,
                        surface,
                        shading="nearest",
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        rasterized=True,
                    )
                finite = surface[np.isfinite(surface)]
                if len(finite) and np.min(finite) < 0.5 < np.max(finite):
                    axis.contour(
                        xs,
                        ys,
                        surface,
                        levels=[0.5],
                        colors="white",
                        linewidths=2.4,
                    )
                if not sparse:
                    delta_curve, rho_curve = dt_curve()
                    axis.plot(delta_curve, rho_curve, color="black", lw=2.2)
                axis.set(
                    xlim=(min(deltas), max(deltas)),
                    ylim=(min(rhos), max(rhos)),
                    xlabel=r"Query fraction $Q/N$" if row_index else "",
                )
                if column == 0:
                    axis.set_ylabel(
                        LABELS[method] + "\n" + r"Sparsity fraction $k/Q$", labelpad=12
                    )
                axis.set_xticks([0.01, 0.04, 0.08] if sparse else [0.2, 0.5, 0.8])
                axis.set_yticks(
                    ([0.1, 0.2, 0.3] if max(rhos) > 0.15 else [0.05, 0.1, 0.15])
                    if sparse
                    else [0.2, 0.5, 0.8]
                )
                if column > 0:
                    axis.tick_params(labelleft=False)
                if row_index == 0:
                    axis.set_title(f"N = {2**size:,}\n{distribution.title()}")
        figure.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            label="Recovery probability",
            shrink=0.9,
            pad=0.02,
        )
        save_figure(figure, directory, filename)


def comparison_style():
    return plt.rc_context(
        {
            **sns.axes_style("darkgrid"),
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def comparison_legend(
    figure,
    axes,
    directory,
    name,
    methods=COMPARISON_METHODS,
    headings=(),
    extra_handles=(),
):
    methods = sorted(
        methods, key=lambda method: LABELS[method].startswith("SPHT (Ours)")
    )
    handles = [
        Line2D([0], [0], color=COLORS[method], lw=2.4, marker="o", label=LABELS[method])
        for method in methods
    ]
    if extra_handles:
        handles[-1:-1] = extra_handles
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.9 if headings else 1), pad=1.0, w_pad=1.0)
    for first, last, text in headings:
        center = (axes[first].get_position().x0 + axes[last].get_position().x1) / 2
        figure.text(center, 0.97, text, ha="center", va="top", fontsize=14)
    save_figure(figure, directory, name)


def matched_successes(results):
    groups = defaultdict(dict)
    for row in results:
        job = row["job"]
        groups[canonical([job["spec"], job["seed"], job["trial"]])][job["method"]] = row
    selected = []
    for pair in groups.values():
        other = "prefix_bch" if "prefix_bch" in pair else "prefix"
        if (
            "cs" in pair
            and other in pair
            and pair["cs"].get("success")
            and pair[other].get("success")
        ):
            selected.extend([pair["cs"], pair[other]])
    return selected


def plot_robust_comparison(results, directory):
    methods = ("cs", "prefix_robust", "stobbe", "swht")
    palette = COMPARISON_PALETTE
    with comparison_style():
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        groups = defaultdict(list)
        for record in results:
            if record.get("error_over_E", 0) > 0:
                job = record["job"]
                groups[
                    (job["method"], job["spec"]["sparsity"], job["spec"]["tail_ratio"])
                ].append(record["error_over_E"])
        intervals = [np.quantile(values, [0.25, 0.75]) for values in groups.values()]
        limits = (
            (
                min(lo for lo, hi in intervals) / 1.08,
                max(hi for lo, hi in intervals) * 1.08,
            )
            if intervals
            else (0.1, 1)
        )
        for column, sparsity in enumerate([8, 16, 32]):
            axis = axes[column]
            for method in methods:
                rows = [
                    r
                    for r in results
                    if r["job"]["method"] == method
                    and r["job"]["spec"]["sparsity"] == sparsity
                ]
                curve(
                    axis,
                    rows,
                    lambda spec: spec["tail_ratio"],
                    "error_over_E",
                    method,
                    color=palette[method],
                )
            axis.set(
                xlabel="Tail / head norm ratio",
                ylabel="Recovery error / E" if column == 0 else "",
                xscale="log",
                yscale="log",
                ylim=limits,
                title=f"k = {sparsity}",
            )
            if limits[1] / limits[0] >= 10:
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                axis.yaxis.set_major_formatter(LogFormatterMathtext())
            else:
                axis.yaxis.set_major_locator(
                    LogLocator(base=10, subs=(2, 3, 4, 5, 6, 8))
                )
                axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(NullFormatter())
        comparison_legend(figure, axes, directory, "robustness_error", methods)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for method in methods:
            rows = [r for r in results if r["job"]["method"] == method]
            assessed = [
                {**r, "within_bound": r["error_over_E"] <= 4}
                for r in rows
                if "error_over_E" in r
            ]
            curve(
                axes[0],
                assessed,
                lambda spec: spec["tail_ratio"],
                "within_bound",
                method,
                probability=True,
                color=palette[method],
            )
            curve(
                axes[1],
                rows,
                lambda spec: spec["sparsity"],
                "queries",
                method,
                color=palette[method],
            )
            curve(
                axes[2],
                rows,
                lambda spec: spec["sparsity"],
                "algorithm_seconds",
                method,
                color=palette[method],
            )
        axes[0].set(
            xlabel="Tail / head norm ratio",
            ylabel="Fraction with error ≤ 4E",
            xscale="log",
            ylim=(0, 1.03),
            title="Recovery within 4E",
        )
        axes[1].set(
            xlabel="k", ylabel="Oracle queries", yscale="log", title="Sampling cost"
        )
        axes[2].set(
            xlabel="k", ylabel="Runtime (s)", yscale="log", title="Computational cost"
        )
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
        indices = np.maximum(
            0, np.searchsorted(trace[:, column], grid, side="right") - 1
        )
        values.append(trace[indices, 1])
    array = np.asarray(values)
    color = COLORS[method]
    axis.plot(grid, np.median(array, axis=0), color=color, label=LABELS[method])
    if len(rows) > 1:
        axis.fill_between(
            grid,
            np.quantile(array, 0.25, axis=0),
            np.quantile(array, 0.75, axis=0),
            color=color,
            alpha=0.1,
        )


def plot_hpo_comparison(results, directory):
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    methods = ["random", "gp_ei", "harmonica", "prefix_bch"]
    results = [r for r in results if r["status"] == "ok"]
    budgets = [
        r["job"]["spec"]["budget"]
        for r in results
        if r["job"]["spec"]["task"] == "digits"
    ]
    maximum_budget = max(budgets, default=320)
    for method in methods:
        digits = [
            r
            for r in results
            if r["job"]["method"] == method
            and r["job"]["spec"]["task"] == "digits"
            and r["job"]["spec"]["budget"] == maximum_budget
        ]
        base = [r for r in digits if r["job"]["spec"]["dimension"] == 12]
        plot_trace(axes[0, 0], base, method)
        plot_trace(axes[0, 1], base, method, True)
        curve(axes[0, 2], digits, lambda spec: spec["dimension"], "test_error", method)
        for column, task in enumerate(["sparse", "hierarchical"]):
            for index, noise in enumerate([0, 1]):
                rows = [
                    r
                    for r in results
                    if r["job"]["method"] == method
                    and r["job"]["spec"]["task"] == task
                    and r["job"]["spec"].get("noise", 0) == noise
                ]
                curve(
                    axes[1, column],
                    rows,
                    lambda spec: spec["dimension"],
                    "best_validation",
                    method,
                    f"{LABELS[method]}, noise={noise}",
                    STYLES[index],
                )
        overhead = [
            {
                **r,
                "overhead": r.get("fit_seconds", 0) + r.get("optimization_seconds", 0),
            }
            for r in digits
        ]
        curve(axes[1, 2], overhead, lambda spec: spec["dimension"], "overhead", method)
    axes[0, 0].set(
        xlabel="Unique objective evaluations",
        ylabel="Best validation error",
        title="Digits: 12 effective variables",
    )
    axes[0, 1].set(
        xlabel="Cold-oracle cost + optimizer overhead (s)",
        ylabel="Best validation error",
        title="Digits: quality versus total cost",
    )
    axes[0, 2].set(
        xlabel="Binary variables including dummies",
        ylabel="Final test error",
        title="Digits: selected models on held-out test data",
    )
    axes[1, 0].set(
        xlabel="Binary variables", ylabel="Final objective", title="Sparse polynomial"
    )
    axes[1, 1].set(
        xlabel="Binary variables",
        ylabel="Final objective",
        title="Hierarchical objective",
    )
    axes[1, 2].set(
        xlabel="Binary variables including dummies",
        ylabel="Optimizer overhead (s)",
        yscale="log",
        title="Digits: computational scaling",
    )
    finish(
        figure,
        axes,
        "Hyperparameter optimization: quality, total cost, and optimizer overhead",
        directory,
        "hpo",
    )


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
            if not axis.lines:
                axis.text(
                    0.5,
                    0.5,
                    "Pending measurements",
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
        for position, kind in enumerate(["graph_edges", "graph_vertices"]):
            if not any(row["job"]["spec"]["kind"] == kind for row in common):
                axis.text(
                    0.5,
                    position,
                    "Pending measurements",
                    ha="center",
                    va="center",
                    fontsize=11,
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


def render(output, suite, config, implementation=None):
    directory = Path(output) / suite
    store = Store(directory / "cache.sqlite")
    metadata = store.get("__run__") or {}
    if not metadata.get("complete"):
        store.close()
        return
    rows = [
        row
        for _, row in store.iter_records("result")
        if implementation is None or row.get("implementation") == implementation
    ]
    store.close()
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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures = Path(output) / "figures"
    if suite == "graphs":
        graph_plots(rows, figures, CONFIG["evaluation"])
    elif suite == "scaling":
        scaling_plots(rows, figures, CONFIG["evaluation"])
    elif suite == "phase":
        plot_phase_comparisons(rows, config, figures)
    elif suite == "robust":
        plot_robust_comparison(rows, figures)
    else:
        plot_hpo_comparison(rows, figures)
    write_csv(directory / "summary.csv", aggregate(rows))
    if suite in {"graphs", "scaling"}:
        csv_file(
            directory / "paired-comparisons.csv",
            paired_effects(rows, CONFIG["evaluation"]),
        )
        coverage = []
        groups = defaultdict(list)
        for row in rows:
            groups[canonical([row["job"]["spec"], row["job"]["method"]])].append(row)
        paired = matched_successes(rows)
        pair_counts = Counter(canonical(row["job"]["spec"]) for row in paired)
        for values in groups.values():
            spec = values[0]["job"]["spec"]
            total = len(values)
            successes = sum(bool(row["success"]) for row in values)
            low, high = wilson(successes, total)
            coverage.append(
                {
                    **spec,
                    "method": values[0]["job"]["method"],
                    "completed": total,
                    "successes": successes,
                    "recovery": successes / total,
                    "recovery_low": low,
                    "recovery_high": high,
                    "jointly_successful_pairs": pair_counts[canonical(spec)] // 2,
                    "timeouts": sum(row["status"] == "timeout" for row in values),
                    "resource_limits": sum(
                        row["status"] == "resource_limit" for row in values
                    ),
                }
            )
        csv_file(directory / "coverage.csv", coverage)
