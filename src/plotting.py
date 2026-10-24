import csv
import io
import json
import math
import os
import textwrap
from collections import defaultdict
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

from .storage import Store, atomic_text, canonical


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
                        rf"Sparsity fraction $k/Q$   {LABELS[method]}", labelpad=12
                    )
                axis.set_xticks([0.01, 0.04, 0.08] if sparse else [0.2, 0.5, 0.8])
                axis.set_yticks([0.05, 0.1, 0.15] if sparse else [0.2, 0.5, 0.8])
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


def graph_operating_points(summaries):
    groups = defaultdict(list)
    for row in summaries:
        if (
            row.get("success_rate") is not None
            and row["success_rate"] >= 0.9
            and "algorithm_seconds_median" in row
        ):
            groups[
                (
                    row["kind"],
                    row["snapshot"],
                    row["dimension"],
                    row["edges"],
                    row["method"],
                )
            ].append(row)
    return [
        min(rows, key=lambda row: row["algorithm_seconds_median"])
        for rows in groups.values()
    ]


def comparison_style():
    return plt.rc_context(
        {
            **sns.axes_style("darkgrid"),
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def comparison_legend(figure, axes, directory, name, methods=COMPARISON_METHODS):
    methods = sorted(
        methods, key=lambda method: LABELS[method].startswith("SPHT (Ours)")
    )
    handles = [
        Line2D([0], [0], color=COLORS[method], lw=2.4, marker="o", label=LABELS[method])
        for method in methods
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(methods),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1), pad=1.0, w_pad=1.0)
    save_figure(figure, directory, name)


def plot_graph_comparison(results, summaries, directory):
    selected = graph_operating_points(summaries)
    panels = [
        ("graph_edges", "edges", "algorithm_seconds_median", "Edges", "Runtime (s)"),
        ("graph_edges", "edges", "queries_median", "Edges", "Oracle queries"),
        (
            "graph_vertices",
            "dimension",
            "algorithm_seconds_median",
            "Vertices",
            "Runtime (s)",
        ),
        ("graph_vertices", "dimension", "queries_median", "Vertices", "Oracle queries"),
    ]
    with comparison_style():
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.8))
        for axis, (kind, xkey, metric, xlabel, ylabel) in zip(axes, panels):
            for method in COMPARISON_METHODS:
                groups = defaultdict(list)
                for row in selected:
                    if (
                        row["kind"] == kind
                        and row["method"] == method
                        and metric in row
                    ):
                        groups[row[xkey]].append(row[metric])
                if groups:
                    xs = sorted(groups)
                    axis.plot(
                        xs,
                        [np.median(groups[x]) for x in xs],
                        "-o",
                        ms=5,
                        color=COLORS[method],
                    )
            axis.set(xlabel=xlabel, ylabel=ylabel, yscale="log")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_minor_formatter(NullFormatter())
        comparison_legend(figure, axes, directory, "graphs")
        figure, axis = plt.subplots(figsize=(8, 4.2))
        kinds = ["graph_edges", "graph_vertices"]
        for index, method in enumerate(COMPARISON_METHODS):
            values = []
            for kind in kinds:
                rows = [
                    r
                    for r in results
                    if r["job"]["method"] == method
                    and r["job"]["spec"]["kind"] == kind
                    and r["status"] in ASSESSED
                ]
                values.append(np.mean([r["success"] for r in rows]) if rows else np.nan)
            axis.barh(
                np.arange(2) + (index - 1.5) * 0.19,
                values,
                height=0.18,
                color=COLORS[method],
            )
        axis.set(
            yticks=np.arange(2),
            yticklabels=["Edges", "Vertices"],
            xlim=(0, 1),
            xlabel="Exact recovery fraction",
        )
        comparison_legend(figure, [axis], directory, "graph_recovery")


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


def plot_scaling_comparison(results, directory):
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    paired = matched_successes(results)
    global_rows = [r for r in results if r["job"]["spec"].get("degree") is None]
    for method in ["cs", "prefix"]:
        for index, sparsity in enumerate([4, 8, 16, 32]):
            rows = [
                r
                for r in global_rows
                if r["job"]["method"] == method
                and r["job"]["spec"]["sparsity"] == sparsity
            ]
            timed = [
                r
                for r in paired
                if r["job"]["method"] == method
                and r["job"]["spec"].get("degree") is None
                and r["job"]["spec"]["sparsity"] == sparsity
            ]
            label = f"{LABELS[method]}, k={sparsity}"
            xvalue = lambda spec: 2 ** spec["dimension"]
            curve(
                axes[0, 0],
                timed,
                xvalue,
                "algorithm_seconds",
                method,
                label,
                STYLES[index],
            )
            curve(axes[0, 1], rows, xvalue, "queries", method, label, STYLES[index])
            curve(
                axes[0, 2], rows, xvalue, "success", method, label, STYLES[index], True
            )
    for column, degree in enumerate([2, 3]):
        for method in ["cs", "prefix_bch"]:
            for index, sparsity in enumerate([8, 16, 32]):
                rows = [
                    r
                    for r in paired
                    if r["job"]["method"] == method
                    and r["job"]["spec"].get("degree") == degree
                    and r["job"]["spec"]["sparsity"] == sparsity
                ]
                curve(
                    axes[1, column],
                    rows,
                    lambda spec: spec["dimension"],
                    "algorithm_seconds",
                    method,
                    f"{LABELS[method]}, k={sparsity}",
                    STYLES[index],
                )
        axes[1, column].set(
            xlabel="Boolean variables",
            ylabel="Runtime (s)",
            title=f"Degree {degree}: jointly successful instances",
            xscale="log",
            yscale="log",
        )
    for method in ["cs", "prefix_bch"]:
        for index, degree in enumerate([2, 3]):
            rows = [
                r
                for r in results
                if r["job"]["method"] == method
                and r["job"]["spec"].get("degree") == degree
            ]
            curve(
                axes[1, 2],
                rows,
                lambda spec: spec["dimension"],
                "success",
                method,
                f"{LABELS[method]}, d={degree}",
                STYLES[index],
                True,
            )
    axes[0, 0].set(
        xlabel="N",
        ylabel="Runtime (s)",
        title="Jointly successful instances",
        xscale="log",
        yscale="log",
    )
    axes[0, 1].set(
        xlabel="N",
        ylabel="Oracle queries",
        title="Unrestricted spectra",
        xscale="log",
        yscale="log",
    )
    axes[0, 2].set(
        xlabel="N",
        ylabel="Recovery probability",
        title="Unrestricted spectra",
        xscale="log",
        ylim=(-0.03, 1.03),
    )
    axes[1, 2].set(
        xlabel="Boolean variables",
        ylabel="Recovery probability",
        title="Low degree: mean across sparsities",
        xscale="log",
        ylim=(-0.03, 1.03),
    )
    censored = sum(r["status"] not in ASSESSED for r in results)
    axes[1, 2].text(
        0.02,
        0.86,
        f"Resource/configuration-limited\ntrials: {censored}",
        transform=axes[1, 2].transAxes,
        fontsize=11,
    )
    finish(
        figure,
        axes,
        "CS versus prefix recovery: scaling and recovery quality",
        directory,
        "scaling",
    )


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


def render(output, suite, config, implementation=None):
    output = Path(output)
    database = output / suite / "cache.sqlite"
    if not database.exists():
        return
    store = Store(database)
    results = []
    for key, value in store.iter_records("result"):
        if implementation is not None and value.get("implementation") != implementation:
            continue
        results.append(
            {
                name: content
                for name, content in value.items()
                if name
                not in {
                    "stages",
                    "frequencies",
                    "values",
                    "query_history",
                    "best_point",
                }
            }
        )
    store.close()
    summaries = aggregate(results)
    write_csv(output / suite / "summary.csv", summaries)
    status = {
        status: sum(r["status"] == status for r in results)
        for status in sorted({r["status"] for r in results})
    }
    atomic_text(
        output / suite / "status.json",
        json.dumps({"completed_trials": len(results), "statuses": status}, indent=2),
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.4,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures = output / "figures"
    if suite == "phase":
        plot_phase_comparisons(results, config, figures)
    elif suite == "graphs":
        write_csv(
            output / suite / "operating_points.csv", graph_operating_points(summaries)
        )
        plot_graph_comparison(results, summaries, figures)
    elif suite == "scaling":
        plot_scaling_comparison(results, figures)
    elif suite == "robust":
        plot_robust_comparison(results, figures)
    elif suite == "hpo":
        plot_hpo_comparison(results, figures)
