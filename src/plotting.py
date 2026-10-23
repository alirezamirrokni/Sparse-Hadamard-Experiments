import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .storage import Store, atomic_text, canonical


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        destination = directory / f"{name}.{extension}"
        temporary = destination.with_name(destination.stem + ".tmp." + extension)
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


COLORS = {
    "cs": "#222222",
    "prefix": "#2478b5",
    "prefix_bch": "#2478b5",
    "prefix_robust": "#2478b5",
    "swht": "#e69f00",
    "stobbe": "#c44e52",
    "random": "#777777",
    "gp_ei": "#9467bd",
    "harmonica": "#e69f00",
}
LABELS = {
    "cs": "CS",
    "prefix": "Proposed prefix",
    "prefix_bch": "Proposed (BCH)",
    "prefix_robust": "Proposed robust",
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
                axis.legend(fontsize=7, frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                "Awaiting completed trials",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#777777",
            )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, directory, name)


def curve(axis, rows, xvalue, metric, method, label=None, style="-", probability=False):
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
    color = COLORS[method]
    axis.plot(
        xs,
        centers,
        style,
        marker="o",
        ms=3,
        lw=1.4,
        color=color,
        label=label or LABELS[method],
    )
    if any(len(groups[x]) > 1 for x in xs):
        axis.fill_between(xs, lower, upper, color=color, alpha=0.08)


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
            figsize=(4.2 * len(columns), 7.1),
            squeeze=False,
            layout="constrained",
        )
        cmap = plt.get_cmap("jet_r").copy()
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
                mesh = axis.pcolormesh(
                    deltas,
                    rhos,
                    matrix,
                    shading="nearest",
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                finite = matrix[np.isfinite(matrix)]
                if len(finite) and np.min(finite) < 0.5 < np.max(finite):
                    axis.contour(
                        deltas,
                        rhos,
                        matrix,
                        levels=[0.5],
                        colors="white",
                        linewidths=1.1,
                    )
                if not sparse:
                    delta_curve, rho_curve = dt_curve()
                    axis.plot(delta_curve, rho_curve, color="black", lw=1)
                axis.set(
                    xlim=(min(deltas), max(deltas)),
                    ylim=(min(rhos), max(rhos)),
                    xlabel=r"Query fraction $Q/N$",
                )
                if column == 0:
                    axis.set_ylabel(LABELS[method] + "\n" + r"Sparsity fraction $k/Q$")
                if row_index == 0:
                    axis.set_title(
                        f"N = {2**size:,}\n{distribution.title()}", fontsize=11
                    )
        figure.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            label="Recovery probability",
            shrink=0.9,
            pad=0.02,
        )
        figure.suptitle(
            "CS and prefix recovery: "
            + ("sparse measurement regime" if sparse else "phase diagrams"),
            fontsize=14,
        )
        figure.supxlabel(
            "White: 50% recovery contour. Gray: not yet assessed."
            + ("" if sparse else " Black: Gaussian asymptotic reference."),
            fontsize=8,
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


def plot_graph_comparison(results, summaries, directory):
    figure, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    selected = graph_operating_points(summaries)
    methods = ["swht", "stobbe", "prefix_bch"]
    settings = [
        (
            axes[0, 0],
            "graph_edges",
            "edges",
            "algorithm_seconds_median",
            "Edges",
            "Runtime (s)",
        ),
        (
            axes[0, 1],
            "graph_edges",
            "edges",
            "queries_median",
            "Edges",
            "Oracle queries",
        ),
        (
            axes[0, 2],
            "graph_vertices",
            "dimension",
            "algorithm_seconds_median",
            "Vertices",
            "Runtime (s)",
        ),
        (
            axes[0, 3],
            "graph_vertices",
            "dimension",
            "queries_median",
            "Vertices",
            "Oracle queries",
        ),
        (
            axes[1, 1],
            "graph_exploratory_edges",
            "edges",
            "algorithm_seconds_median",
            "Edges",
            "Runtime (s)",
        ),
        (
            axes[1, 2],
            "graph_exploratory_vertices",
            "dimension",
            "algorithm_seconds_median",
            "Vertices",
            "Runtime (s)",
        ),
    ]
    for axis, kind, xkey, metric, xlabel, ylabel in settings:
        for method in methods:
            groups = defaultdict(list)
            for row in selected:
                if row["kind"] == kind and row["method"] == method and metric in row:
                    groups[row[xkey]].append(row[metric])
            if groups:
                xs = sorted(groups)
                axis.plot(
                    xs,
                    [np.median(groups[x]) for x in xs],
                    "-o",
                    ms=3,
                    color=COLORS[method],
                    label=LABELS[method],
                )
        axis.set(
            xlabel=xlabel,
            ylabel=ylabel,
            yscale="log",
            title=kind.replace("graph_", "").replace("_", " ").title(),
        )
    axis = axes[1, 0]
    for method in methods:
        rows = [
            r
            for r in results
            if r["job"]["spec"]["kind"] == "graph_reconstruction"
            and r["job"]["method"] == method
            and "edge_error" in r
        ]
        if rows:
            axis.scatter(
                [r["queries"] for r in rows],
                [r["edge_error"] for r in rows],
                s=12,
                alpha=0.45,
                color=COLORS[method],
                label=LABELS[method],
            )
    axis.set(
        xlabel="Oracle queries",
        ylabel="Symmetric edge error / true edges",
        xscale="log",
        title="Reconstruction: all tested settings",
    )
    axis = axes[1, 3]
    kinds = [
        "graph_edges",
        "graph_vertices",
        "graph_reconstruction",
        "graph_exploratory_edges",
        "graph_exploratory_vertices",
    ]
    for index, method in enumerate(methods):
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
            np.arange(len(kinds)) + (index - 1) * 0.23,
            values,
            height=0.22,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.set(
        yticks=np.arange(5),
        yticklabels=[
            "Edges",
            "Vertices",
            "Reconstruction",
            "Expl. edges",
            "Expl. vertices",
        ],
        xlim=(0, 1),
        xlabel="Recovery fraction",
        title="All settings, before operating-point selection",
    )
    finish(
        figure,
        axes,
        "Graph recovery: runtime, sampling, and reconstruction",
        directory,
        "graphs",
    )


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
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
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
        0.03,
        f"Resource/configuration-limited trials: {censored}",
        transform=axes[1, 2].transAxes,
        fontsize=8,
    )
    finish(
        figure,
        axes,
        "CS versus prefix recovery: scaling and recovery quality",
        directory,
        "scaling",
    )


def plot_robust_comparison(results, directory):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for column, sparsity in enumerate([8, 16, 32]):
        for method in ["cs", "prefix_robust"]:
            rows = [
                r
                for r in results
                if r["job"]["method"] == method
                and r["job"]["spec"]["sparsity"] == sparsity
            ]
            curve(
                axes[0, column],
                rows,
                lambda spec: spec["tail_ratio"],
                "error_over_E",
                method,
            )
        axes[0, column].axhline(4, color="#777777", ls="--", lw=1, label="4E target")
        axes[0, column].set(
            xlabel="Tail / head norm ratio",
            ylabel="Recovery error / E",
            xscale="log",
            yscale="log",
            title=f"k = {sparsity}",
        )
    for method in ["cs", "prefix_robust"]:
        rows = [r for r in results if r["job"]["method"] == method]
        assessed = [
            {**r, "within_bound": r["error_over_E"] <= 4}
            for r in rows
            if "error_over_E" in r
        ]
        curve(
            axes[1, 0],
            assessed,
            lambda spec: spec["tail_ratio"],
            "within_bound",
            method,
            probability=True,
        )
        curve(axes[1, 1], rows, lambda spec: spec["sparsity"], "queries", method)
        curve(
            axes[1, 2], rows, lambda spec: spec["sparsity"], "algorithm_seconds", method
        )
    axes[1, 0].set(
        xlabel="Tail / head norm ratio",
        ylabel="Fraction with error ≤ 4E",
        xscale="log",
        ylim=(-0.03, 1.03),
        title="All sparsities and distributions",
    )
    axes[1, 1].set(
        xlabel="k", ylabel="Oracle queries", yscale="log", title="Sampling cost"
    )
    axes[1, 2].set(
        xlabel="k", ylabel="Runtime (s)", yscale="log", title="Computational cost"
    )
    finish(
        figure,
        axes,
        "Approximate recovery: quality and cost with a spectral tail",
        directory,
        "robustness",
    )


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
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
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
            "font.size": 9,
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
