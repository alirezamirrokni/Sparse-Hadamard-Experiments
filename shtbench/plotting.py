import csv
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .storage import Store, atomic_text, canonical, digest


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
        successes = sum(bool(row.get("success", False)) for row in eligible)
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


def plot_phase(results, summaries, directory):
    for kind in ["reference", "phase"]:
        sizes = sorted({row["size"] for row in summaries if row["kind"] == kind})
        for size in sizes:
            distributions = sorted(
                {
                    row["distribution"]
                    for row in summaries
                    if row["kind"] == kind and row["size"] == size
                }
            )
            for distribution in distributions:
                cells = [
                    row
                    for row in summaries
                    if row["kind"] == kind
                    and row["size"] == size
                    and row["distribution"] == distribution
                ]
                methods = sorted({row["method"] for row in cells})
                deltas = sorted({row["delta"] for row in cells})
                rhos = sorted({row["rho"] for row in cells})
                matrices = {}
                for method in methods:
                    selected = [row for row in cells if row["method"] == method]
                    matrix = np.full((len(rhos), len(deltas)), np.nan)
                    for row in selected:
                        matrix[rhos.index(row["rho"]), deltas.index(row["delta"])] = (
                            row["success_rate"]
                        )
                    matrices[method] = matrix
                    figure, axis = plt.subplots(figsize=(6.2, 4.6))
                    mesh = axis.pcolormesh(
                        deltas,
                        rhos,
                        matrix,
                        shading="nearest",
                        cmap="jet_r",
                        vmin=0,
                        vmax=1,
                        rasterized=True,
                    )
                    delta_curve, rho_curve = dt_curve()
                    axis.plot(
                        delta_curve,
                        rho_curve,
                        color="black",
                        lw=1.4,
                        label="Gaussian asymptotic reference",
                    )
                    if (
                        len(deltas) > 1
                        and len(rhos) > 1
                        and np.nanmin(matrix) < 0.5 < np.nanmax(matrix)
                    ):
                        axis.contour(
                            deltas,
                            rhos,
                            matrix,
                            levels=[0.5],
                            colors="white",
                            linewidths=1.4,
                        )
                    axis.set(
                        xlabel=r"$\delta=Q/N$",
                        ylabel=r"$\rho=k/Q$",
                        xlim=(0, 1),
                        ylim=(0, 1),
                        title=f"{method}, N={size if kind == 'reference' else 2**size}, {distribution}",
                    )
                    axis.legend(loc="upper left", fontsize=7)
                    figure.colorbar(mesh, ax=axis, label="Recovery probability")
                    save_figure(
                        figure, directory, f"{kind}_{size}_{distribution}_{method}"
                    )
                if "cs" in matrices:
                    for method in methods:
                        if not method.startswith("prefix"):
                            continue
                        difference = matrices[method] - matrices["cs"]
                        figure, axes = plt.subplots(
                            1, 3, figsize=(15, 4), constrained_layout=True
                        )
                        for axis, name in zip(axes[:2], ["cs", method]):
                            mesh = axis.pcolormesh(
                                deltas,
                                rhos,
                                matrices[name],
                                shading="nearest",
                                cmap="jet_r",
                                vmin=0,
                                vmax=1,
                                rasterized=True,
                            )
                            axis.set(
                                title=name,
                                xlabel=r"$Q/N$",
                                ylabel=r"$k/Q$",
                                xlim=(0, 1),
                                ylim=(0, 1),
                            )
                        figure.colorbar(mesh, ax=axes[:2], label="Recovery probability")
                        mesh = axes[2].pcolormesh(
                            deltas,
                            rhos,
                            difference,
                            shading="nearest",
                            cmap="RdBu",
                            vmin=-1,
                            vmax=1,
                            rasterized=True,
                        )
                        axes[2].set(
                            title=f"{method} − cs",
                            xlabel=r"$Q/N$",
                            ylabel=r"$k/Q$",
                            xlim=(0, 1),
                            ylim=(0, 1),
                        )
                        figure.colorbar(
                            mesh, ax=axes[2], label="Probability difference"
                        )
                        save_figure(
                            figure,
                            directory,
                            f"phase_comparison_{size}_{distribution}_{method}",
                        )
                if len(deltas) > 1 and len(rhos) > 1:
                    figure, axis = plt.subplots(figsize=(6.2, 4.6))
                    for method, matrix in matrices.items():
                        if np.nanmin(matrix) < 0.5 < np.nanmax(matrix):
                            contour = axis.contour(deltas, rhos, matrix, levels=[0.5])
                            axis.plot(
                                [], [], label=method, color=contour.get_edgecolor()[0]
                            )
                    axis.plot(*dt_curve(), "k--", label="Gaussian asymptotic reference")
                    axis.set(xlabel=r"$Q/N$", ylabel=r"$k/Q$", xlim=(0, 1), ylim=(0, 1))
                    axis.legend(fontsize=8)
                    save_figure(
                        figure, directory, f"universality_{kind}_{size}_{distribution}"
                    )
    pairs = defaultdict(dict)
    for row in results:
        if row["job"]["spec"]["kind"] == "phase":
            pairs[canonical([row["job"]["spec"], row["job"]["trial"]])][
                row["job"]["method"]
            ] = row
    paired = []
    for key, pair in pairs.items():
        baseline = pair.get("cs")
        if baseline is None:
            continue
        for method, value in pair.items():
            if method.startswith("prefix"):
                comparable = baseline.get("success") and value.get("success")
                paired.append(
                    {
                        **value["job"]["spec"],
                        "trial": value["job"]["trial"],
                        "method": method,
                        "both_successful": bool(comparable),
                        "baseline_success": bool(baseline.get("success")),
                        "method_success": bool(value.get("success")),
                        "speedup": baseline.get("algorithm_seconds", 0)
                        / max(value.get("algorithm_seconds", 1), 1e-12)
                        if comparable
                        else None,
                        "prefix_rounds": value.get("rounds"),
                        "method_queries": value.get("queries"),
                        "baseline_queries": baseline.get("queries"),
                    }
                )
    write_csv(Path(directory).parent / "paired_phase.csv", paired)
    for method in sorted({row["method"] for row in paired}):
        data = [
            row for row in paired if row["method"] == method and row["both_successful"]
        ]
        if not data:
            continue
        figure, axis = plt.subplots(figsize=(6.2, 4.6))
        for rounds in [1, 2]:
            selected = [
                row for row in data if (row["prefix_rounds"] == 1) == (rounds == 1)
            ]
            axis.scatter(
                [row["rho"] for row in selected],
                [row["speedup"] for row in selected],
                s=12,
                alpha=0.6,
                label="One round" if rounds == 1 else "Multiple rounds",
            )
        axis.axhline(1, color="black", lw=1)
        axis.set(
            xlabel=r"$k/Q$",
            ylabel="CS time / prefix time",
            yscale="log",
            title="Paired trials where both methods recover",
        )
        axis.legend()
        save_figure(figure, directory, f"phase_speedup_{method}")


def plot_graphs(summaries, directory, target=0.9):
    groups = defaultdict(list)
    for row in summaries:
        if row["kind"].startswith("graph"):
            groups[(row["kind"], row["snapshot"])].append(row)
    operating_points = []
    for (kind, snapshot), rows in groups.items():
        if "reconstruction" in kind:
            figure, axis = plt.subplots(figsize=(6.2, 4.6))
            for method in sorted({row["method"] for row in rows}):
                points = [
                    row
                    for row in rows
                    if row["method"] == method and "edge_error" in row
                ]
                axis.scatter(
                    [row["queries"] for row in points],
                    [row["edge_error"] for row in points],
                    label=method,
                    s=18,
                )
            axis.set(
                xlabel="Unique oracle evaluations",
                ylabel="Symmetric edge difference / true edges",
                title=f"Snapshots {snapshot} and {snapshot + 2}",
            )
            axis.legend()
            save_figure(figure, directory, f"{kind}_{snapshot}")
            continue
        xname = "dimension" if "vertices" in kind else "edges"
        selected = []
        for method in sorted({row["method"] for row in rows}):
            for x in sorted({row[xname] for row in rows}):
                eligible = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row[xname] == x
                    and row["success_rate"] is not None
                    and row["success_rate"] >= target
                    and "algorithm_seconds" in row
                ]
                if eligible:
                    best = min(eligible, key=lambda row: row["algorithm_seconds"])
                    selected.append(best)
                    operating_points.append(
                        {
                            **best,
                            "selection": "descriptive sweep operating point, not a held-out estimate",
                        }
                    )
        for metric, label in [
            ("algorithm_seconds", "Mean recovery time (s)"),
            ("queries", "Mean unique oracle evaluations"),
        ]:
            figure, axis = plt.subplots(figsize=(6.2, 4.6))
            for method in sorted({row["method"] for row in selected}):
                points = sorted(
                    [row for row in selected if row["method"] == method],
                    key=lambda row: row[xname],
                )
                axis.errorbar(
                    [row[xname] for row in points],
                    [row[metric] for row in points],
                    yerr=[
                        row.get(metric + "_std", 0) / np.sqrt(row["trials"])
                        for row in points
                    ],
                    marker="o",
                    markersize=3,
                    capsize=2,
                    label=method,
                )
            axis.set(
                xlabel="Vertices" if xname == "dimension" else "Edges",
                ylabel=label,
                title=f"Snapshots {snapshot} and {snapshot + 2}; observed success ≥ {target:.0%}",
            )
            if selected:
                axis.legend()
            save_figure(figure, directory, f"{kind}_{snapshot}_{metric}")
        figure, axis = plt.subplots(figsize=(6.2, 4.6))
        for method in sorted({row["method"] for row in rows}):
            points = [
                row
                for row in rows
                if row["method"] == method and "algorithm_seconds" in row
            ]
            axis.scatter(
                [row["queries"] for row in points],
                [row["algorithm_seconds"] for row in points],
                s=15,
                alpha=0.6,
                label=method,
            )
        axis.set(xlabel="Unique oracle evaluations", ylabel="Recovery time (s)")
        axis.legend()
        save_figure(figure, directory, f"{kind}_{snapshot}_time_samples")
    write_csv(Path(directory).parent / "graph_operating_points.csv", operating_points)


def plot_scaling(summaries, directory):
    for kind in ["scaling", "robust"]:
        rows = [row for row in summaries if row["kind"] == kind]
        if not rows:
            continue
        xname = "tail_ratio" if kind == "robust" else "dimension"
        for metric in (
            ["relative_error", "error_over_E", "queries"]
            if kind == "robust"
            else ["algorithm_seconds", "queries", "success_rate"]
        ):
            figure, axis = plt.subplots(figsize=(6.2, 4.6))
            groups = defaultdict(list)
            for row in rows:
                if metric in row:
                    group = canonical(
                        {
                            key: row.get(key)
                            for key in ["method", "degree", "sparsity", "tail_size"]
                        }
                    )
                    groups[group].append(row)
            for group, points in groups.items():
                points.sort(key=lambda row: row[xname])
                label = json.loads(group)
                axis.plot(
                    [row[xname] for row in points],
                    [row[metric] for row in points],
                    marker="o",
                    ms=3,
                    label=f"{label['method']}, k={label['sparsity']}, d={label['degree']}",
                )
            axis.set(xlabel=xname.replace("_", " "), ylabel=metric.replace("_", " "))
            if metric in {"algorithm_seconds", "queries", "relative_error"}:
                axis.set_yscale("log")
            axis.legend(fontsize=7)
            save_figure(figure, directory, f"{kind}_{metric}")


def plot_hpo(results, directory):
    groups = defaultdict(list)
    for row in results:
        if row["job"]["spec"]["kind"] == "hpo" and row["status"] == "ok":
            groups[canonical(row["job"]["spec"])].append(row)
    for group, rows in groups.items():
        spec = json.loads(group)
        name = f"hpo_{spec['task']}_{spec['dimension']}_{spec['budget']}_{digest(spec)[:6]}"
        for axis_type in ["evaluations", "time"]:
            figure, axis = plt.subplots(figsize=(6.2, 4.6))
            for method in sorted({row["job"]["method"] for row in rows}):
                selected = [
                    row
                    for row in rows
                    if row["job"]["method"] == method and row.get("trace")
                ]
                if not selected:
                    continue
                upper = min(
                    row["trace"][-1][0 if axis_type == "evaluations" else 2]
                    for row in selected
                )
                grid = np.linspace(1 if axis_type == "evaluations" else 0, upper, 150)
                curves = []
                for row in selected:
                    trace = np.asarray(row["trace"])
                    x = trace[:, 0 if axis_type == "evaluations" else 2]
                    indices = np.searchsorted(x, grid, side="right") - 1
                    curve = trace[np.maximum(indices, 0), 1]
                    curve[indices < 0] = np.nan
                    curves.append(curve)
                curves = np.asarray(curves)
                counts = np.sum(np.isfinite(curves), axis=0)
                mean = np.divide(
                    np.nansum(curves, axis=0),
                    counts,
                    out=np.full(len(grid), np.nan),
                    where=counts > 0,
                )
                variance = np.divide(
                    np.nansum((curves - mean) ** 2, axis=0),
                    counts,
                    out=np.zeros(len(grid)),
                    where=counts > 0,
                )
                error = np.sqrt(variance) / np.sqrt(np.maximum(counts, 1))
                axis.step(grid, mean, where="post", label=method)
                axis.fill_between(
                    grid,
                    mean - 1.96 * error,
                    mean + 1.96 * error,
                    alpha=0.15,
                    step="post",
                )
            axis.set(
                xlabel="Unique objective evaluations"
                if axis_type == "evaluations"
                else "Cold oracle cost + measured optimizer overhead (s)",
                ylabel="Best validation objective",
                title=f"{spec['task']}, {spec['dimension']} binary variables",
            )
            axis.legend(fontsize=8)
            save_figure(figure, directory, f"{name}_{axis_type}")
        figure, axis = plt.subplots(figsize=(6.2, 4.6))
        methods = sorted({row["job"]["method"] for row in rows})
        bottoms = np.zeros(len(methods))
        for metric, label in [
            ("oracle_seconds", "Objective evaluations"),
            ("fit_seconds", "Spectrum fitting"),
            ("optimization_seconds", "Surrogate optimization"),
        ]:
            values = [
                np.mean(
                    [
                        row.get(metric, 0)
                        for row in rows
                        if row["job"]["method"] == method
                    ]
                )
                for method in methods
            ]
            axis.bar(methods, values, bottom=bottoms, label=label)
            bottoms += values
        axis.set(ylabel="Seconds", title="Measured time decomposition")
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=8)
        save_figure(figure, directory, f"{name}_time_breakdown")
    if groups:
        figure, axis = plt.subplots(figsize=(6.2, 4.6))
        scaling = defaultdict(list)
        for rows in groups.values():
            for row in rows:
                scaling[(row["job"]["method"], row["job"]["spec"]["task"])].append(row)
        for (method, task), rows in scaling.items():
            dimensions = sorted({row["job"]["spec"]["dimension"] for row in rows})
            values = [
                np.median(
                    [
                        row.get("fit_seconds", 0) + row.get("optimization_seconds", 0)
                        for row in rows
                        if row["job"]["spec"]["dimension"] == n
                    ]
                )
                for n in dimensions
            ]
            axis.plot(dimensions, values, "o-", label=f"{method}, {task}")
        axis.set(
            xlabel="Binary variables",
            ylabel="Median optimizer overhead (s)",
            yscale="log",
        )
        axis.legend(fontsize=7)
        save_figure(figure, directory, "hpo_optimizer_scaling")


def render(database, directory, implementation=None, timing_workers=None):
    store = Store(database)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"trials.{os.getpid()}.tmp"
    results = []
    specifications = {}
    configurations = {}
    with temporary.open("w") as stream:
        for key, value in store.iter_records("result"):
            if (
                implementation is not None
                and value.get("implementation") != implementation
            ):
                continue
            if (
                timing_workers is not None
                and value["job"]["settings"].get("timing_workers", 1) != timing_workers
            ):
                continue
            stream.write(canonical(value) + "\n")
            compact = {
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
            job = dict(value["job"])
            spec_key = canonical(job["spec"])
            settings_key = canonical(job["settings"])
            job["spec"] = specifications.setdefault(spec_key, job["spec"])
            job["settings"] = configurations.setdefault(settings_key, job["settings"])
            compact["job"] = job
            results.append(compact)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(directory / "trials.jsonl")
    store.close()
    summaries = aggregate(results)
    write_csv(directory / "summary.csv", summaries)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures = directory / "figures"
    plot_phase(results, summaries, figures)
    plot_graphs(summaries, figures)
    plot_scaling(summaries, figures)
    plot_hpo(results, figures)
    statuses = {
        status: sum(row["status"] == status for row in results)
        for status in sorted({row["status"] for row in results})
    }
    atomic_text(
        directory / "status.json",
        json.dumps({"completed_trials": len(results), "statuses": statuses}, indent=2),
    )
