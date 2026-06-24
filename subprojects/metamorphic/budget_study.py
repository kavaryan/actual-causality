"""Increase N under a real wall-clock timeout and plot budgeted boxplots.

Each method-instance pair runs in an isolated process. A run is terminated at
T seconds. Timeout runs are kept in the CSV outputs but are dropped from the
boxplots. The study stops when all four method groups time out at one N.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from case_studies import make_av_problem, make_lift_problem
from search import (
    approximate_best_first,
    best_first_with_literal_subset_shrink,
    breadth_first_literal,
    nsga2_pareto,
)

ROOT = Path(__file__).resolve().parent
METHODS = ["bfs", "approx_best_first", "nsga2", "best_first_shrink"]
LABELS = {
    "bfs": "BFS",
    "approx_best_first": "Approx. best-first",
    "nsga2": "NSGA-II",
    "best_first_shrink": "Best-first + shrink",
}
COLORS = {
    "bfs": "#bdbdbd",
    "approx_best_first": "#9ecae1",
    "nsga2": "#fdae6b",
    "best_first_shrink": "#a1d99b",
}


def _worker(queue, case_study, n_vars, parameter_1, parameter_2,
            instance_seed, method, nsga_population, nsga_generations):
    try:
        rng = np.random.default_rng(instance_seed)
        problem = (
            make_lift_problem(n_vars, rng, parameter_1, parameter_2)
            if case_study == "lift"
            else make_av_problem(n_vars, rng, parameter_1, parameter_2)
        )
        started = time.perf_counter()
        if method == "bfs":
            result = breadth_first_literal(problem)
        elif method == "approx_best_first":
            result = approximate_best_first(problem)
        elif method == "nsga2":
            result = nsga2_pareto(
                problem, instance_seed, nsga_population, nsga_generations
            )
        elif method == "best_first_shrink":
            result = best_first_with_literal_subset_shrink(problem)
        else:
            raise ValueError(method)
        queue.put({
            "elapsed": time.perf_counter() - started,
            "success": result.solution is not None,
            "cardinality": (
                len(result.solution) if result.solution is not None else np.nan
            ),
            "robustness": result.robustness,
            "error": "",
        })
    except BaseException:
        queue.put({
            "elapsed": np.nan,
            "success": False,
            "cardinality": np.nan,
            "robustness": np.nan,
            "error": traceback.format_exc(),
        })


def run_with_timeout(*worker_args, budget: float) -> dict:
    # Fork keeps each two-second run isolated without paying Python/import
    # startup cost for every method-instance pair on Linux.
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(queue, *worker_args))
    started = time.perf_counter()
    process.start()
    process.join(budget)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "time": budget,
            "timeout": True,
            "success": False,
            "cardinality": np.nan,
            "robustness": np.nan,
            "error": "",
        }
    wall_time = time.perf_counter() - started
    if queue.empty():
        return {
            "time": min(wall_time, budget),
            "timeout": False,
            "success": False,
            "cardinality": np.nan,
            "robustness": np.nan,
            "error": f"worker exited with code {process.exitcode}",
        }
    result = queue.get()
    return {
        "time": min(float(result["elapsed"]), budget),
        "timeout": False,
        "success": bool(result["success"]),
        "cardinality": result["cardinality"],
        "robustness": result["robustness"],
        "error": result["error"],
    }


def run_budget_study(
    budget=2.0, start_n=5, step=5, max_n=500, trials=3, seed=7,
    nsga_population=60, nsga_generations=30, sizes=None,
) -> pd.DataFrame:
    rows = []
    subjects = [
        ("lift", speed_class, speed, density_class, density)
        for speed_class, speed in {"S1": 0.5, "S2": 1.0, "S3": 1.5}.items()
        for density_class, density in {"C1": 0.5, "C2": 1.0, "C3": 1.5}.items()
    ] + [
        ("av", speed_class, speed, distance_class, distance)
        for speed_class, speed in {"slow": 2.5, "fast": 7.5}.items()
        for distance_class, distance in {"short": 5.0, "long": 15.0}.items()
    ]
    n_values = sizes if sizes is not None else range(start_n, max_n + 1, step)
    for n_vars in n_values:
        for (
            case_study,
            parameter_1_class,
            parameter_1,
            parameter_2_class,
            parameter_2,
        ) in subjects:
            for trial in range(trials):
                instance_seed = (
                    seed + (100000 if case_study == "av" else 0)
                    + n_vars * 1000 + trial * 10
                    + int(parameter_1 * 13) + int(parameter_2 * 17)
                )
                for method in METHODS:
                    result = run_with_timeout(
                        case_study, n_vars, parameter_1, parameter_2,
                        instance_seed, method, nsga_population,
                        nsga_generations, budget=budget,
                    )
                    rows.append({
                        "case_study": case_study,
                        "n_vars": n_vars,
                        "trial": trial,
                        "parameter_1_class": parameter_1_class,
                        "parameter_1": parameter_1,
                        "parameter_2_class": parameter_2_class,
                        "parameter_2": parameter_2,
                        "method": method,
                        **result,
                    })
        frame = pd.DataFrame(rows)
        current = frame[frame.n_vars == n_vars]
        timeout_rates = current.groupby("method").timeout.mean()
        print(
            f"N={n_vars}: " + ", ".join(
                f"{method}={timeout_rates.get(method, 0):.0%}"
                for method in METHODS
            ),
            flush=True,
        )
        if all(timeout_rates.get(method, 0) > 0.5 for method in METHODS):
            break
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["n_vars", "method"], as_index=False).agg(
        mean_time=("time", "mean"),
        median_time=("time", "median"),
        median_cardinality=("cardinality", "median"),
        median_robustness=("robustness", "median"),
        timeout_rate=("timeout", "mean"),
        runs=("timeout", "size"),
    )


def perform_budget_anova(
    results: pd.DataFrame, completed_only: bool, budget: float
) -> pd.DataFrame:
    """Paper-style Type-II ANOVA on log wall-clock time.

    completed_only=True drops timeout rows. completed_only=False treats timeout
    rows as exactly the budget, which is a lower bound on true runtime.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    data = results.copy()
    policy = "completed_only" if completed_only else "timeout_as_budget"
    if completed_only:
        data = data[~data.timeout].copy()
    else:
        data["time"] = data["time"].fillna(budget).clip(upper=budget)
    data = data[data.time > 0].copy()
    data["log_time"] = np.log(data["time"])
    data = data[np.isfinite(data.log_time)].copy()

    tables = []
    for case_study in ("lift", "av"):
        for method in METHODS:
            current = data[
                (data.case_study == case_study)
                & (data.method == method)
            ].copy()
            if current.empty:
                continue
            # OLS needs observations beyond the fully saturated design. If a
            # timeout policy leaves too little data, emit a compact diagnostic
            # row instead of failing the whole budget run.
            factors = [
                "C(parameter_1_class)",
                "C(parameter_2_class)",
                "C(n_vars)",
            ]
            formula = "log_time ~ " + " * ".join(factors)
            try:
                model = ols(formula, data=current).fit()
                if model.df_resid <= 0:
                    raise ValueError(
                        "insufficient residual degrees of freedom; "
                        "increase --trials or reduce the ANOVA formula"
                    )
                table = anova_lm(model, typ=2).reset_index()
                table = table.rename(columns={"index": "factor"})
            except Exception as exc:
                table = pd.DataFrame([{
                    "factor": "ANOVA_FAILED",
                    "sum_sq": np.nan,
                    "df": np.nan,
                    "F": np.nan,
                    "PR(>F)": np.nan,
                    "error": str(exc),
                }])
            table.insert(0, "policy", policy)
            table.insert(1, "case_study", case_study)
            table.insert(2, "method", method)
            table.insert(3, "n_observations", len(current))
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_budget_results(
    results: pd.DataFrame,
    budget: float,
    output: Path,
    sizes=None,
    case_study: str | None = None,
    panels: tuple[str, ...] = ("time", "cardinality", "robustness"),
) -> None:
    if case_study is not None:
        results = results[results.case_study == case_study].copy()
        if results.empty:
            raise ValueError(f"No rows found for case_study={case_study!r}")
    plotted = results[(~results.timeout)].copy()
    timeout_rates = results.groupby(["method", "n_vars"]).timeout.mean()
    sizes = sorted(sizes if sizes is not None else results.n_vars.unique())
    x = np.arange(len(sizes))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))

    panel_specs = {
        "time": ("time", "Execution Time (seconds)", True),
        "cardinality": ("cardinality", "Found Set Cardinality", False),
        "robustness": ("robustness", "Robustness", False),
    }
    unknown_panels = sorted(set(panels) - set(panel_specs))
    if unknown_panels:
        raise ValueError(f"Unknown plot panel(s): {unknown_panels}")
    selected_panels = [panel_specs[panel] for panel in panels]
    figsize = (6.16, 4.56) if len(selected_panels) == 1 else (6.16, 8.0)
    fig, axes = plt.subplots(
        len(selected_panels), 1, figsize=figsize, sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, (column, ylabel, log_scale) in zip(axes, selected_panels):
        for offset, method in zip(offsets, METHODS):
            groups = [
                plotted[
                    (plotted.method == method)
                    & (plotted.n_vars == n_vars)
                    & plotted[column].notna()
                ][column].to_numpy()
                for n_vars in sizes
            ]
            kept = [
                (position, group)
                for position, n_vars, group in zip(x + offset, sizes, groups)
                if len(group) > 0
                and timeout_rates.get((method, n_vars), 0.0) <= 0.5
            ]
            if not kept:
                continue
            positions, values = zip(*kept)
            box = ax.boxplot(
                values,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "#333333", "linewidth": 1.1},
                boxprops={
                    "facecolor": COLORS[method],
                    "edgecolor": "#2f6db2",
                    "linewidth": 0.8,
                },
                whiskerprops={"color": "#2f6db2", "linewidth": 0.8},
                capprops={"color": "#2f6db2", "linewidth": 0.8},
            )
            if ax is axes[0]:
                box["boxes"][0].set_label(LABELS[method])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        if log_scale:
            ax.set_yscale("log")
            positive = plotted.loc[plotted[column] > 0, column]
            if not positive.empty:
                lower = max(positive.min() / 2, 1e-6)
                upper = max(budget * 1.4, positive.max() * 1.25)
                ax.set_ylim(lower, upper)
        ax.grid(True, axis="both", alpha=0.22, linestyle="--")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(sizes)
    axes[-1].set_xlabel("Number of Variables", fontsize=15)
    axes[0].legend(
        ncol=4, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.02), fontsize=7.5,
        columnspacing=0.9, handlelength=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=100)
    plt.close(fig)


def plot_times(
    results: pd.DataFrame,
    budget: float,
    output: Path,
    sizes=None,
    runtime_only: bool = False,
) -> None:
    panels = ("time",) if runtime_only else ("time", "cardinality", "robustness")
    plot_budget_results(results, budget, output, sizes=sizes, panels=panels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--start-n", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-n", type=int, default=500)
    parser.add_argument(
        "--sizes",
        type=str,
        default="10,20,50,100",
        help="Comma-separated N values to run and plot. Use empty string to use start/step/max.",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nsga-population", type=int, default=60)
    parser.add_argument("--nsga-generations", type=int, default=30)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Skip ANOVA generation; useful for quick plotting/runtime checks.",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help=(
            "Also write runtime-only versions of the budget plots. The default "
            "three-panel plots are still generated."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "budget_results"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sizes = (
        [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
        if args.sizes.strip()
        else None
    )
    results = run_budget_study(
        args.budget, args.start_n, args.step, args.max_n, args.trials,
        args.seed, args.nsga_population, args.nsga_generations, sizes=sizes,
    )
    summary = summarize(results)
    results.to_csv(args.output_dir / "budget_runs.csv", index=False)
    summary.to_csv(args.output_dir / "budget_summary.csv", index=False)
    if not args.smoke_test:
        perform_budget_anova(
            results, completed_only=True, budget=args.budget
        ).to_csv(args.output_dir / "budget_anova_completed_only.csv", index=False)
        perform_budget_anova(
            results, completed_only=False, budget=args.budget
        ).to_csv(args.output_dir / "budget_anova_timeout_as_budget.csv", index=False)
    plot_budget_results(
        results, args.budget, args.output_dir / "budget_times.png", sizes=sizes
    )
    if args.runtime_only:
        plot_budget_results(
            results,
            args.budget,
            args.output_dir / "budget_runtime_only.png",
            sizes=sizes,
            panels=("time",),
        )
    for case_study in sorted(results.case_study.unique()):
        plot_budget_results(
            results,
            args.budget,
            args.output_dir / f"{case_study}_budget_times.png",
            sizes=sizes,
            case_study=case_study,
        )
        if args.runtime_only:
            plot_budget_results(
                results,
                args.budget,
                args.output_dir / f"{case_study}_budget_runtime_only.png",
                sizes=sizes,
                case_study=case_study,
                panels=("time",),
            )


if __name__ == "__main__":
    main()
