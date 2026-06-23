"""Budgeted RQ2 bundle-size plots for each search method.

Runs lift-only bundle-size ablations under a real wall-clock timeout. For each
method, the plot compares B=1,2,5 across N and omits a box when more than half
of that method/N/B group timed out.
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

from case_studies import make_lift_problem
from search import (
    bundled_best_first,
    bundled_best_first_with_literal_subset_shrink,
    bundled_breadth_first_literal,
    bundled_nsga2_pareto,
)


ROOT = Path(__file__).resolve().parent
METHODS = ["bfs", "approx_best_first", "nsga2", "best_first_shrink"]
METHOD_LABELS = {
    "bfs": "BFS",
    "approx_best_first": "Approx. best-first",
    "nsga2": "NSGA-II",
    "best_first_shrink": "Best-first + shrink",
}
BUNDLE_COLORS = {
    1: "#6bb6ff",
    2: "#ff6b5f",
    5: "#62c96f",
}


def _worker(
    queue,
    n_vars,
    speed,
    density,
    instance_seed,
    method,
    bundle_size,
    nsga_population,
    nsga_generations,
):
    try:
        rng = np.random.default_rng(instance_seed)
        problem = make_lift_problem(n_vars, rng, speed, density)
        started = time.perf_counter()
        if method == "bfs":
            result = bundled_breadth_first_literal(problem, bundle_size)
        elif method == "approx_best_first":
            result = bundled_best_first(problem, bundle_size)
        elif method == "nsga2":
            result = bundled_nsga2_pareto(
                problem,
                instance_seed,
                bundle_size,
                population=nsga_population,
                generations=nsga_generations,
            )
        elif method == "best_first_shrink":
            result = bundled_best_first_with_literal_subset_shrink(
                problem, bundle_size
            )
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


def run_budgeted_rq2(
    sizes=(10, 20, 50, 100),
    bundle_sizes=(1, 2, 5),
    budget=2.0,
    trials=3,
    seed=7,
    nsga_population=60,
    nsga_generations=30,
) -> pd.DataFrame:
    rows = []
    speeds = {"S1": 0.5, "S2": 1.0, "S3": 1.5}
    densities = {"C1": 0.5, "C2": 1.0, "C3": 1.5}
    for n_vars in sizes:
        for speed_class, speed in speeds.items():
            for density_class, density in densities.items():
                for trial in range(trials):
                    instance_seed = (
                        seed + n_vars * 1000 + trial * 10
                        + int(speed * 13) + int(density * 17)
                    )
                    for method in METHODS:
                        for bundle_size in bundle_sizes:
                            result = run_with_timeout(
                                n_vars,
                                speed,
                                density,
                                instance_seed,
                                method,
                                bundle_size,
                                nsga_population,
                                nsga_generations,
                                budget=budget,
                            )
                            rows.append({
                                "case_study": "lift",
                                "n_vars": n_vars,
                                "trial": trial,
                                "parameter_1_class": speed_class,
                                "parameter_1": speed,
                                "parameter_2_class": density_class,
                                "parameter_2": density,
                                "method": method,
                                "bundle_size": bundle_size,
                                **result,
                            })
        frame = pd.DataFrame(rows)
        current = frame[frame.n_vars == n_vars]
        timeout_rates = current.groupby(["method", "bundle_size"]).timeout.mean()
        print(f"N={n_vars}", flush=True)
        for method in METHODS:
            parts = [
                f"B={bundle_size}:{timeout_rates.get((method, bundle_size), 0):.0%}"
                for bundle_size in bundle_sizes
            ]
            print(f"  {method}: " + ", ".join(parts), flush=True)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(
        ["n_vars", "method", "bundle_size"], as_index=False
    ).agg(
        mean_time=("time", "mean"),
        median_time=("time", "median"),
        timeout_rate=("timeout", "mean"),
        runs=("timeout", "size"),
        success_rate=("success", "mean"),
        median_cardinality=("cardinality", "median"),
        median_robustness=("robustness", "median"),
    )


def plot_method(
    results: pd.DataFrame,
    method: str,
    budget: float,
    output: Path,
    sizes=None,
    bundle_sizes=None,
) -> None:
    sizes = sorted(sizes if sizes is not None else results.n_vars.unique())
    bundle_sizes = sorted(
        bundle_sizes if bundle_sizes is not None else results.bundle_size.unique()
    )
    plotted = results[(results.method == method) & (~results.timeout)].copy()
    method_results = results[results.method == method]
    timeout_rates = method_results.groupby(
        ["n_vars", "bundle_size"]
    ).timeout.mean()
    x = np.arange(len(sizes))
    width = 0.22
    offsets = np.linspace(-width, width, len(bundle_sizes))

    fig, ax = plt.subplots(figsize=(6.16, 4.56))
    for offset, bundle_size in zip(offsets, bundle_sizes):
        groups = [
            plotted[
                (plotted.n_vars == n_vars)
                & (plotted.bundle_size == bundle_size)
                & plotted.time.notna()
            ].time.to_numpy()
            for n_vars in sizes
        ]
        kept = [
            (position, group)
            for position, n_vars, group in zip(x + offset, sizes, groups)
            if len(group) > 0
            and timeout_rates.get((n_vars, bundle_size), 0.0) <= 0.5
        ]
        if not kept:
            continue
        positions, values = zip(*kept)
        color = BUNDLE_COLORS.get(int(bundle_size), "#bdbdbd")
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=width * 0.95,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "#333333", "linewidth": 1.1},
            boxprops={"edgecolor": "#2f6db2", "linewidth": 0.8},
            whiskerprops={"color": "#2f6db2", "linewidth": 0.8},
            capprops={"color": "#2f6db2", "linewidth": 0.8},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.85)
        ax.plot([], [], color=color, linewidth=8, label=f"B = {bundle_size}")

    ax.set_yscale("log")
    positive = plotted.loc[plotted.time > 0, "time"]
    if not positive.empty:
        ax.set_ylim(max(positive.min() / 2, 1e-6), budget * 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("Number of Lifts", fontsize=15)
    ax.set_ylabel("Execution Time (seconds)", fontsize=15)
    ax.set_title(METHOD_LABELS[method], fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="both", alpha=0.22, linestyle="--")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def create_method_plots(
    results: pd.DataFrame,
    output_dir: Path,
    budget: float,
    sizes=None,
    bundle_sizes=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        plot_method(
            results,
            method,
            budget,
            output_dir / f"lift_rq2_budget_{method}.png",
            sizes=sizes,
            bundle_sizes=bundle_sizes,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument(
        "--sizes", type=str, default="10,20,50,100",
        help="Comma-separated N values.",
    )
    parser.add_argument(
        "--bundle-sizes", type=str, default="1,2,5",
        help="Comma-separated bundle sizes.",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nsga-population", type=int, default=60)
    parser.add_argument("--nsga-generations", type=int, default=30)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    bundle_sizes = [
        int(value.strip()) for value in args.bundle_sizes.split(",")
        if value.strip()
    ]
    if args.smoke_test:
        sizes = sizes[:2]
        args.trials = 1
        args.nsga_population = min(args.nsga_population, 20)
        args.nsga_generations = min(args.nsga_generations, 8)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    results = run_budgeted_rq2(
        sizes=sizes,
        bundle_sizes=bundle_sizes,
        budget=args.budget,
        trials=args.trials,
        seed=args.seed,
        nsga_population=args.nsga_population,
        nsga_generations=args.nsga_generations,
    )
    summary = summarize(results)
    results.to_csv(args.results_dir / "rq2_budget_bundle_results.csv", index=False)
    summary.to_csv(args.results_dir / "rq2_budget_bundle_summary.csv", index=False)
    create_method_plots(
        results, args.figures_dir, args.budget,
        sizes=sizes, bundle_sizes=bundle_sizes,
    )


if __name__ == "__main__":
    main()
