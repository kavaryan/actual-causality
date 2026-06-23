# Budgeted four-method metamorphic search study

This directory now keeps only the budgeted experiment workflow plus the core
mock case studies and search implementations needed by that workflow.

The compared methods are:

1. BFS baseline.
2. Approximate best-first search.
3. NSGA-II Pareto search.
4. Best-first search followed by exhaustive subset shrinkage.

All retained plots report actual wall-clock execution time in seconds. Each
method run is isolated in a separate process and terminated at the requested
budget.

## Fixed-budget N study

```bash
~/.venv/bin/python subprojects/metamorphic-2/budget_study.py --budget 2
```

This writes:

- `budget_results/budget_runs.csv`
- `budget_results/budget_summary.csv`
- `budget_results/budget_anova_completed_only.csv`
- `budget_results/budget_anova_timeout_as_budget.csv`
- `budget_results/budget_times.png`

`budget_times.png` contains three shared-x boxplot panels:

- execution time in seconds;
- cardinality of the found causal set;
- robustness of the found causal set.

A method/N box is omitted when more than half of the corresponding runs time
out.

For a quick run that skips ANOVA:

```bash
~/.venv/bin/python subprojects/metamorphic-2/budget_study.py --smoke-test --budget 0.5 --sizes 10 --trials 1 --output-dir /tmp/metamorphic2_budget_smoke
```

## Budgeted bundle-size study

```bash
~/.venv/bin/python subprojects/metamorphic-2/rq2_budget_bundle_study.py --budget 2
```

This writes:

- `results/rq2_budget_bundle_results.csv`
- `results/rq2_budget_bundle_summary.csv`
- `figures/lift_rq2_budget_bfs.png`
- `figures/lift_rq2_budget_approx_best_first.png`
- `figures/lift_rq2_budget_nsga2.png`
- `figures/lift_rq2_budget_best_first_shrink.png`

Each method gets its own plot comparing bundle sizes `B = 1, 2, 5` for
`N = 10, 20, 50, 100`. A method/N/B box is omitted when more than half of the
corresponding runs time out.

For a quick run:

```bash
~/.venv/bin/python subprojects/metamorphic-2/rq2_budget_bundle_study.py --smoke-test --budget 0.5 --results-dir /tmp/metamorphic2_rq2_budget_smoke/results --figures-dir /tmp/metamorphic2_rq2_budget_smoke/figures
```

## Notebook

`metamorphic_budget_study.ipynb` runs the two retained budget studies and
displays the generated plots.
