import numpy as np

from case_studies import make_lift_problem
from search import (
    approximate_best_first,
    best_first_with_literal_subset_shrink,
    breadth_first_literal,
    bundled_best_first,
    bundled_best_first_with_literal_subset_shrink,
    bundled_breadth_first_literal,
    bundled_nsga2_pareto,
    nsga2_pareto,
)


def test_budgeted_unbundled_method_variants_return_real_solutions():
    problem = make_lift_problem(
        8, np.random.default_rng(3), speed=1.0, call_density=0.5
    )
    results = [
        breadth_first_literal(problem),
        approximate_best_first(problem),
        nsga2_pareto(problem, seed=3, population=30, generations=15),
        best_first_with_literal_subset_shrink(problem),
    ]
    assert all(result.solution is not None for result in results)
    assert all(problem.robustness(result.solution) > 0 for result in results)
    assert all(result.time > 0 for result in results)
    assert all(result.evaluations > 0 for result in results)


def test_budgeted_bundled_method_variants_return_real_solutions():
    problem = make_lift_problem(
        12, np.random.default_rng(13), speed=1.0, call_density=0.5
    )
    results = [
        bundled_breadth_first_literal(problem, bundle_size=2),
        bundled_best_first(problem, bundle_size=2),
        bundled_nsga2_pareto(
            problem, seed=13, bundle_size=2, population=20, generations=8
        ),
        bundled_best_first_with_literal_subset_shrink(problem, bundle_size=2),
    ]
    assert all(result.solution is not None for result in results)
    assert all(problem.robustness(result.solution) > 0 for result in results)
    assert all(result.time > 0 for result in results)
    assert all(result.evaluations > 0 for result in results)


def test_bundled_best_first_variants_return_real_solutions():
    problem = make_lift_problem(
        10, np.random.default_rng(11), speed=1.0, call_density=0.5
    )
    results = [
        bundled_best_first(problem, bundle_size=1),
        bundled_best_first(problem, bundle_size=2),
        bundled_best_first_with_literal_subset_shrink(problem, bundle_size=5),
    ]
    assert all(result.solution is not None for result in results)
    assert all(problem.robustness(result.solution) > 0 for result in results)
    assert all(result.time > 0 for result in results)
    assert all(result.evaluations > 0 for result in results)
