"""Genuine implementations of the four agreed search methods."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import heapq
from math import comb

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from case_studies import CausalProblem


@dataclass(frozen=True)
class SearchResult:
    method: str
    solution: frozenset[int] | None
    robustness: float
    evaluations: int
    time: float
    timeout: bool = False


class EvaluationBudgetExceeded(RuntimeError):
    pass


class Evaluator:
    def __init__(self, problem: CausalProblem, budget: float | None = None):
        self.problem = problem
        self.cache: dict[frozenset[int], float] = {}
        self.simulation_time = 0.0
        self.budget = budget

    def __call__(self, candidate: frozenset[int]) -> float:
        if candidate not in self.cache:
            robustness, cost = self.problem.evaluate(candidate)
            if self.budget is not None and self.simulation_time + cost > self.budget:
                self.simulation_time = self.budget
                raise EvaluationBudgetExceeded
            self.cache[candidate] = robustness
            self.simulation_time += cost
        return self.cache[candidate]


def _finish(method, solution, robustness, evaluator) -> SearchResult:
    return SearchResult(
        method, solution, robustness, len(evaluator.cache),
        evaluator.simulation_time,
    )


def breadth_first(problem: CausalProblem, budget: float | None = None) -> SearchResult:
    """Exact BFS by cardinality, aggregated over exchangeable mock variables.

    The paper mocks depend only on the number of active indicators. Therefore
    all subsets with the same numbers of flipped active and inactive variables
    have identical outcomes. We still account for every subset and every
    simulator cost, but do so combinatorially rather than materialising as many
    as C(100, k) Python objects.
    """
    active_indices = [i for i, value in enumerate(problem.initial) if value == 1]
    inactive_indices = [i for i, value in enumerate(problem.initial) if value == 0]
    active_count = len(active_indices)
    inactive_count = len(inactive_indices)
    evaluations = 0
    simulation_time = 0.0
    for cardinality in range(1, problem.n_vars + 1):
        solution = None
        solution_robustness = np.nan
        minimum_inactive = max(0, cardinality - active_count)
        maximum_inactive = min(cardinality, inactive_count)
        for flipped_inactive in range(minimum_inactive, maximum_inactive + 1):
            flipped_active = cardinality - flipped_inactive
            multiplicity = (
                comb(inactive_count, flipped_inactive)
                * comb(active_count, flipped_active)
            )
            new_active_count = (
                active_count + flipped_inactive - flipped_active
            )
            output = problem.simulator.simulate(new_active_count)
            group_cost = multiplicity * output if np.isfinite(output) else 0.0
            if budget is not None and simulation_time + group_cost > budget:
                return SearchResult(
                    "bfs", None, np.nan, evaluations,
                    budget, timeout=True,
                )
            evaluations += multiplicity
            if np.isfinite(output):
                simulation_time += group_cost
                robustness = problem.threshold - output
            else:
                robustness = -1e12
            if robustness > 0 and solution is None:
                solution = frozenset(
                    inactive_indices[:flipped_inactive]
                    + active_indices[:flipped_active]
                )
                solution_robustness = robustness
        if solution is not None:
            return SearchResult(
                "bfs", solution, solution_robustness,
                evaluations, simulation_time,
            )
    return SearchResult("bfs", None, np.nan, evaluations, simulation_time)


def breadth_first_literal(problem: CausalProblem) -> SearchResult:
    """Literal subset-enumerating BFS for wall-clock timeout experiments."""
    evaluations = 0
    simulation_time = 0.0
    for cardinality in range(1, problem.n_vars + 1):
        for indices in combinations(range(problem.n_vars), cardinality):
            candidate = frozenset(indices)
            robustness, cost = problem.evaluate(candidate)
            evaluations += 1
            simulation_time += cost
            if robustness > 0:
                return SearchResult(
                    "bfs", candidate, robustness, evaluations, simulation_time
                )
    return SearchResult("bfs", None, np.nan, evaluations, simulation_time)


def approximate_best_first(
    problem: CausalProblem, budget: float | None = None
) -> SearchResult:
    """Return the first positive-robustness set popped by best-first search."""
    evaluate = Evaluator(problem, budget)
    empty = frozenset()
    queue = [((0, 0, ()), empty)]
    queued = {empty}
    while queue:
        _, candidate = heapq.heappop(queue)
        try:
            robustness = evaluate(candidate)
        except EvaluationBudgetExceeded:
            return SearchResult(
                "approx_best_first", None, np.nan, len(evaluate.cache),
                evaluate.simulation_time, timeout=True,
            )
        if candidate and robustness > 0:
            return _finish(
                "approx_best_first", candidate, robustness, evaluate
            )
        for index in range(problem.n_vars):
            if index in candidate:
                continue
            child = candidate | {index}
            if child in queued:
                continue
            queued.add(child)
            priority = (
                -problem.heuristic_gain(child),
                len(child),
                tuple(sorted(child)),
            )
            heapq.heappush(queue, (priority, child))
    return _finish("approx_best_first", None, np.nan, evaluate)


def _shrink_solution_by_cardinality(
    problem: CausalProblem,
    first: SearchResult,
    method: str,
    budget: float | None = None,
) -> SearchResult:
    if first.timeout:
        return SearchResult(
            method, None, np.nan, first.evaluations,
            first.time, timeout=True,
        )
    if first.solution is None:
        return SearchResult(
            method, None, np.nan, first.evaluations, first.time,
        )
    members = sorted(first.solution)
    active_members = [i for i in members if problem.initial[i] == 1]
    inactive_members = [i for i in members if problem.initial[i] == 0]
    active_count = len(active_members)
    inactive_count = len(inactive_members)
    evaluations = first.evaluations
    simulation_time = first.time
    for cardinality in range(1, len(members) + 1):
        solution = None
        solution_robustness = np.nan
        minimum_inactive = max(0, cardinality - active_count)
        maximum_inactive = min(cardinality, inactive_count)
        for flipped_inactive in range(minimum_inactive, maximum_inactive + 1):
            flipped_active = cardinality - flipped_inactive
            multiplicity = (
                comb(inactive_count, flipped_inactive)
                * comb(active_count, flipped_active)
            )
            candidate = frozenset(
                inactive_members[:flipped_inactive]
                + active_members[:flipped_active]
            )
            robustness, cost = problem.evaluate(candidate)
            group_cost = multiplicity * cost
            if budget is not None and simulation_time + group_cost > budget:
                return SearchResult(
                    method, None, np.nan, evaluations,
                    budget, timeout=True,
                )
            evaluations += multiplicity
            simulation_time += group_cost
            if robustness > 0 and solution is None:
                solution = candidate
                solution_robustness = robustness
        if solution is not None:
            return SearchResult(
                method, solution, solution_robustness,
                evaluations, simulation_time,
            )
    raise RuntimeError("The discovered cause was not found during subset enumeration")


def best_first_with_subset_shrink(
    problem: CausalProblem, budget: float | None = None
) -> SearchResult:
    """Best-first discovery followed by exhaustive, non-greedy subset shrinkage."""
    first = approximate_best_first(problem, budget)
    return _shrink_solution_by_cardinality(
        problem, first, "best_first_shrink", budget
    )


def _make_bundles(n_vars: int, bundle_size: int) -> tuple[tuple[int, ...], ...]:
    if bundle_size < 1:
        raise ValueError("bundle_size must be >= 1")
    return tuple(
        tuple(range(start, min(start + bundle_size, n_vars)))
        for start in range(0, n_vars, bundle_size)
    )


def _bundle_neighbors(
    candidate: frozenset[int],
    bundles: tuple[tuple[int, ...], ...],
) -> tuple[frozenset[int], ...]:
    candidate_set = set(candidate)
    neighbors = []
    for bundle in bundles:
        bundle_set = set(bundle)
        in_bundle = bundle_set & candidate_set
        if len(in_bundle) == len(bundle_set):
            neighbors.append(frozenset(candidate_set - bundle_set))
        elif not in_bundle:
            neighbors.append(frozenset(candidate_set | bundle_set))
        else:
            neighbors.append(frozenset(candidate_set - bundle_set))
            neighbors.append(frozenset(candidate_set | bundle_set))
    return tuple(neighbors)


def bundled_breadth_first_literal(
    problem: CausalProblem,
    bundle_size: int,
) -> SearchResult:
    """Literal BFS over a bundled-neighbor graph for wall-clock studies."""
    bundles = _make_bundles(problem.n_vars, bundle_size)
    start = frozenset()
    frontier = deque([start])
    visited = {start}
    evaluations = 0
    simulation_time = 0.0
    while frontier:
        candidate = frontier.popleft()
        for child in _bundle_neighbors(candidate, bundles):
            if child in visited:
                continue
            visited.add(child)
            robustness, cost = problem.evaluate(child)
            evaluations += 1
            simulation_time += cost
            if child and robustness > 0:
                return SearchResult(
                    "bundled_bfs", child, robustness,
                    evaluations, simulation_time,
                )
            frontier.append(child)
    return SearchResult("bundled_bfs", None, np.nan, evaluations, simulation_time)


def bundled_best_first(
    problem: CausalProblem,
    bundle_size: int,
    budget: float | None = None,
) -> SearchResult:
    """Best-first search whose neighbors flip fixed consecutive bundles."""
    evaluate = Evaluator(problem, budget)
    bundles = _make_bundles(problem.n_vars, bundle_size)
    empty = frozenset()
    queue = [((0, 0, ()), empty)]
    queued = {empty}
    while queue:
        _, candidate = heapq.heappop(queue)
        try:
            robustness = evaluate(candidate)
        except EvaluationBudgetExceeded:
            return SearchResult(
                "bundled_best_first", None, np.nan, len(evaluate.cache),
                evaluate.simulation_time, timeout=True,
            )
        if candidate and robustness > 0:
            return _finish(
                "bundled_best_first", candidate, robustness, evaluate
            )
        for child in _bundle_neighbors(candidate, bundles):
            if child in queued:
                continue
            queued.add(child)
            priority = (
                -problem.heuristic_gain(child),
                len(child),
                tuple(sorted(child)),
            )
            heapq.heappush(queue, (priority, child))
    return _finish("bundled_best_first", None, np.nan, evaluate)


def bundled_best_first_with_subset_shrink(
    problem: CausalProblem,
    bundle_size: int,
    budget: float | None = None,
) -> SearchResult:
    """Bundled best-first discovery followed by exhaustive subset shrinkage."""
    first = bundled_best_first(problem, bundle_size, budget)
    return _shrink_solution_by_cardinality(
        problem, first, "bundled_best_first_shrink", budget
    )


def bundled_best_first_with_literal_subset_shrink(
    problem: CausalProblem,
    bundle_size: int,
) -> SearchResult:
    """Bundled best-first followed by literal subset enumeration."""
    first = bundled_best_first(problem, bundle_size)
    if first.solution is None:
        return SearchResult(
            "bundled_best_first_shrink", None, np.nan,
            first.evaluations, first.time,
        )
    members = sorted(first.solution)
    evaluations = first.evaluations
    simulation_time = first.time
    for cardinality in range(1, len(members) + 1):
        for indices in combinations(members, cardinality):
            candidate = frozenset(indices)
            robustness, cost = problem.evaluate(candidate)
            evaluations += 1
            simulation_time += cost
            if robustness > 0:
                return SearchResult(
                    "bundled_best_first_shrink", candidate, robustness,
                    evaluations, simulation_time,
                )
    raise RuntimeError("The discovered cause was not found during subset enumeration")


def best_first_with_literal_subset_shrink(
    problem: CausalProblem,
) -> SearchResult:
    """Best-first followed by literal enumeration of every discovered subset."""
    first = approximate_best_first(problem)
    if first.solution is None:
        return SearchResult(
            "best_first_shrink", None, np.nan, first.evaluations, first.time
        )
    members = sorted(first.solution)
    evaluations = first.evaluations
    simulation_time = first.time
    for cardinality in range(1, len(members) + 1):
        for indices in combinations(members, cardinality):
            candidate = frozenset(indices)
            robustness, cost = problem.evaluate(candidate)
            evaluations += 1
            simulation_time += cost
            if robustness > 0:
                return SearchResult(
                    "best_first_shrink", candidate, robustness,
                    evaluations, simulation_time,
                )
    raise RuntimeError("The discovered cause was not found during subset enumeration")


class _NSGAProblem(ElementwiseProblem):
    def __init__(self, problem: CausalProblem, evaluate: Evaluator):
        super().__init__(n_var=problem.n_vars, n_obj=2, xl=0, xu=1, vtype=bool)
        self.evaluate_candidate = evaluate

    def _evaluate(self, x, out, *args, **kwargs):
        candidate = frozenset(int(index) for index in np.flatnonzero(x))
        robustness = self.evaluate_candidate(candidate)
        # pymoo minimizes both objectives: cardinality and negative robustness.
        out["F"] = [len(candidate), -robustness]


class _BundledNSGAProblem(ElementwiseProblem):
    def __init__(
        self,
        problem: CausalProblem,
        evaluate: Evaluator,
        bundles: tuple[tuple[int, ...], ...],
    ):
        super().__init__(n_var=len(bundles), n_obj=2, xl=0, xu=1, vtype=bool)
        self.evaluate_candidate = evaluate
        self.bundles = bundles

    def candidate_from_vector(self, x) -> frozenset[int]:
        selected = set()
        for bundle_index in np.flatnonzero(x):
            selected.update(self.bundles[int(bundle_index)])
        return frozenset(selected)

    def _evaluate(self, x, out, *args, **kwargs):
        candidate = self.candidate_from_vector(x)
        robustness = self.evaluate_candidate(candidate)
        # pymoo minimizes both objectives: actual cardinality and negative
        # robustness. The decision variables are bundle selectors.
        out["F"] = [len(candidate), -robustness]


def nsga2_pareto(problem: CausalProblem, seed: int, population: int = 60,
                 generations: int = 30,
                 budget: float | None = None) -> SearchResult:
    """Select minimum cardinality with positive robustness from NSGA-II's front."""
    evaluate = Evaluator(problem, budget)
    algorithm = NSGA2(
        pop_size=population,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )
    try:
        result = minimize(
            _NSGAProblem(problem, evaluate), algorithm, ("n_gen", generations),
            seed=seed, verbose=False,
        )
    except EvaluationBudgetExceeded:
        return SearchResult(
            "nsga2", None, np.nan, len(evaluate.cache),
            evaluate.simulation_time, timeout=True,
        )
    feasible = []
    if result.X is not None:
        for vector in np.atleast_2d(result.X):
            candidate = frozenset(int(index) for index in np.flatnonzero(vector))
            robustness = evaluate(candidate)
            if candidate and robustness > 0:
                feasible.append((len(candidate), -robustness, candidate, robustness))
    if not feasible:
        return _finish("nsga2", None, np.nan, evaluate)
    _, _, candidate, robustness = min(feasible)
    return _finish("nsga2", candidate, robustness, evaluate)


def bundled_nsga2_pareto(
    problem: CausalProblem,
    seed: int,
    bundle_size: int,
    population: int = 60,
    generations: int = 30,
    budget: float | None = None,
) -> SearchResult:
    """NSGA-II over bundle bit-vectors, selecting min-cardinality positive set."""
    evaluate = Evaluator(problem, budget)
    bundles = _make_bundles(problem.n_vars, bundle_size)
    nsga_problem = _BundledNSGAProblem(problem, evaluate, bundles)
    algorithm = NSGA2(
        pop_size=population,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )
    try:
        result = minimize(
            nsga_problem, algorithm, ("n_gen", generations),
            seed=seed, verbose=False,
        )
    except EvaluationBudgetExceeded:
        return SearchResult(
            "bundled_nsga2", None, np.nan, len(evaluate.cache),
            evaluate.simulation_time, timeout=True,
        )
    feasible = []
    if result.X is not None:
        for vector in np.atleast_2d(result.X):
            candidate = nsga_problem.candidate_from_vector(vector)
            robustness = evaluate(candidate)
            if candidate and robustness > 0:
                feasible.append((len(candidate), -robustness, candidate, robustness))
    if not feasible:
        return _finish("bundled_nsga2", None, np.nan, evaluate)
    _, _, candidate, robustness = min(feasible)
    return _finish("bundled_nsga2", candidate, robustness, evaluate)
