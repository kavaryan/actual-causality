"""Mock lift and AV case studies used by the budgeted experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


class MockLiftsSimulator:
    """Paper lift mock: AWT decreases with active lifts."""

    def __init__(self, average_max_time=2.0, simulator_startup_cost=0.1,
                 speed=1.0, call_density=0.0):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost
        self.speed = speed
        self.call_density = max(call_density, 0.0)

    def simulate(self, num_lifts: int) -> float:
        if num_lifts == 0:
            return float("inf")
        return (
            self.stretch_coefficient
            * (1.0 + self.call_density)
            / (num_lifts * self.speed)
            + self.startup_cost
        )


class MockAVSimulator:
    """Paper AV mock, retained exactly in its inverse obstacle-count form."""

    def __init__(self, distance=10.0, speed=5.0, average_max_time=1.0,
                 simulator_startup_cost=0.1):
        self.distance = distance
        self.speed = speed
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost

    def simulate(self, num_obstacles: int) -> float:
        if num_obstacles == 0:
            return float("inf")
        return (
            self.distance
            / self.speed
            * self.stretch_coefficient
            / num_obstacles
            + self.startup_cost
        )


@dataclass(frozen=True)
class CausalProblem:
    case_study: str
    initial: tuple[int, ...]
    threshold: float
    simulator: object
    parameter_1_name: str
    parameter_1: float
    parameter_2_name: str
    parameter_2: float

    @property
    def n_vars(self) -> int:
        return len(self.initial)

    def robustness(self, candidate: frozenset[int]) -> float:
        return self.evaluate(candidate)[0]

    def evaluate(self, candidate: frozenset[int]) -> tuple[float, float]:
        changed = list(self.initial)
        for index in candidate:
            changed[index] = 1 - changed[index]
        output = self.simulator.simulate(sum(changed))
        if not np.isfinite(output):
            return -1e12, 0.0
        # As in metamorphic-1, simulator output is also the simulated execution
        # cost accumulated across calls. This avoids timing Python microseconds.
        return self.threshold - output, output

    def heuristic_gain(self, candidate: frozenset[int]) -> int:
        # Both paper mocks improve when the number of active indicators increases.
        return sum(1 if self.initial[index] == 0 else -1 for index in candidate)


def _configuration(n_vars: int, rng: np.random.Generator) -> tuple[int, ...]:
    # Keep enough inactive indicators for the threshold-reducing intervention
    # to be reachable in every generated benchmark instance.
    active = int(rng.integers(1, max(2, n_vars // 2 + 1)))
    values = np.zeros(n_vars, dtype=int)
    values[rng.choice(n_vars, size=active, replace=False)] = 1
    return tuple(int(value) for value in values)


def make_lift_problem(n_vars: int, rng: np.random.Generator, speed: float,
                      call_density: float, threshold_coeff: float = 0.8) -> CausalProblem:
    initial = _configuration(n_vars, rng)
    simulator = MockLiftsSimulator(speed=speed, call_density=call_density)
    threshold = simulator.simulate(sum(initial)) * threshold_coeff
    return CausalProblem(
        "lift", initial, threshold, simulator,
        "speed", speed, "call_density", call_density,
    )


def make_av_problem(n_vars: int, rng: np.random.Generator, speed: float,
                    distance: float, threshold_coeff: float = 0.8) -> CausalProblem:
    initial = _configuration(n_vars, rng)
    simulator = MockAVSimulator(speed=speed, distance=distance)
    threshold = simulator.simulate(sum(initial)) * threshold_coeff
    return CausalProblem(
        "av", initial, threshold, simulator,
        "speed", speed, "distance", distance,
    )
