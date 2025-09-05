import sys
import os
sys.path.append('subprojects/metamorphic')
sys.path.append('subprojects/metamorphic/case-studies/lift')

from mock_lift_simulation import MockLiftsSimulator
from search_formulation import SearchSpace
from experiment import run_exp_subject_individual

# using actual simulator
# from lift_simulation import LiftSimulation, run_lift_simulation_for_lifts
# simulator_func = run_lift_simulation_for_lifts

# using a mock simulator
mock_simulator = MockLiftsSimulator(average_max_time=10000)
simulator_func = mock_simulator.simulate

search_space = SearchSpace(simulator_func)

r = run_exp_subject_individual(20, 0.5, simulator_func, search_space)
print(r)
