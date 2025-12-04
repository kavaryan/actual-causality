#!/usr/bin/env python3
"""
Generate CSV data for RQ1 scalability study comparing BFS vs A* for AV case study.
Tests 4 subject classes: 2 speed × 2 obstacle density with configurable obstacle counts.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add path to access the case study modules
sys.path.append('.')

# Import shared configuration
from av_rq1_settings import (
    DEFAULT_NUM_VARS_LIST, DEFAULT_NUM_TRIALS, DEFAULT_TD_COEFF, DEFAULT_TIMEOUT,
    SPEED_CLASSES, DENSITY_CLASSES, AVERAGE_MAX_TIME, PROB_ACTIVE, 
    SIMULATOR_STARTUP_COST, P_A, P_B
)

from subprojects.metamorphic.case_studies.av.mock_av_simulation import MockAVSimulator
from subprojects.metamorphic.case_studies.av.av import AVSearchSpace
from subprojects.metamorphic.av_run_single_experiment import av_run_single_experiment

def run_rq1_scalability_study(num_vars_list=None, 
                             num_trials=None, td_coeff=None, timeout=None):
    """
    Run RQ1 scalability study comparing BFS vs A*.
    
    Tests 4 subject classes:
    - 2 speed classes: slow (2.5), fast (7.5)
    - 2 obstacle density classes: low (0.3), high (0.7)
    - Configurable obstacle counts and trials
    """
    # Use defaults if not provided
    if num_vars_list is None:
        num_vars_list = DEFAULT_NUM_VARS_LIST
    if num_trials is None:
        num_trials = DEFAULT_NUM_TRIALS
    if td_coeff is None:
        td_coeff = DEFAULT_TD_COEFF
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    # Print actual configuration being used
    print("Starting AV RQ1 Scalability Study Data Generation...")
    print(f"Testing {len(SPEED_CLASSES)} speed × {len(DENSITY_CLASSES)} obstacle density = {len(SPEED_CLASSES) * len(DENSITY_CLASSES)} subject classes")
    print(f"Speed classes: {list(SPEED_CLASSES.keys())} (values: {list(SPEED_CLASSES.values())})")
    print(f"Density classes: {list(DENSITY_CLASSES.keys())} (values: {list(DENSITY_CLASSES.values())})")
    print(f"Obstacle counts: {num_vars_list} with {num_trials} trials each")
    print(f"Timeout: {timeout}s")
    
    results = []
    
    # Use global constants for subject classes
    speed_classes = SPEED_CLASSES
    density_classes = DENSITY_CLASSES
    
    # Shared average_max_time that works for all configurations
    average_max_time = AVERAGE_MAX_TIME
    
    for speed_name, speed_val in speed_classes.items():
        for density_name, density_val in density_classes.items():
            for num_vars in tqdm(num_vars_list, desc=f"Speed={speed_name}, Density={density_name}"):
                for trial in tqdm(range(num_trials), desc=f"N={num_vars}", leave=False):
                    # Generate random configuration based on density class
                    v = np.random.binomial(1, density_val, size=num_vars)
                    
                    # Ensure at least one obstacle is active
                    if sum(v) == 0:
                        v[np.random.randint(num_vars)] = 1
                    
                    # Calculate initial TD and threshold using a temporary simulator
                    temp_simulator = MockAVSimulator(
                        p_a=P_A, p_b=P_B, speed=speed_val, 
                        average_max_time=average_max_time, 
                        simulator_startup_cost=SIMULATOR_STARTUP_COST
                    )
                    initial_td = temp_simulator.simulate(sum(v))
                    td_thr = initial_td * td_coeff
                    
                    # Test both methods on the same configuration
                    methods_to_test = [
                        ('bfs', {}),
                        ('mm', {})
                    ]
                    
                    for method, kwargs in methods_to_test:
                        # Create fresh simulator and search space for each experiment
                        simulator = MockAVSimulator(
                            p_a=P_A, p_b=P_B, speed=speed_val,
                            average_max_time=average_max_time,
                            simulator_startup_cost=SIMULATOR_STARTUP_COST
                        )
                        search_space = AVSearchSpace(simulator, td_thr=td_thr, num_vars=num_vars)
                        
                        result = av_run_single_experiment(td_thr, v, method, search_space, timeout, **kwargs)
                        result.update({
                            'num_vars': num_vars,
                            'speed_class': speed_name,
                            'speed_value': speed_val,
                            'density_class': density_name,
                            'density_value': density_val,
                            'trial': trial,
                            'initial_td': initial_td,
                            'td_thr': td_thr,
                            'initial_active': sum(v),
                            'prob_active': density_val
                        })
                        results.append(result)
    
    return pd.DataFrame(results)

def main():
    """Generate RQ1 data and save to CSV."""
    # Run RQ1 experiments
    df_rq1 = run_rq1_scalability_study(
        num_vars_list=DEFAULT_NUM_VARS_LIST,
        num_trials=DEFAULT_NUM_TRIALS,
        td_coeff=DEFAULT_TD_COEFF,
        timeout=DEFAULT_TIMEOUT
    )
    
    # Save results to CSV
    df_rq1.to_csv('av_rq1_scalability_results.csv', index=False)
    print(f"Results saved to: av_rq1_scalability_results.csv")
    
    # Print summary
    print(f"Total experiments: {len(df_rq1)}")
    print(f"Obstacle counts: {sorted(df_rq1['num_vars'].unique())}")
    print(f"Speed classes: {list(df_rq1['speed_class'].unique())}")
    print(f"Density classes: {list(df_rq1['density_class'].unique())}")
    print(f"Methods: {list(df_rq1['method'].unique())}")

if __name__ == "__main__":
    main()
