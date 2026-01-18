#!/usr/bin/env python3
"""
Generate CSV data for RQ1 scalability study comparing BFS vs Bundled A*.
Tests 4 subject classes: 2 speed × 2 call density with 3 lift counts each.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os


# Import shared configuration
from subprojects.metamorphic.lift_rq1_settings import (
    DEFAULT_NUM_LIFTS_LIST, DEFAULT_NUM_TRIALS, DEFAULT_AWT_COEFF, DEFAULT_TIMEOUT,
)

from subprojects.metamorphic.case_studies.lift.mock_lift_simulation import MockLiftsSimulator
from subprojects.metamorphic.case_studies.lift.lift import LiftSearchSpace
from subprojects.metamorphic.lift_run_single_experiment import lift_run_single_experiment

def run_rq1_scalability_study(num_lifts_list, 
                             num_trials, awt_coeff, timeout, speed_classes, density_classes, bundle_size=5, average_max_time=2.0, prob_active=0.7):
    """
    Run RQ1 scalability study comparing BFS vs Bundled A*.
    
    Tests 4 subject classes:
    - 2 speed classes: slow (0.5), fast (2.0)
    - 2 call density classes: low (0.1), high (0.5)
    - Configurable lift counts and trials
    """
  
    
    # Print actual configuration being used
    print("Starting RQ1 Scalability Study Data Generation...")
    print(f"Testing {len(speed_classes)} speed × {len(density_classes)} call density = {len(speed_classes) * len(density_classes)} subject classes")
    print(f"Speed classes: {list(speed_classes.keys())} (values: {list(speed_classes.values())})")
    print(f"Density classes: {list(density_classes.keys())} (values: {list(density_classes.values())})")
    print(f"Lift counts: {num_lifts_list} with {num_trials} trials each")
    print(f"Bundle size: {bundle_size}, Timeout: {timeout}s")
    
    results = []
    
    for speed_name, speed_val in speed_classes.items():
        for density_name, density_val in density_classes.items():
            for num_lifts in tqdm(num_lifts_list, desc=f"Speed={speed_name}, Density={density_name}"):
                for trial in tqdm(range(num_trials), desc=f"N={num_lifts}", leave=False):
                    # Generate random configuration
                    v = np.random.binomial(1, prob_active, size=num_lifts)
                    
                    # Ensure at least one lift is active
                    if sum(v) == 0:
                        v[np.random.randint(num_lifts)] = 1
                    
                    # Calculate initial AWT and threshold using a temporary simulator
                    temp_simulator = MockLiftsSimulator(
                        average_max_time=average_max_time,
                        simulator_startup_cost=0.1,
                        speed=speed_val,
                        call_density=density_val
                    )
                    initial_awt = temp_simulator.simulate(sum(v))
                    awt_thr = initial_awt * awt_coeff
                    
                    # Test both methods on the same configuration
                    methods_to_test = [
                        ('bfs', {}),
                        ('mm_bundled', {'bundle_size': bundle_size})
                    ]
                    
                    for method, kwargs in methods_to_test:
                        # Create fresh simulator and search space for each experiment
                        simulator = MockLiftsSimulator(
                            average_max_time=average_max_time,
                            simulator_startup_cost=0.1,
                            speed=speed_val,
                            call_density=density_val
                        )
                        search_space = LiftSearchSpace(simulator, awt_thr=awt_thr, num_vars=num_lifts)
                        
                        result = lift_run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
                        result.update({
                            'num_lifts': num_lifts,
                            'speed_class': speed_name,
                            'speed_value': speed_val,
                            'density_class': density_name,
                            'density_value': density_val,
                            'trial': trial,
                            'initial_awt': initial_awt,
                            'awt_thr': awt_thr,
                            'initial_active': sum(v),
                            'prob_active': prob_active
                        })
                        results.append(result)
    
    return pd.DataFrame(results)

def main():
    """Generate RQ1 data and save to CSV."""
    # Run RQ1 experiments
    df_rq1 = run_rq1_scalability_study(
        num_lifts_list=DEFAULT_NUM_LIFTS_LIST,
        num_trials=DEFAULT_NUM_TRIALS,
        awt_coeff=DEFAULT_AWT_COEFF,
        timeout=DEFAULT_TIMEOUT
    )
    
    # Save results to CSV
    df_rq1.to_csv('rq1_scalability_results.csv', index=False)
    print(f"Results saved to: rq1_scalability_results.csv")
    
    # Print summary
    print(f"Total experiments: {len(df_rq1)}")
    print(f"Lift counts: {sorted(df_rq1['num_lifts'].unique())}")
    print(f"Speed classes: {list(df_rq1['speed_class'].unique())}")
    print(f"Density classes: {list(df_rq1['density_class'].unique())}")
    print(f"Methods: {list(df_rq1['method'].unique())}")

if __name__ == "__main__":
    main()
