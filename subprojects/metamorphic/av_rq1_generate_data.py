#!/usr/bin/env python3
"""
Generate CSV data for RQ1 scalability study comparing BFS vs A* for AV case study.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add path to access the case study modules
sys.path.append('.')

from subprojects.metamorphic.case_studies.av.mock_av_simulation import MockAVSimulator
from subprojects.metamorphic.case_studies.av.av import AVSearchSpace
from subprojects.metamorphic.av_run_single_experiment import av_run_single_experiment

def run_rq1_scalability_study(num_vars_list=[5, 10, 15, 20, 25, 30], 
                             num_trials=10, td_coeff=0.8, timeout=2):
    """
    Run RQ1 scalability study comparing BFS vs A* with 2-second timeout.
    """
    results = []
    
    for num_vars in tqdm(num_vars_list, desc="Problem Sizes"):
        for trial in tqdm(range(num_trials), desc=f"N={num_vars}", leave=False):
            # Generate random configuration
            prob_active = 0.5
            v = np.random.binomial(1, prob_active, size=num_vars)
            
            # Ensure at least one obstacle is active
            if sum(v) == 0:
                v[np.random.randint(num_vars)] = 1
            
            # Calculate initial TD and threshold using a temporary simulator
            temp_simulator = MockAVSimulator(p_a=(0, 0), p_b=(10, 10), speed=5.0, 
                                           average_max_time=1.0, simulator_startup_cost=0.1)
            initial_td = temp_simulator.simulate(sum(v))
            td_thr = initial_td * td_coeff
            
            # Test both methods on the same configuration
            methods_to_test = [
                ('bfs', {}),
                ('mm', {})
            ]
            
            for method, kwargs in methods_to_test:
                # Create fresh simulator and search space for each experiment
                simulator = MockAVSimulator(p_a=(0, 0), p_b=(10, 10), speed=5.0, 
                                          average_max_time=1.0, simulator_startup_cost=0.1)
                search_space = AVSearchSpace(simulator, td_thr=td_thr, num_vars=num_vars)
                
                result = av_run_single_experiment(td_thr, v, method, search_space, timeout, **kwargs)
                result.update({
                    'num_vars': num_vars,
                    'trial': trial,
                    'initial_td': initial_td,
                    'td_thr': td_thr,
                    'initial_active': sum(v),
                    'prob_active': prob_active
                })
                results.append(result)
    
    return pd.DataFrame(results)

def main():
    """Generate RQ1 data and save to CSV."""
    print("Starting AV RQ1 Scalability Study Data Generation...")
    print("Comparing BFS vs A* with 2-second timeout")
    
    # Run RQ1 experiments
    df_rq1 = run_rq1_scalability_study()
    
    # Save results to CSV
    print("Saving CSV results...")
    df_rq1.to_csv('av_rq1_scalability_results.csv', index=False)
    print(f"✓ Results saved to: av_rq1_scalability_results.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("AV RQ1 DATA GENERATION COMPLETED")
    print("="*50)
    print(f"Total experiments: {len(df_rq1)}")
    print(f"Problem sizes: {sorted(df_rq1['num_vars'].unique())}")
    print(f"Methods: {list(df_rq1['method'].unique())}")
    
    # Success rates summary
    success_summary = df_rq1.groupby(['num_vars', 'method']).agg({
        'success': ['count', 'sum', 'mean'],
        'timeout': 'sum'
    }).round(3)
    print("\nSuccess rates by problem size and method:")
    print(success_summary)

if __name__ == "__main__":
    main()
