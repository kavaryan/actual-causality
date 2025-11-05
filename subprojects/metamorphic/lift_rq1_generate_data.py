#!/usr/bin/env python3
"""
Generate CSV data for RQ1 scalability study comparing BFS vs Bundled A*.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add path to access the case study modules
sys.path.append('.')
# sys.path.append('case_studies/lift')

from subprojects.metamorphic.case_studies.lift.mock_lift_simulation import MockLiftsSimulator
from subprojects.metamorphic.case_studies.lift.lift import LiftSearchSpace
from run_single_experiment import run_single_experiment

def run_rq1_scalability_study(num_vars_list=[5, 10, 15, 50], 
                             num_trials=10, awt_coeff=0.8, timeout=2):
    """
    Run RQ1 scalability study comparing BFS vs Bundled A* with 2-second timeout.
    
    Uses fixed bundle_size=5 for all problem sizes.
    """
    results = []
    
    for num_vars in tqdm(num_vars_list, desc="Problem Sizes"):
        # Use fixed bundle size of 5
        bundle_size = 5
        
        for trial in tqdm(range(num_trials), desc=f"N={num_vars}", leave=False):
            # Generate random configuration
            prob_active = 0.5
            v = np.random.binomial(1, prob_active, size=num_vars)
            
            # Ensure at least one lift is active
            if sum(v) == 0:
                v[np.random.randint(num_vars)] = 1
            
            # Calculate initial AWT and threshold using a temporary simulator
            temp_simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
            initial_awt = temp_simulator.simulate(sum(v))
            awt_thr = initial_awt * awt_coeff
            
            # Test both methods on the same configuration
            methods_to_test = [
                ('bfs', {}),
                ('mm_bundled', {'bundle_size': bundle_size})
            ]
            
            for method, kwargs in methods_to_test:
                # Create fresh simulator and search space for each experiment
                simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
                search_space = LiftSearchSpace(simulator, awt_thr=awt_thr, num_vars=num_vars)
                
                result = run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
                result.update({
                    'num_vars': num_vars,
                    'bundle_size': bundle_size,
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
    print("Starting RQ1 Scalability Study Data Generation...")
    print("Comparing BFS vs Bundled A* with 2-second timeout")
    
    # Run RQ1 experiments
    df_rq1 = run_rq1_scalability_study()
    
    # Save results to CSV
    print("Saving CSV results...")
    df_rq1.to_csv('rq1_scalability_results.csv', index=False)
    print(f"✓ Results saved to: rq1_scalability_results.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("RQ1 DATA GENERATION COMPLETED")
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
