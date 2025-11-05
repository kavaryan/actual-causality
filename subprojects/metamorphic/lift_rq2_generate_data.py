#!/usr/bin/env python3
"""
Generate CSV data for RQ2 ablation study: effect of bundling and different bundle sizes.
Compares A* (no bundling) vs A* Bundled with fixed bundle sizes: 5, 10, 20.
Tests on N=40 and N=60 (BFS excluded due to impracticality).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add path to access the case study modules
sys.path.append('.')

from case_studies.lift.mock_lift_simulation import MockLiftsSimulator
from case_studies.lift.lift import LiftSearchSpace
from run_single_experiment import run_single_experiment

def run_rq2_ablation_study(num_vars_list=[40, 60], 
                          bundle_sizes=[5, 10, 20],
                          num_trials=20, awt_coeff=0.8, timeout=30):
    """
    Run RQ2 ablation study comparing A* vs A* Bundled with different fixed bundle sizes.
    
    :param num_vars_list: List of problem sizes to test
    :param bundle_sizes: List of fixed bundle sizes to test
    :param num_trials: Number of trials per configuration
    :param awt_coeff: AWT coefficient for threshold calculation
    :param timeout: Timeout in seconds per experiment
    """
    results = []
    
    for num_vars in tqdm(num_vars_list, desc="Problem Sizes"):
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
            
            # Test methods: A* (no bundling) + A* Bundled with different bundle sizes
            methods_to_test = [('mm', {})]  # A* without bundling
            
            # Add A* Bundled with different bundle sizes
            for bundle_size in bundle_sizes:
                if bundle_size < num_vars:  # Only test if bundle size is smaller than problem size
                    methods_to_test.append(('mm_bundled', {'bundle_size': bundle_size}))
            
            for method, kwargs in methods_to_test:
                # Create fresh simulator and search space for each experiment
                simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
                search_space = LiftSearchSpace(simulator, awt_thr=awt_thr, num_vars=num_vars)
                
                result = run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
                result.update({
                    'num_vars': num_vars,
                    'bundle_size': kwargs.get('bundle_size', None),
                    'trial': trial,
                    'initial_awt': initial_awt,
                    'awt_thr': awt_thr,
                    'initial_active': sum(v),
                    'prob_active': prob_active
                })
                results.append(result)
    
    return pd.DataFrame(results)

def create_method_label(row):
    """Create readable labels for methods."""
    if row['method'] == 'mm':
        return 'A* (No Bundling)'
    elif row['method'] == 'mm_bundled':
        bundle_size = row.get('bundle_size', 'Unknown')
        return f'A* Bundled (size={bundle_size})'
    return row['method']

def main():
    """Generate RQ2 data and save to CSV."""
    print("Starting RQ2 Ablation Study Data Generation...")
    print("Comparing A* vs A* Bundled with different bundle sizes")
    print("Problem sizes: N=40, N=60")
    print("Bundle sizes: 5, 10, 20")
    
    # Run RQ2 experiments
    df_rq2 = run_rq2_ablation_study()
    
    # Add method labels
    df_rq2['method_label'] = df_rq2.apply(create_method_label, axis=1)
    
    # Save results to CSV
    print("Saving CSV results...")
    df_rq2.to_csv('lift_rq2_ablation_results.csv', index=False)
    print(f"✓ Results saved to: lift_rq2_ablation_results.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("RQ2 DATA GENERATION COMPLETED")
    print("="*50)
    print(f"Total experiments: {len(df_rq2)}")
    print(f"Problem sizes: {sorted(df_rq2['num_vars'].unique())}")
    print(f"Methods: {list(df_rq2['method'].unique())}")
    print(f"Bundle sizes tested: {sorted([x for x in df_rq2['bundle_size'].unique() if x is not None])}")
    
    # Success rates summary
    success_summary = df_rq2.groupby(['num_vars', 'method_label']).agg({
        'success': ['count', 'sum', 'mean'],
        'timeout': 'sum',
        'time': ['mean', 'median']
    }).round(3)
    print("\nSuccess rates and timing by problem size and method:")
    print(success_summary)

if __name__ == "__main__":
    main()
