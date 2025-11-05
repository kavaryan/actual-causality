import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import os

# Add the case-studies/lift directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'case-studies', 'lift'))

from mock_lift_simulation import MockLiftsSimulator
from search_formulation import SearchSpace
from subprojects.metamorphic.lift_run_single_experiment import lift_run_single_experiment

def run_comparison_study(num_vars_list = [5, 10, 15], num_trials = 20, awt_coeff = 0.8,
        timeout = 30, simulator_startup_cost=0.1):
    """Run the full comparison study across different numbers of variables."""
    
    # Initialize simulator and search space
    simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=simulator_startup_cost)
    search_space = SearchSpace(simulator.simulate)
    
    results = []
    
    for num_vars in tqdm(num_vars_list, desc="Variables"):
        for trial in tqdm(range(num_trials), desc=f"Trials (N={num_vars})", leave=True):
            # Generate one random configuration for this trial
            v = np.random.randint(0, 2, size=num_vars)
            
            # Calculate initial AWT and threshold
            initial_awt = simulator.simulate(sum(v))
            awt_thr = initial_awt * awt_coeff
            
            # Test all methods on the same v
            methods_to_test = [
                ('bfs', {}),
                ('mm', {}),
                ('mm_bundled', {'bundle_size': max(1, num_vars // 5)}),
                ('mm_bundled', {'bundle_size': num_vars // 2})
            ]
            
            for method, kwargs in methods_to_test:
                result = lift_run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
                result.update({
                    'num_vars': num_vars,
                    'trial': trial,
                    'initial_awt': initial_awt,
                    'awt_thr': awt_thr,
                    'initial_active': sum(v)
                })
                results.append(result)
    
    return pd.DataFrame(results)

def create_method_label(row):
    """Create a readable label for each method."""
    if row['method'] == 'bfs':
        return 'BFS'
    elif row['method'] == 'mm':
        return 'A*'
    elif row['method'] == 'mm_bundled':
        bundle_size = row.get('bundle_size', 2)
        num_vars = row.get('num_vars', 10)
        
        # Categorize bundle sizes consistently
        if bundle_size == max(1, num_vars // 5):
            return 'A* Bundled (N/5)'
        elif bundle_size == num_vars // 2:
            return 'A* Bundled (N/2)'
        else:
            return f'A* Bundled (size={bundle_size})'
    return row['method']

def plot_execution_time_vs_num_vars(df, ax1, ci=95, methods='all'):
    """Create beautiful seaborn plots with confidence intervals."""
    
    if methods != 'all':
        df = df[df['method'].isin(methods)]

    # Add method labels
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    sns.lineplot(data=df.loc[df['success'], ['num_vars', 'time', 'method_label']], x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax1, err_style='band', errorbar=('ci', ci))
    ax1.set_title('Simulated Execution Time vs Number of Variables')
    ax1.set_xlabel('Number of Variables')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xscale('log')
    if methods == 'all':    
        ax1.set_yscale('log')
    # ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.legend(title='Method', loc='lower right')
    ax1.grid(True, alpha=0.3)
    
if __name__ == "__main__":
    print("Running comparison study...")
    df = run_comparison_study()
    
    # Save results
    df.to_csv('comparison_results.csv', index=False)
    print(f"Results saved to comparison_results.csv")
    
    # Create plots
    plot_execution_time_vs_num_vars(df)
