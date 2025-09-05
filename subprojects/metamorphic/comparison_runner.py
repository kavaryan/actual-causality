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
from run_single_experiment import run_single_experiment

def run_comparison_study(num_vars_list = [5, 10, 15], num_trials = 20, awt_coeff = 0.8, timeout = 30):
    """Run the full comparison study across different numbers of variables."""
    
    # Initialize simulator and search space
    simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
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
                ('mm_bundled', {'bundle_size': 2}),
                ('mm_bundled', {'bundle_size': num_vars // 2})
            ]
            
            for method, kwargs in methods_to_test:
                result = run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
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
        return f'A* Bundled (size={bundle_size})'
    return row['method']

def plot_results(df, ci=95):
    """Create beautiful seaborn plots with confidence intervals."""
    
    # Add method labels
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution time vs number of variables
    ax1 = axes[0, 0]
    sns.lineplot(data=df.loc[df['success'], ['num_vars', 'time']], x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax1, err_style='band', errorbar=('ci', ci))
    ax1.set_title('Execution Time vs Number of Variables')
    ax1.set_xlabel('Number of Variables')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_yscale('log')
    ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    print("Running comparison study...")
    df = run_comparison_study()
    
    # Save results
    df.to_csv('comparison_results.csv', index=False)
    print(f"Results saved to comparison_results.csv")
    
    # Create plots
    plot_results(df)
