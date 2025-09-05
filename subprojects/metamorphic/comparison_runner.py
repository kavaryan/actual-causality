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

def run_comparison_study():
    """Run the full comparison study across different numbers of variables."""
    
    # Parameters
    num_vars_list = [5, 10, 15, 20, 25]
    num_trials = 20
    awt_coeff = 0.8  # Target 80% of initial AWT
    timeout = 30
    
    # Initialize simulator and search space
    simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
    search_space = SearchSpace(simulator.simulate)
    
    results = []
    
    for num_vars in tqdm(num_vars_list, desc="Variables"):
        for trial in tqdm(range(num_trials), desc=f"Trials (N={num_vars})", leave=False):
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

def plot_results(df):
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
    sns.lineplot(data=df[df['success']], x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax1, err_style='band', ci=95)
    ax1.set_title('Execution Time vs Number of Variables')
    ax1.set_xlabel('Number of Variables')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rate vs number of variables
    ax2 = axes[0, 1]
    success_rate = df.groupby(['num_vars', 'method_label'])['success'].mean().reset_index()
    sns.lineplot(data=success_rate, x='num_vars', y='success', hue='method_label', 
                marker='s', ax=ax2)
    ax2.set_title('Success Rate vs Number of Variables')
    ax2.set_xlabel('Number of Variables')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1.05)
    ax2.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of execution times for successful runs
    ax3 = axes[1, 0]
    successful_df = df[df['success']]
    sns.boxplot(data=successful_df, x='num_vars', y='time', hue='method_label', ax=ax3)
    ax3.set_title('Distribution of Execution Times (Successful Runs)')
    ax3.set_xlabel('Number of Variables')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timeout rate vs number of variables
    ax4 = axes[1, 1]
    timeout_rate = df.groupby(['num_vars', 'method_label'])['timeout'].mean().reset_index()
    sns.lineplot(data=timeout_rate, x='num_vars', y='timeout', hue='method_label', 
                marker='^', ax=ax4)
    ax4.set_title('Timeout Rate vs Number of Variables')
    ax4.set_xlabel('Number of Variables')
    ax4.set_ylabel('Timeout Rate')
    ax4.set_ylim(0, 1.05)
    ax4.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for num_vars in sorted(df['num_vars'].unique()):
        print(f"\nNumber of Variables: {num_vars}")
        print("-" * 40)
        subset = df[df['num_vars'] == num_vars]
        
        for method_label in subset['method_label'].unique():
            method_data = subset[subset['method_label'] == method_label]
            success_rate = method_data['success'].mean()
            timeout_rate = method_data['timeout'].mean()
            
            if success_rate > 0:
                successful_times = method_data[method_data['success']]['time']
                mean_time = successful_times.mean()
                std_time = successful_times.std()
                print(f"{method_label:20s}: Success={success_rate:.2f}, "
                      f"Timeout={timeout_rate:.2f}, "
                      f"Time={mean_time:.3f}±{std_time:.3f}s")
            else:
                print(f"{method_label:20s}: Success={success_rate:.2f}, "
                      f"Timeout={timeout_rate:.2f}, Time=N/A")

if __name__ == "__main__":
    print("Running comparison study...")
    df = run_comparison_study()
    
    # Save results
    df.to_csv('comparison_results.csv', index=False)
    print(f"Results saved to comparison_results.csv")
    
    # Create plots
    plot_results(df)
