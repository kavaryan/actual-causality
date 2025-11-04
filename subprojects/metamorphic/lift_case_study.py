import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import os
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

sys.path.append('../..')

from case_studies.lift.mock_lift_simulation import MockLiftsSimulator
from case_studies.lift.lift import LiftSearchSpace
from run_single_experiment import run_single_experiment

def run_lift_case_study(num_vars_list=[5, 8, 10, 12, 15], num_trials=40, awt_coeff=0.8, 
                       timeout=60, simulator_startup_cost=0.1, poisson_lambdas=[2.0, 3.0, 4.0]):
    """
    Run the comprehensive lift case study with multiple subtypes.
    
    Each subtype is characterized by:
    - Different number of total lifts (num_vars)
    - Different customer demand distribution (poisson_lambda)
    """
    
    results = []
    subtype_counter = 0
    
    # Create 10 subtypes by combining different configurations
    subtypes = []
    for num_vars in num_vars_list:
        for lambda_param in poisson_lambdas[:2]:  # Use first 2 lambda values to get 10 subtypes
            subtypes.append((num_vars, lambda_param))
            subtype_counter += 1
            if subtype_counter >= 10:
                break
        if subtype_counter >= 10:
            break
    
    for subtype_idx, (num_vars, lambda_param) in enumerate(tqdm(subtypes, desc="Subtypes")):
        # Initialize simulator with varying parameters based on subtype
        simulator = MockLiftsSimulator(
            average_max_time=1.0 + 0.1 * lambda_param,  # Vary based on demand
            simulator_startup_cost=simulator_startup_cost
        )
        search_space = LiftSearchSpace(simulator.simulate, awt_thr=0.0, num_vars=num_vars)  # Will be updated per trial
        
        for trial in tqdm(range(num_trials), desc=f"Subtype {subtype_idx+1} (N={num_vars}, λ={lambda_param})", leave=False):
            # Generate random configuration influenced by Poisson parameter
            # Higher lambda means more demand, so more lifts should be active initially
            prob_active = min(0.8, 0.3 + lambda_param * 0.1)
            v = np.random.binomial(1, prob_active, size=num_vars)
            
            # Ensure at least one lift is active
            if sum(v) == 0:
                v[np.random.randint(num_vars)] = 1
            
            # Calculate initial AWT and threshold
            initial_awt = simulator.simulate(sum(v))
            awt_thr = initial_awt * awt_coeff
            search_space.awt_thr = awt_thr
            
            # Test all methods on the same configuration
            methods_to_test = [
                ('bfs', {}),
                ('mm', {}),
                ('mm_bundled', {'bundle_size': max(1, num_vars // 5)}),  # N/5 bundling
                ('mm_bundled', {'bundle_size': max(1, num_vars // 2)})   # N/2 bundling
            ]
            
            for method, kwargs in methods_to_test:
                result = run_single_experiment(awt_thr, v, method, search_space, timeout, **kwargs)
                result.update({
                    'subtype': subtype_idx + 1,
                    'num_vars': num_vars,
                    'lambda_param': lambda_param,
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
    if row['method'] == 'bfs':
        return 'BFS (Exhaustive)'
    elif row['method'] == 'mm':
        return 'A* (Metamorphic)'
    elif row['method'] == 'mm_bundled':
        bundle_size = row.get('bundle_size', 2)
        num_vars = row.get('num_vars', 10)
        
        if bundle_size == max(1, num_vars // 5):
            return 'A* Bundled (N/5)'
        elif bundle_size == max(1, num_vars // 2):
            return 'A* Bundled (N/2)'
        else:
            return f'A* Bundled (size={bundle_size})'
    return row['method']

def perform_wilcoxon_tests(df):
    """Perform pairwise Wilcoxon signed-rank tests between methods."""
    # Filter successful runs only
    df_success = df[df['success'] == True].copy()
    df_success['method_label'] = df_success.apply(create_method_label, axis=1)
    
    methods = df_success['method_label'].unique()
    results = []
    
    # Group by (subtype, trial) to get paired samples
    grouped = df_success.groupby(['subtype', 'trial'])
    
    for method1 in methods:
        for method2 in methods:
            if method1 >= method2:  # Avoid duplicate comparisons
                continue
                
            # Get paired samples
            times1 = []
            times2 = []
            
            for (subtype, trial), group in grouped:
                group_methods = group['method_label'].values
                if method1 in group_methods and method2 in group_methods:
                    time1 = group[group['method_label'] == method1]['time'].iloc[0]
                    time2 = group[group['method_label'] == method2]['time'].iloc[0]
                    times1.append(time1)
                    times2.append(time2)
            
            if len(times1) >= 10:  # Need sufficient samples
                try:
                    statistic, p_value = wilcoxon(times1, times2, alternative='two-sided')
                    results.append({
                        'Method 1': method1,
                        'Method 2': method2,
                        'n_pairs': len(times1),
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'median_diff': np.median(np.array(times1) - np.array(times2))
                    })
                except ValueError as e:
                    # Handle cases where differences are all zero
                    results.append({
                        'Method 1': method1,
                        'Method 2': method2,
                        'n_pairs': len(times1),
                        'statistic': np.nan,
                        'p_value': 1.0,
                        'significant': False,
                        'median_diff': 0.0
                    })
    
    return pd.DataFrame(results)

def create_comprehensive_plots(df):
    """Create comprehensive plots for the case study."""
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: RQ1 - Scalability comparison (BFS vs A*)
    ax1 = plt.subplot(2, 3, 1)
    df_rq1 = df[(df['method'].isin(['bfs', 'mm'])) & (df['success'])].copy()
    sns.lineplot(data=df_rq1, x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax1, err_style='band', errorbar=('ci', 95))
    ax1.set_title('RQ1: Scalability Comparison\n(BFS vs A* Metamorphic)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Variables (Lifts)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_yscale('log')
    ax1.legend(title='Method')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RQ2a - Effect of bundling (A* vs A* Bundled)
    ax2 = plt.subplot(2, 3, 2)
    df_rq2a = df[(df['method'].isin(['mm', 'mm_bundled'])) & (df['success'])].copy()
    sns.lineplot(data=df_rq2a, x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax2, err_style='band', errorbar=('ci', 95))
    ax2.set_title('RQ2a: Effect of Bundling\n(A* vs A* Bundled)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Variables (Lifts)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.legend(title='Method')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RQ2b - Bundle size comparison
    ax3 = plt.subplot(2, 3, 3)
    df_rq2b = df[(df['method'] == 'mm_bundled') & (df['success'])].copy()
    sns.lineplot(data=df_rq2b, x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax3, err_style='band', errorbar=('ci', 95))
    ax3.set_title('RQ2b: Bundle Size Effect\n(Different Bundle Sizes)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Variables (Lifts)')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.legend(title='Bundle Size')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate by method
    ax4 = plt.subplot(2, 3, 4)
    success_rates = df.groupby(['method_label', 'num_vars'])['success'].mean().reset_index()
    sns.lineplot(data=success_rates, x='num_vars', y='success', hue='method_label', 
                marker='o', ax=ax4)
    ax4.set_title('Success Rate by Method', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Variables (Lifts)')
    ax4.set_ylabel('Success Rate')
    ax4.set_ylim(0, 1.05)
    ax4.legend(title='Method')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Distribution of execution times
    ax5 = plt.subplot(2, 3, 5)
    df_success = df[df['success']].copy()
    sns.boxplot(data=df_success, x='method_label', y='time', ax=ax5)
    ax5.set_title('Distribution of Execution Times\n(Successful Runs Only)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Method')
    ax5.set_ylabel('Execution Time (seconds)')
    ax5.set_yscale('log')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Timeout analysis
    ax6 = plt.subplot(2, 3, 6)
    timeout_rates = df.groupby(['method_label', 'num_vars'])['timeout'].mean().reset_index()
    sns.lineplot(data=timeout_rates, x='num_vars', y='timeout', hue='method_label', 
                marker='o', ax=ax6)
    ax6.set_title('Timeout Rate by Method', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Number of Variables (Lifts)')
    ax6.set_ylabel('Timeout Rate')
    ax6.set_ylim(0, 1.05)
    ax6.legend(title='Method')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("="*80)
    print("LIFT CASE STUDY - SUMMARY STATISTICS")
    print("="*80)
    
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    # Overall statistics
    print(f"\nTotal experiments: {len(df)}")
    print(f"Number of subtypes: {df['subtype'].nunique()}")
    print(f"Trials per subtype: {df['trial'].nunique()}")
    print(f"Variable sizes tested: {sorted(df['num_vars'].unique())}")
    
    # Success rates by method
    print("\n" + "="*50)
    print("SUCCESS RATES BY METHOD")
    print("="*50)
    success_by_method = df.groupby('method_label').agg({
        'success': ['count', 'sum', 'mean'],
        'timeout': 'mean'
    }).round(3)
    success_by_method.columns = ['Total', 'Successful', 'Success_Rate', 'Timeout_Rate']
    print(success_by_method)
    
    # Execution time statistics (successful runs only)
    print("\n" + "="*50)
    print("EXECUTION TIME STATISTICS (SUCCESSFUL RUNS)")
    print("="*50)
    df_success = df[df['success']].copy()
    time_stats = df_success.groupby('method_label')['time'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    print(time_stats)
    
    # Performance by problem size
    print("\n" + "="*50)
    print("PERFORMANCE BY PROBLEM SIZE")
    print("="*50)
    size_stats = df_success.groupby(['num_vars', 'method_label'])['time'].agg([
        'count', 'mean', 'median'
    ]).round(4)
    print(size_stats)

def save_results_and_plots(df, wilcoxon_results, fig):
    """Save all results to files."""
    # Save raw results
    df.to_csv('lift_case_study_results.csv', index=False)
    print(f"\nRaw results saved to: lift_case_study_results.csv")
    
    # Save Wilcoxon test results
    wilcoxon_results.to_csv('lift_case_study_wilcoxon_tests.csv', index=False)
    print(f"Wilcoxon test results saved to: lift_case_study_wilcoxon_tests.csv")
    
    # Save plots
    fig.savefig('lift_case_study_plots.png', dpi=300, bbox_inches='tight')
    fig.savefig('lift_case_study_plots.pdf', bbox_inches='tight')
    print(f"Plots saved to: lift_case_study_plots.png and lift_case_study_plots.pdf")

def run_rq1_scalability_study(num_vars_list=[5, 10, 15, 50], 
                             num_trials=10, awt_coeff=0.8, timeout=2):
    """
    Run RQ1 scalability study comparing BFS vs Bundled A* with 2-second timeout.
    
    For N in [5,10,15]: use bundle_size=5
    For N >= 50: use bundle_size=N/10
    """
    results = []
    
    # Initialize simulator
    simulator = MockLiftsSimulator(average_max_time=1.0, simulator_startup_cost=0.1)
    
    for num_vars in tqdm(num_vars_list, desc="Problem Sizes"):
        # Determine bundle size
        if num_vars <= 15:
            bundle_size = 5
        else:
            bundle_size = max(1, num_vars // 10)
        
        search_space = LiftSearchSpace(simulator.simulate, awt_thr=0.0, num_vars=num_vars)
        
        for trial in tqdm(range(num_trials), desc=f"N={num_vars}", leave=False):
            # Generate random configuration
            prob_active = 0.5
            v = np.random.binomial(1, prob_active, size=num_vars)
            
            # Ensure at least one lift is active
            if sum(v) == 0:
                v[np.random.randint(num_vars)] = 1
            
            # Calculate initial AWT and threshold
            initial_awt = simulator.simulate(sum(v))
            awt_thr = initial_awt * awt_coeff
            search_space.awt_thr = awt_thr
            
            # Test both methods on the same configuration
            methods_to_test = [
                ('bfs', {}),
                ('mm_bundled', {'bundle_size': bundle_size})
            ]
            
            for method, kwargs in methods_to_test:
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
    """Run the complete lift case study."""
    print("Starting Lift Case Study...")
    print("This may take several minutes to complete.")
    
    # Run the experiments
    df = run_lift_case_study(
        num_vars_list=[5, 8, 10, 12, 15],
        num_trials=40,
        awt_coeff=0.8,
        timeout=60,
        simulator_startup_cost=0.1,
        poisson_lambdas=[2.0, 3.0, 4.0]
    )
    
    # Perform statistical tests
    print("\nPerforming Wilcoxon signed-rank tests...")
    wilcoxon_results = perform_wilcoxon_tests(df)
    
    # Create plots
    print("\nCreating plots...")
    fig = create_comprehensive_plots(df)
    
    # Print summary
    print_summary_statistics(df)
    
    # Print Wilcoxon results
    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*80)
    print(wilcoxon_results.to_string(index=False))
    
    # Save everything
    save_results_and_plots(df, wilcoxon_results, fig)
    
    # Show plots
    plt.show()
    
    print("\n" + "="*80)
    print("CASE STUDY COMPLETED SUCCESSFULLY!")
    print("="*80)

def main_rq1():
    """Run RQ1 scalability study only - just collect data and save to CSV."""
    print("Starting RQ1 Scalability Study...")
    print("Comparing BFS vs Bundled A* with 2-second timeout")
    
    # Run RQ1 experiments
    df_rq1 = run_rq1_scalability_study()
    
    # Add method labels for summary
    df_rq1['method_label'] = df_rq1.apply(create_method_label, axis=1)
    
    # Print summary
    print("\n" + "="*50)
    print("RQ1 SCALABILITY STUDY RESULTS")
    print("="*50)
    
    # Success rates by method and problem size
    success_summary = df_rq1.groupby(['num_vars', 'method_label']).agg({
        'success': ['count', 'sum', 'mean'],
        'timeout': 'sum',
        'time': ['mean', 'median']
    }).round(3)
    print("\nSuccess rates and timing by problem size:")
    print(success_summary)
    
    # Save results to CSV
    print("Saving CSV results...")
    df_rq1.to_csv('rq1_scalability_results.csv', index=False)
    print(f"✓ Results saved to: rq1_scalability_results.csv")
    
    print("\n" + "="*50)
    print("RQ1 DATA COLLECTION COMPLETED!")
    print("Use 'python plot_rq1_results.py' to create plots from the saved data.")
    print("="*50)


if __name__ == "__main__":
    main()
