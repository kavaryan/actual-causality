#!/usr/bin/env python3
"""
Analyze RQ2 ablation study results and create visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def create_method_label(row):
    """Create readable labels for methods."""
    if row['method'] == 'mm':
        return 'A* (No Bundling)'
    elif row['method'] == 'mm_bundled':
        bundle_size = row.get('bundle_size', 'Unknown')
        return f'A* Bundled (size={bundle_size})'
    return row['method']

def perform_wilcoxon_tests(df):
    """Perform pairwise Wilcoxon signed-rank tests between methods."""
    # Filter successful runs only
    df_success = df[df['success'] == True].copy()
    
    methods = df_success['method_label'].unique()
    results = []
    
    # Group by (num_vars, trial) to get paired samples
    grouped = df_success.groupby(['num_vars', 'trial'])
    
    for method1 in methods:
        for method2 in methods:
            if method1 >= method2:  # Avoid duplicate comparisons
                continue
                
            # Get paired samples
            times1 = []
            times2 = []
            
            for (num_vars, trial), group in grouped:
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

def create_rq2_plots(df):
    """Create comprehensive plots for RQ2 ablation study."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Performance comparison by bundle size
    ax1 = plt.subplot(2, 3, 1)
    df_success = df[df['success']].copy()
    sns.lineplot(data=df_success, x='num_vars', y='time', hue='method_label', 
                marker='o', ax=ax1, err_style='band', errorbar=('ci', 95))
    ax1.set_title('RQ2: Effect of Bundle Size\n(A* vs A* Bundled)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Variables (Lifts)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rate by method
    ax2 = plt.subplot(2, 3, 2)
    success_rates = df.groupby(['method_label', 'num_vars'])['success'].mean().reset_index()
    sns.lineplot(data=success_rates, x='num_vars', y='success', hue='method_label', 
                marker='o', ax=ax2)
    ax2.set_title('Success Rate by Method', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Variables (Lifts)')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1.05)
    ax2.legend(title='Method')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of execution times (box plot)
    ax3 = plt.subplot(2, 3, 3)
    sns.boxplot(data=df_success, x='method_label', y='time', ax=ax3)
    ax3.set_title('Distribution of Execution Times\n(Successful Runs Only)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timeout analysis
    ax4 = plt.subplot(2, 3, 4)
    timeout_rates = df.groupby(['method_label', 'num_vars'])['timeout'].mean().reset_index()
    sns.lineplot(data=timeout_rates, x='num_vars', y='timeout', hue='method_label', 
                marker='o', ax=ax4)
    ax4.set_title('Timeout Rate by Method', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Variables (Lifts)')
    ax4.set_ylabel('Timeout Rate')
    ax4.set_ylim(0, 1.05)
    ax4.legend(title='Method')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Bundle size effect (bar plot)
    ax5 = plt.subplot(2, 3, 5)
    bundled_data = df_success[df_success['method'] == 'mm_bundled'].copy()
    if not bundled_data.empty:
        bundle_stats = bundled_data.groupby('bundle_size')['time'].mean().reset_index()
        sns.barplot(data=bundle_stats, x='bundle_size', y='time', ax=ax5)
        ax5.set_title('Average Time by Bundle Size\n(A* Bundled Only)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Bundle Size')
        ax5.set_ylabel('Average Execution Time (seconds)')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Speedup comparison (relative to no bundling)
    ax6 = plt.subplot(2, 3, 6)
    # Calculate speedup relative to A* (no bundling)
    speedup_data = []
    for num_vars in df_success['num_vars'].unique():
        subset = df_success[df_success['num_vars'] == num_vars]
        baseline_time = subset[subset['method'] == 'mm']['time'].mean()
        
        for method_label in subset['method_label'].unique():
            if method_label != 'A* (No Bundling)':
                method_time = subset[subset['method_label'] == method_label]['time'].mean()
                speedup = baseline_time / method_time if method_time > 0 else 0
                speedup_data.append({
                    'num_vars': num_vars,
                    'method_label': method_label,
                    'speedup': speedup
                })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        sns.lineplot(data=speedup_df, x='num_vars', y='speedup', hue='method_label', 
                    marker='o', ax=ax6)
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax6.set_title('Speedup vs A* (No Bundling)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Number of Variables (Lifts)')
        ax6.set_ylabel('Speedup Factor')
        ax6.legend(title='Method')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("="*80)
    print("RQ2 ABLATION STUDY - SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal experiments: {len(df)}")
    print(f"Problem sizes tested: {sorted(df['num_vars'].unique())}")
    print(f"Bundle sizes tested: {sorted([x for x in df['bundle_size'].unique() if x is not None])}")
    print(f"Trials per configuration: {df['trial'].nunique()}")
    
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

def main():
    """Analyze RQ2 results and create visualizations."""
    # Load results
    try:
        df = pd.read_csv('lift_rq2_ablation_results.csv')
    except FileNotFoundError:
        print("Error: lift_rq2_ablation_results.csv not found!")
        print("Please run lift_rq2_generate_data.py first to generate the data.")
        return
    
    # Add method labels
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    print("Starting RQ2 Ablation Study Analysis...")
    
    # Perform statistical tests
    print("\nPerforming Wilcoxon signed-rank tests...")
    wilcoxon_results = perform_wilcoxon_tests(df)
    
    # Create plots
    print("\nCreating plots...")
    fig = create_rq2_plots(df)
    
    # Print summary
    print_summary_statistics(df)
    
    # Print Wilcoxon results
    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*80)
    print(wilcoxon_results.to_string(index=False))
    
    # Save results
    wilcoxon_results.to_csv('lift_rq2_wilcoxon_tests.csv', index=False)
    print(f"\nWilcoxon test results saved to: lift_rq2_wilcoxon_tests.csv")
    
    # Save plots
    fig.savefig('lift_rq2_ablation_plots.png', dpi=300, bbox_inches='tight')
    fig.savefig('lift_rq2_ablation_plots.pdf', bbox_inches='tight')
    print(f"Plots saved to: lift_rq2_ablation_plots.png and lift_rq2_ablation_plots.pdf")
    
    print("\n" + "="*80)
    print("RQ2 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
