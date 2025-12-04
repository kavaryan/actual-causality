#!/usr/bin/env python3
"""
Load AV RQ1 results from CSV and create plots.
Analyzes improvement ratios across speed and obstacle density classes.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from av_rq1_settings import SPEED_CLASSES, DENSITY_CLASSES

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def calculate_improvement_ratios(df):
    """Calculate log ratio of improvement (BFS time / A* time)."""
    # Filter successful runs only
    df_success = df[df['success']].copy()
    
    # Pivot to get BFS and A* times side by side
    pivot_df = df_success.pivot_table(
        index=['speed_class', 'density_class', 'num_vars', 'trial'],
        columns='method',
        values='time',
        aggfunc='first'
    ).reset_index()
    
    # Calculate improvement ratio and log ratio
    pivot_df['improvement_ratio'] = pivot_df['bfs'] / pivot_df['mm']
    pivot_df['log_ratio'] = np.log(pivot_df['improvement_ratio'])
    
    # Remove rows where either method failed
    pivot_df = pivot_df.dropna(subset=['bfs', 'mm'])
    
    return pivot_df

def create_method_label(row):
    """Create readable labels for methods."""
    if row['method'] == 'bfs':
        return 'BFS (Exhaustive)'
    elif row['method'] == 'mm':
        return 'A* (Metamorphic)'
    return row['method']

def create_improvement_plots(df_ratios, save_plots=True):
    """Create grid plot showing log improvement ratios."""
    # Get unique speed and density classes from the data
    speed_classes = sorted(df_ratios['speed_class'].unique())
    density_classes = sorted(df_ratios['density_class'].unique())
    
    # Create subplot grid based on actual data
    n_speed = len(speed_classes)
    n_density = len(density_classes)
    fig, axes = plt.subplots(n_speed, n_density, figsize=(4*n_density, 4*n_speed))
    
    # Handle case where we have only one row or column
    if n_speed == 1 and n_density == 1:
        axes = [[axes]]
    elif n_speed == 1:
        axes = [axes]
    elif n_density == 1:
        axes = [[ax] for ax in axes]
    
    for i, speed in enumerate(speed_classes):
        for j, density in enumerate(density_classes):
            ax = axes[i, j]
            
            # Filter data for this speed/density combination
            subset = df_ratios[
                (df_ratios['speed_class'] == speed) & 
                (df_ratios['density_class'] == density)
            ]
            
            if len(subset) > 0:
                # Create box plot
                box_data = []
                positions = []
                labels = []
                
                for k, num_vars in enumerate(sorted(subset['num_vars'].unique())):
                    var_data = subset[subset['num_vars'] == num_vars]['log_ratio']
                    if len(var_data) > 0:
                        box_data.append(var_data)
                        positions.append(k)
                        labels.append(str(num_vars))
                
                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    ax.set_xlabel('Number of Obstacles')
                    ax.set_ylabel('Log Improvement Ratio')
                    ax.set_title(f'Speed: {speed.title()}, Density: {density.title()}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Speed: {speed.title()}, Density: {density.title()}')
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('av_rq1_improvement_ratios.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def create_aggregated_box_plot(df_ratios, save_plots=True):
    """Create aggregated box plot of log ratios by number of obstacles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique obstacle counts
    obstacle_counts = sorted(df_ratios['num_vars'].unique())
    
    # Prepare data for box plot
    box_data = []
    
    for num_vars in obstacle_counts:
        var_data = df_ratios[df_ratios['num_vars'] == num_vars]['log_ratio']
        if len(var_data) > 0:
            box_data.append(var_data)
    
    if box_data:
        # Use actual obstacle counts as positions
        bp = ax.boxplot(box_data, positions=obstacle_counts, widths=0.8, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xticks(obstacle_counts)
        ax.set_xticklabels([str(x) for x in obstacle_counts])
        ax.set_xlabel('Number of Obstacles', fontsize=24)
        ax.set_ylabel('Log Improvement Ratio', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('av_rq1_aggregated_log_ratios.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main():
    """Main function to create plots and analysis from saved data."""
    try:
        # Load data
        df = pd.read_csv('av_rq1_scalability_results.csv')
        print(f"Loaded {len(df)} rows of data")
        
        # Calculate improvement ratios
        df_ratios = calculate_improvement_ratios(df)
        print(f"Calculated improvement ratios for {len(df_ratios)} valid comparisons")
        
        # Print basic statistics
        print("\nImprovement Ratio Statistics:")
        print("=" * 40)
        print(f"Mean log ratio: {df_ratios['log_ratio'].mean():.3f}")
        print(f"Std log ratio: {df_ratios['log_ratio'].std():.3f}")
        print(f"Min improvement ratio: {df_ratios['improvement_ratio'].min():.3f}")
        print(f"Max improvement ratio: {df_ratios['improvement_ratio'].max():.3f}")
        
        # Create improvement plots
        matplotlib.use('Agg')
        create_improvement_plots(df_ratios, save_plots=True)
        print("Improvement ratio plots saved as: av_rq1_improvement_ratios.png")
        
        # Create aggregated box plot
        create_aggregated_box_plot(df_ratios, save_plots=True)
        print("Aggregated log ratio plot saved as: av_rq1_aggregated_log_ratios.png")
        
        # Save processed data
        df_ratios.to_csv('av_rq1_improvement_ratios.csv', index=False)
        print("Improvement ratios saved to: av_rq1_improvement_ratios.csv")
        
        # Also create simple plot for backward compatibility
        create_simple_plot(df)
        
    except FileNotFoundError:
        print("File av_rq1_scalability_results.csv not found. Run av_rq1_generate_data.py first.")
    except Exception as e:
        print(f"Error: {e}")
        raise

def create_simple_plot(df):
    """Create simple plot for backward compatibility."""
    # Add method labels
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    # Filter successful runs
    df_success = df[df['success']].copy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique methods and colors
    methods = df_success['method_label'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot each method
    for i, method in enumerate(methods):
        method_data = df_success[df_success['method_label'] == method]
        # Group by num_vars and calculate mean time
        grouped = method_data.groupby('num_vars')['time'].mean()
        
        ax.plot(grouped.index, grouped.values, 'o-', 
               color=colors[i % len(colors)], label=method, 
               linewidth=2, markersize=8)
    
    # Set labels and formatting
    ax.set_xlabel('Number of Variables (Obstacles)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set log scale for y-axis if there's a wide range
    if df_success['time'].max() / df_success['time'].min() > 100:
        ax.set_yscale('log')
    
    # Save plot
    fig.savefig('av_rq1_scalability_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Simple scalability plot saved as: av_rq1_scalability_plot.png")
    
    # Save plot data to CSV for external use
    plot_data = []
    for method in methods:
        method_data = df_success[df_success['method_label'] == method]
        grouped = method_data.groupby('num_vars')['time'].mean()
        for num_vars, mean_time in grouped.items():
            plot_data.append({
                'method': method,
                'num_vars': num_vars,
                'mean_time': mean_time
            })
    
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv('av_rq1_plot_data.csv', index=False)
    print("Plot data saved to: av_rq1_plot_data.csv")

if __name__ == "__main__":
    main()
