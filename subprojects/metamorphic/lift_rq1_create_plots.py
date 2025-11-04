#!/usr/bin/env python3
"""
Load RQ1 results from CSV and create plots.
Separated from data collection to avoid matplotlib issues.
"""

import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

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

def load_and_plot_rq1_results(csv_file='rq1_scalability_results.csv',adf=None):
    """Load RQ1 results from CSV and create plot."""
    print(f"Loading results from {csv_file}...")
    
    if adf is None:
        matplotlib.use('Agg')  # Use non-interactive backend
        
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} rows of data")
        except FileNotFoundError:
            print(f"✗ File {csv_file} not found. Run lift_rq1_generate_data.py first to generate data.")
            return
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return
    else:
        df = adf
    
    # Add method labels
    df['method_label'] = df.apply(create_method_label, axis=1)
    
    # Filter successful runs
    df_success = df[df['success']].copy()
    print(f"✓ Found {len(df_success)} successful runs out of {len(df)} total")
    
    # Print data summary
    print("\nData summary:")
    print(f"Methods: {list(df_success['method_label'].unique())}")
    print(f"Problem sizes: {sorted(df_success['num_vars'].unique())}")
    print(f"Time range: {df_success['time'].min():.3f} to {df_success['time'].max():.3f} seconds")
    
    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique methods and colors
    methods = df_success['method_label'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot each method
    for i, method in enumerate(methods):
        method_data = df_success[df_success['method_label'] == method]
        # Group by num_vars and calculate mean time
        grouped = method_data.groupby('num_vars')['time'].mean()
        
        print(f"  Plotting {method}: {len(grouped)} data points")
        
        ax.plot(grouped.index, grouped.values, 'o-', 
               color=colors[i % len(colors)], label=method, 
               linewidth=2, markersize=8)
    
    # Set labels and formatting
    ax.set_xlabel('Number of Variables (Lifts)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    # ax.set_title('RQ1: Scalability Comparison - BFS vs Bundled A*', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set log scale for y-axis if there's a wide range
    if df_success['time'].max() / df_success['time'].min() > 100:
        ax.set_yscale('log')
        print("  Using log scale for y-axis due to wide time range")
    
    print("✓ Plot created successfully")
    
    # Save plot
    if adf is None:
        print("Saving plot...")
        fig.savefig('rq1_scalability_plot.png', dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved as: rq1_scalability_plot.png")    
        plt.close(fig)
    else:
        plt.show()
    
    # Print final summary
    print("\nFinal summary:")
    summary = df_success.groupby(['num_vars', 'method_label'])['time'].agg(['count', 'mean', 'std']).round(3)
    print(summary)
    
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
    plot_df.to_csv('rq1_plot_data.csv', index=False)
    print(f"✓ Plot data saved to: rq1_plot_data.csv")

def main():
    """Main function to create plots from saved data."""
    print("=" * 60)
    print("RQ1 RESULTS PLOTTER")
    print("=" * 60)
    
    # Create plot from CSV data
    load_and_plot_rq1_results()
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
