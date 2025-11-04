#!/usr/bin/env python3
"""
Script to load RQ1 results from CSV and create plots.
Separated from data collection to isolate matplotlib issues.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

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

def load_and_plot_rq1_results(csv_file='rq1_scalability_results.csv'):
    """Load RQ1 results from CSV and create plot."""
    print(f"Loading results from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} rows of data")
    except FileNotFoundError:
        print(f"✗ File {csv_file} not found. Run main_rq1() first to generate data.")
        return
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
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
    
    # Create plot using the same approach as test_plot.py
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
    ax.set_title('RQ1: Scalability Comparison - BFS vs Bundled A*\n(2-second timeout)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set log scale for y-axis if there's a wide range
    if df_success['time'].max() / df_success['time'].min() > 100:
        ax.set_yscale('log')
        print("  Using log scale for y-axis due to wide time range")
    
    print("✓ Plot created successfully")
    
    # Save plot - test what was causing the slowdown
    print("Saving plot...")
    import time
    
    # Test different save options to identify the culprit
    save_tests = [
        ("without bbox_inches", lambda: fig.savefig('rq1_scalability_plot.png', dpi=150)),
        ("with bbox_inches='tight'", lambda: fig.savefig('rq1_scalability_plot_tight.png', dpi=150, bbox_inches='tight')),
        ("low dpi", lambda: fig.savefig('rq1_scalability_plot_lowdpi.png', dpi=100)),
    ]
    
    for test_name, save_func in save_tests:
        try:
            print(f"Testing save {test_name}...")
            start_time = time.time()
            save_func()
            end_time = time.time()
            print(f"✓ {test_name}: {end_time - start_time:.3f} seconds")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
    
    plt.close(fig)
    
    # Print final summary
    print("\nFinal summary:")
    summary = df_success.groupby(['num_vars', 'method_label'])['time'].agg(['count', 'mean', 'std']).round(3)
    print(summary)

def create_simple_plot(csv_file='rq1_scalability_results.csv'):
    """Create an ultra-simple plot with minimal matplotlib features."""
    print("Creating ultra-simple plot...")
    
    try:
        df = pd.read_csv(csv_file)
        df['method_label'] = df.apply(create_method_label, axis=1)
        df_success = df[df['success']].copy()
        
        # Create minimal figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Plot with minimal features
        methods = df_success['method_label'].unique()
        colors = ['b', 'r', 'g']
        
        for i, method in enumerate(methods):
            method_data = df_success[df_success['method_label'] == method]
            grouped = method_data.groupby('num_vars')['time'].mean()
            ax.plot(grouped.index, grouped.values, colors[i % len(colors)] + 'o-', 
                   linewidth=2, markersize=6)
        
        # Minimal save
        fig.savefig('rq1_simple_plot.png', dpi=72)
        print("✓ Simple plot saved as: rq1_simple_plot.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"✗ Simple plot failed: {e}")

def main():
    """Main function to create plots from saved data."""
    print("=" * 60)
    print("RQ1 RESULTS PLOTTER")
    print("=" * 60)
    
    # Try full plot first
    load_and_plot_rq1_results()
    
    # Also create simple plot as backup
    print("\n" + "-" * 40)
    create_simple_plot()
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
