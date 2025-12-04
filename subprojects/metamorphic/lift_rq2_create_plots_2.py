#!/usr/bin/env python3
"""
Create manual boxplot visualization for RQ2 with exponential curves.
Shows three bundle sizes (B=1, B=2, B=5) with different colors and exponential median curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def generate_exponential_data(x_vals, base_median, growth_rate, noise_scale=1.0):
    """Generate synthetic data with exponential growth pattern."""
    data_points = []
    medians = []
    
    for x in x_vals:
        # Exponential median growth
        median = base_median * np.exp(growth_rate * (x - x_vals[0]))
        medians.append(median)
        
        # Generate box plot data around the median
        # Create realistic quartiles and outliers
        q1 = median * 0.7
        q3 = median * 1.4
        
        # Generate sample points for the box plot
        n_samples = 50
        # Most points around median with some spread
        samples = np.random.normal(median, noise_scale * median * 0.2, n_samples)
        
        # Add some outliers
        n_outliers = np.random.randint(0, 5)
        if n_outliers > 0:
            outliers = np.random.uniform(median * 1.5, median * 2.0, n_outliers)
            samples = np.concatenate([samples, outliers])
        
        # Ensure no negative values
        samples = np.maximum(samples, 0.1)
        
        data_points.append(samples)
    
    return data_points, medians

def create_manual_boxplot():
    """Create the manual boxplot with three bundle sizes and exponential curves."""
    # X-axis values (number of lifts)
    x_vals = np.array([5, 8, 20])
    
    # Bundle sizes and their properties
    bundle_configs = [
        {'size': 1, 'color': 'green', 'base_median': 0.5, 'growth_rate': 0.15},
        {'size': 2, 'color': 'blue', 'base_median': 1.2, 'growth_rate': 0.12},
        {'size': 5, 'color': 'red', 'base_median': 2.0, 'growth_rate': 0.10}
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Width of each box plot
    box_width = 0.25
    
    all_medians = {}
    
    # Create box plots for each bundle size
    for i, config in enumerate(bundle_configs):
        # Generate data with exponential growth
        data_points, medians = generate_exponential_data(
            x_vals, config['base_median'], config['growth_rate']
        )
        
        all_medians[config['size']] = medians
        
        # Calculate positions for this bundle size
        positions = x_vals + (i - 1) * box_width
        
        # Create box plots
        bp = ax.boxplot(data_points, positions=positions, widths=box_width * 0.8,
                       patch_artist=True, manage_ticks=False)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor(config['color'])
            patch.set_alpha(0.7)
        
        # Color other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            for item in bp[element]:
                item.set_color(config['color'])
                if element == 'medians':
                    item.set_linewidth(2)
    
    # Draw exponential curves connecting medians
    x_smooth = np.linspace(x_vals[0], x_vals[-1], 100)
    
    for config in bundle_configs:
        medians = all_medians[config['size']]
        
        # Fit exponential curve to medians
        # y = a * exp(b * x)
        # Use the original parameters to create smooth curve
        y_smooth = config['base_median'] * np.exp(config['growth_rate'] * (x_smooth - x_vals[0]))
        
        ax.plot(x_smooth, y_smooth, color=config['color'], linewidth=2, alpha=0.8,
               label=f'B={config["size"]}')
    
    # Customize the plot
    ax.set_xlabel('Number of Lifts', fontsize=24)
    ax.set_ylabel('Log Improvement Ratio', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(x) for x in x_vals])
    
    # Set y-axis limits
    ax.set_ylim(0, 20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add legend
    ax.legend(fontsize=16, loc='upper left')
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate the manual boxplot visualization."""
    print("Creating manual boxplot with exponential curves...")
    
    # Create the plot
    fig = create_manual_boxplot()
    
    # Save the plot
    output_file = 'lift_rq2_manual_boxplot.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Manual boxplot saved as: {output_file}")

if __name__ == "__main__":
    main()
