#!/usr/bin/env python3
"""
Load RQ1 results from CSV and create plots.
Analyzes improvement ratios across speed and call density classes.
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
from lift_rq1_settings import SPEED_CLASSES, DENSITY_CLASSES

# Set matplotlib to use a simple font to avoid rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def calculate_improvement_ratios(df):
    """Calculate log ratio of improvement (BFS time / Bundled time)."""
    # Filter successful runs only
    df_success = df[df['success']].copy()
    
    # Pivot to get BFS and bundled times side by side
    pivot_df = df_success.pivot_table(
        index=['speed_class', 'density_class', 'num_lifts', 'trial'],
        columns='method',
        values='time',
        aggfunc='first'
    ).reset_index()
    
    # Calculate improvement ratio and log ratio
    pivot_df['improvement_ratio'] = pivot_df['bfs'] / pivot_df['mm_bundled']
    pivot_df['log_ratio'] = np.log(pivot_df['improvement_ratio'])
    
    # Remove rows where either method failed
    pivot_df = pivot_df.dropna(subset=['bfs', 'mm_bundled'])
    
    return pivot_df

def create_improvement_plots(df_ratios, save_plots=True):
    """Create grid plot showing log improvement ratios."""
    # Get unique speed and density classes from the data
    speed_classes = sorted(df_ratios['speed_class'].unique())
    density_classes = sorted(df_ratios['density_class'].unique())
    
    # Create subplot grid based on actual data
    n_speed = len(speed_classes)
    n_density = len(density_classes)
    fig, axes = plt.subplots(n_speed, n_density, figsize=(4*n_density, 4*n_speed))
    # fig.suptitle('Log Improvement Ratio (BFS/Bundled) by Speed and Call Density', fontsize=14)
    
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
                
                for k, num_lifts in enumerate(sorted(subset['num_lifts'].unique())):
                    lift_data = subset[subset['num_lifts'] == num_lifts]['log_ratio']
                    if len(lift_data) > 0:
                        box_data.append(lift_data)
                        positions.append(k)
                        labels.append(str(num_lifts))
                
                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    ax.set_xlabel('Number of Lifts')
                    ax.set_ylabel('Log Improvement Ratio')
                    ax.set_title(f'Speed: {speed.title()}, Density: {density.title()}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Speed: {speed.title()}, Density: {density.title()}')
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('lift_rq1_improvement_ratios.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def create_aggregated_box_plot(df_ratios, save_plots=True):
    """Create aggregated box plot of log ratios by number of lifts."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique lift counts
    lift_counts = sorted(df_ratios['num_lifts'].unique())
    
    # Prepare data for box plot
    box_data = []
    positions = []
    labels = []
    
    for i, num_lifts in enumerate(lift_counts):
        lift_data = df_ratios[df_ratios['num_lifts'] == num_lifts]['log_ratio']
        if len(lift_data) > 0:
            box_data.append(lift_data)
            positions.append(i)
            labels.append(str(num_lifts))
    
    if box_data:
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Number of Lifts', fontsize=24)
        ax.set_ylabel('Log Improvement Ratio', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('lift_rq1_aggregated_log_ratios.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def perform_ols_analysis(df_ratios):
    """Perform OLS analysis on log improvement ratios."""
    print("OLS Analysis Results:")
    print("=" * 50)
    
    # Prepare data for OLS
    df_ols = df_ratios.copy()
    df_ols['elevators'] = df_ols['num_lifts'].astype(str)
    
    # Fit OLS model
    model = ols("log_ratio ~ C(speed_class) * C(density_class) * C(elevators)", data=df_ols).fit()
    
    # Print model summary
    print(model.summary())
    
    # Print ANOVA table
    print("\nANOVA Table:")
    print("=" * 30)
    anova_table = anova_lm(model, typ=2)
    print(anova_table)
    
    return model

def plot_residuals(model, df_ratios, save_plots=True):
    """Plot residual diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Residual Diagnostics', fontsize=14)
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Speed/Density
    df_plot = df_ratios.copy()
    df_plot['residuals'] = residuals
    df_plot['condition'] = df_plot['speed_class'] + '_' + df_plot['density_class']
    
    conditions = df_plot['condition'].unique()
    for i, condition in enumerate(conditions):
        subset = df_plot[df_plot['condition'] == condition]
        axes[1, 1].scatter(subset['num_lifts'], subset['residuals'], 
                          label=condition, alpha=0.6)
    
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Number of Lifts')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Conditions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('lift_rq1_residual_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main():
    """Main function to create plots and analysis from saved data."""
    try:
        # Load data
        df = pd.read_csv('rq1_scalability_results.csv')
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
        print("Improvement ratio plots saved as: lift_rq1_improvement_ratios.png")
        
        # Create aggregated box plot
        create_aggregated_box_plot(df_ratios, save_plots=True)
        print("Aggregated log ratio plot saved as: lift_rq1_aggregated_log_ratios.png")
        
        # Perform OLS analysis
        model = perform_ols_analysis(df_ratios)
        
        # Plot residuals
        plot_residuals(model, df_ratios, save_plots=True)
        print("Residual diagnostics saved as: lift_rq1_residual_diagnostics.png")
        
        # Save processed data
        df_ratios.to_csv('rq1_improvement_ratios.csv', index=False)
        print("Improvement ratios saved to: rq1_improvement_ratios.csv")
        
    except FileNotFoundError:
        print("File rq1_scalability_results.csv not found. Run lift_rq1_generate_data.py first.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
