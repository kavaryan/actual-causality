#!/usr/bin/env python3
"""
Compare k-leg liability with Shapley values for the static AEB system.

This script generates 20 random contexts and compares the liability values
computed using k-leg liability (with k=2) versus Shapley values.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import numpy as np
import matplotlib.pyplot as plt
from core.scm import read_system
from core.failure import ClosedHalfSpaceFailureSet
from subprojects.liab.k_leg_liab import k_leg_liab
from subprojects.liab.shapley_liab import shapley_liab

def load_static_aeb_system():
    """Load the static AEB system from the configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'av_static.conf')
    return read_system(config_path)

def create_perfect_aeb_system():
    """Create a perfect AEB system that never has collisions (specification)."""
    from core.scm import Component, SCMSystem, BoundedFloatInterval
    
    # Create components for a perfect system
    components = [
        Component("dist = dist_u"),
        Component("vel = vel_u"), 
        Component("radar_conf = radar_conf_u"),
        Component("critical_dist = vel * 0.8 + (vel * vel) / (2 * 8.0)"),
        Component("dist_margin = dist - critical_dist"),
        Component("aeb_should_trigger = 1.0"),  # Perfect system always triggers AEB
        Component("brake_force = 1.0"),  # Perfect system always applies full braking
        Component("collision_risk = 0.0"),  # Perfect system has no collision risk
        Component("collision = 0.0"),  # Perfect system never has collisions
        Component("collision_severity = 0.0"),  # Perfect system has no collision severity
    ]
    
    # Use same domains as original system
    domains = {
        'dist_u': BoundedFloatInterval(1.0, 50.0),
        'vel_u': BoundedFloatInterval(2.0, 20.0),
        'radar_conf_u': BoundedFloatInterval(0.0, 1.0),
        'dist': BoundedFloatInterval(0.0, 100.0),
        'vel': BoundedFloatInterval(0.0, 100.0),
        'radar_conf': BoundedFloatInterval(0.0, 100.0),
        'critical_dist': BoundedFloatInterval(-50.0, 100.0),
        'dist_margin': BoundedFloatInterval(-50.0, 100.0),
        'aeb_should_trigger': BoundedFloatInterval(0.0, 1.0),
        'brake_force': BoundedFloatInterval(0.0, 1.0),
        'collision_risk': BoundedFloatInterval(0.0, 1.0),
        'collision': BoundedFloatInterval(0.0, 1.0),
        'collision_severity': BoundedFloatInterval(0.0, 1.0),
    }
    
    return SCMSystem(components, domains)

def generate_random_contexts(system, num_contexts=20, seed=42):
    """Generate random contexts for the AEB system."""
    np.random.seed(seed)
    contexts = []
    
    for i in range(num_contexts):
        context = {
            'dist_u': np.random.uniform(1.0, 50.0),      # Distance: 1-50m
            'vel_u': np.random.uniform(2.0, 20.0),       # Velocity: 2-20 m/s
            'radar_conf_u': np.random.uniform(0.0, 1.0)  # Radar confidence: 0-1
        }
        contexts.append(context)
    
    return contexts

def compare_liability_methods():
    """Compare k-leg liability with Shapley values over multiple contexts."""
    print("=== Comparing K-leg Liability vs Shapley Values ===\n")
    
    # Load system
    system = load_static_aeb_system()
    
    # Convert to System objects for liability calculations
    # We need to create System objects from the SCM
    scm = system
    
    # Create specification system (ideal case - no collision)
    # We need different systems for meaningful liability comparison
    # S will be a "perfect" system that never has collisions
    S = create_perfect_aeb_system()  # Specification system (perfect)
    T = system  # Implementation system (actual)
    
    # Define failure set - collision occurs (collision > 0.1 to capture more cases)
    F = ClosedHalfSpaceFailureSet({'collision': (0.1, 'ge')})
    
    # Generate random contexts
    contexts = generate_random_contexts(system, num_contexts=20)
    
    # Store results
    k_leg_results = []
    shapley_results = []
    collision_states = []
    
    print("Analyzing contexts...")
    print("Context | Dist | Vel  | Radar | Collision")
    print("-" * 50)
    
    for i, context in enumerate(contexts):
        # Get system state
        state = system.get_state(context)
        collision_value = state['collision']
        collision_states.append(collision_value)
        
        # Only analyze contexts where collision actually occurs
        if collision_value > 0.1:
            try:
                # Calculate k-leg liability (k=2)
                print(f"        Computing k-leg liability...")
                k_leg_liab_values = k_leg_liab(T, S, context, F, k=2)
                k_leg_results.append(k_leg_liab_values)
                
                # Calculate Shapley values
                print(f"        Computing Shapley values...")
                shapley_values = shapley_liab(T, S, context, F, k=-1)
                shapley_results.append(shapley_values)
                
                # Print results for this context
                print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f}")
                
                # Debug: Show system states
                t_state = T.get_state(context)
                s_state = S.get_state(context)
                print(f"        T collision: {t_state['collision']:.3f}, S collision: {s_state['collision']:.3f}")
                print(f"        F.contains(T): {F.contains(t_state)}, F.contains(S): {F.contains(s_state)}")
                
                # Print all k-leg components with their liability values
                print(f"        K-leg liabilities: {dict(sorted(k_leg_liab_values.items(), key=lambda x: x[1], reverse=True))}")
                
                # Print all Shapley components with their liability values
                print(f"        Shapley liabilities: {dict(sorted(shapley_values.items(), key=lambda x: x[1], reverse=True))}")
                
                # Verify they sum to 1
                k_leg_sum = sum(k_leg_liab_values.values())
                shapley_sum = sum(shapley_values.values())
                print(f"        Sums - K-leg: {k_leg_sum:.6f}, Shapley: {shapley_sum:.6f}")
                print()
                
            except Exception as e:
                print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f} | Error: {str(e)}")
        else:
            print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f} | No collision")
    
    print(f"\nTotal contexts analyzed: {len(contexts)}")
    print(f"Contexts with collision: {len(k_leg_results)}")
    print(f"Contexts without collision: {len(contexts) - len(k_leg_results)}")
    
    # Analyze results if we have collision cases
    if k_leg_results and shapley_results:
        analyze_liability_differences(k_leg_results, shapley_results)
        plot_liability_comparison(k_leg_results, shapley_results)
        print_liability_summary_table(k_leg_results, shapley_results)
    else:
        print("\nNo collision cases found in the random contexts.")
        print("Try adjusting the context generation to include more dangerous scenarios.")

def analyze_liability_differences(k_leg_results, shapley_results):
    """Analyze differences between k-leg and Shapley liability values."""
    print("\n=== Liability Method Comparison Analysis ===")
    
    # Get all component names
    all_components = set()
    for result in k_leg_results + shapley_results:
        all_components.update(result.keys())
    all_components = sorted(list(all_components))
    
    print(f"Components analyzed: {all_components}")
    
    # Compare average liability values
    print("\nAverage liability values:")
    print("Component | K-leg Avg | Shapley Avg | Difference")
    print("-" * 50)
    
    for comp in all_components:
        k_leg_vals = [result.get(comp, 0) for result in k_leg_results]
        shapley_vals = [result.get(comp, 0) for result in shapley_results]
        
        k_leg_avg = np.mean(k_leg_vals)
        shapley_avg = np.mean(shapley_vals)
        diff = abs(k_leg_avg - shapley_avg)
        
        print(f"{comp:9s} | {k_leg_avg:9.3f} | {shapley_avg:11.3f} | {diff:10.3f}")
    
    # Correlation analysis
    print("\nCorrelation between methods:")
    for comp in all_components:
        k_leg_vals = [result.get(comp, 0) for result in k_leg_results]
        shapley_vals = [result.get(comp, 0) for result in shapley_results]
        
        if len(k_leg_vals) > 1 and np.std(k_leg_vals) > 1e-6 and np.std(shapley_vals) > 1e-6:
            correlation = np.corrcoef(k_leg_vals, shapley_vals)[0, 1]
            print(f"{comp}: correlation = {correlation:.3f}")

def print_liability_summary_table(k_leg_results, shapley_results):
    """Print a summary table with components as columns and methods as rows."""
    print("\n=== Liability Summary Table ===")
    
    # Get all component names
    all_components = set()
    for result in k_leg_results + shapley_results:
        all_components.update(result.keys())
    all_components = sorted(list(all_components))
    
    if not all_components:
        print("No components found in results.")
        return
    
    # Calculate mean values for each component
    k_leg_means = {}
    shapley_means = {}
    
    for comp in all_components:
        k_leg_vals = [result.get(comp, 0) for result in k_leg_results]
        shapley_vals = [result.get(comp, 0) for result in shapley_results]
        
        k_leg_means[comp] = np.mean(k_leg_vals)
        shapley_means[comp] = np.mean(shapley_vals)
    
    # Print table header
    header = "Method    |"
    for comp in all_components:
        header += f" {comp:>12s} |"
    print(header)
    print("-" * len(header))
    
    # Print K-leg row
    k_leg_row = "K-leg     |"
    for comp in all_components:
        k_leg_row += f" {k_leg_means[comp]:12.6f} |"
    print(k_leg_row)
    
    # Print Shapley row
    shapley_row = "Shapley   |"
    for comp in all_components:
        shapley_row += f" {shapley_means[comp]:12.6f} |"
    print(shapley_row)
    
    # Print difference row
    diff_row = "Diff      |"
    for comp in all_components:
        diff = abs(k_leg_means[comp] - shapley_means[comp])
        diff_row += f" {diff:12.6f} |"
    print(diff_row)

def plot_liability_comparison(k_leg_results, shapley_results):
    """Create plots comparing the two liability methods."""
    try:
        # Get all component names
        all_components = set()
        for result in k_leg_results + shapley_results:
            all_components.update(result.keys())
        all_components = sorted(list(all_components))
        
        # Create scatter plots for each component
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, comp in enumerate(all_components[:4]):  # Plot up to 4 components
            if i >= len(axes):
                break
                
            k_leg_vals = [result.get(comp, 0) for result in k_leg_results]
            shapley_vals = [result.get(comp, 0) for result in shapley_results]
            
            axes[i].scatter(k_leg_vals, shapley_vals, alpha=0.7)
            axes[i].plot([0, max(max(k_leg_vals), max(shapley_vals))], 
                        [0, max(max(k_leg_vals), max(shapley_vals))], 'r--', alpha=0.5)
            axes[i].set_xlabel('K-leg Liability')
            axes[i].set_ylabel('Shapley Liability')
            axes[i].set_title(f'Component: {comp}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(all_components), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('liability_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as 'liability_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    except Exception as e:
        print(f"\nError creating plots: {e}")

def focused_collision_analysis():
    """Generate contexts specifically designed to cause collisions."""
    print("\n=== Focused Collision Analysis ===")
    
    system = load_static_aeb_system()
    
    # Create contexts that are more likely to cause collisions
    focused_contexts = [
        {'dist_u': 1.9, 'vel_u': 15.0, 'radar_conf_u': 0.9},  # Close, fast, good radar
        {'dist_u': 1.8, 'vel_u': 12.0, 'radar_conf_u': 0.3},  # Close, moderate speed, poor radar
        {'dist_u': 1.7, 'vel_u': 8.0, 'radar_conf_u': 0.8},   # Very close, moderate speed
        {'dist_u': 1.5, 'vel_u': 20.0, 'radar_conf_u': 0.5},  # Very close, very fast
        {'dist_u': 1.6, 'vel_u': 18.0, 'radar_conf_u': 0.1},  # Close, fast, radar failure
        {'dist_u': 1.4, 'vel_u': 10.0, 'radar_conf_u': 0.7},  # Very close, moderate
        {'dist_u': 1.3, 'vel_u': 14.0, 'radar_conf_u': 0.4},  # Very close, fast, poor radar
        {'dist_u': 1.1, 'vel_u': 16.0, 'radar_conf_u': 0.8},  # Extremely close, fast
    ]
    
    S = create_perfect_aeb_system()
    T = system
    F = ClosedHalfSpaceFailureSet({'collision': (0.1, 'ge')})
    
    print("Focused collision contexts:")
    print("Context | Dist | Vel  | Radar | Collision")
    print("-" * 50)
    
    for i, context in enumerate(focused_contexts):
        state = system.get_state(context)
        collision_value = state['collision']
        
        if collision_value > 0.1:
            try:
                k_leg_values = k_leg_liab(T, S, context, F, k=2)
                shapley_values = shapley_liab(T, S, context, F, k=-1)
                
                print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f}")
                
                # Print all liability values for this context
                print(f"        K-leg liabilities: {dict(sorted(k_leg_values.items(), key=lambda x: x[1], reverse=True))}")
                print(f"        Shapley liabilities: {dict(sorted(shapley_values.items(), key=lambda x: x[1], reverse=True))}")
                
                # Verify they sum to 1
                k_leg_sum = sum(k_leg_values.values())
                shapley_sum = sum(shapley_values.values())
                print(f"        Sums - K-leg: {k_leg_sum:.6f}, Shapley: {shapley_sum:.6f}")
                print()
                
            except Exception as e:
                print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f} | Error: {str(e)}")
        else:
            print(f"{i+1:7d} | {context['dist_u']:4.1f} | {context['vel_u']:4.1f} | {context['radar_conf_u']:5.2f} | {collision_value:9.3f} | No collision")

if __name__ == '__main__':
    try:
        compare_liability_methods()
        focused_collision_analysis()
        
        print("\n=== Analysis Complete ===")
        print("This comparison shows how k-leg liability (k=2) differs from Shapley values")
        print("for the static AEB system across different collision scenarios.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
