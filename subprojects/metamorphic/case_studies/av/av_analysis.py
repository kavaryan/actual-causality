#!/usr/bin/env python3
"""
AEB (Automatic Emergency Braking) System Analysis

This script analyzes causes of collisions in a simple AEB system model.
The model includes:
- Initial distance to obstacle
- Vehicle velocity
- Radar confidence
- AEB braking decision
- Physics simulation over 3 time steps
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from core.scm import read_system
from core.hp_modified import find_all_causes, pretty_print_causes
import numpy as np

def load_aeb_system():
    """Load the AEB system from the configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'av.conf')
    system = read_system(config_path)
    system.display_dag('aeb_scm_dag.png')
    return system

def analyze_collision_scenario():
    """Analyze a scenario where collision occurs."""
    print("=== AEB System Collision Analysis ===\n")
    
    system = load_aeb_system()
    
    # Scenario 1: High-speed approach with poor radar detection
    print("Scenario 1: High-speed approach with poor radar detection")
    context1 = {
        'dist_t1_u': 8.0,      # Close initial distance
        'vel_t1_u': 15.0,      # High speed (54 km/h)
        'radar_conf_t1_u': 0.5  # Poor radar confidence
    }
    
    state1 = system.get_state(context1)
    print(f"Initial state: dist={state1['dist_t1']:.1f}m, vel={state1['vel_t1']:.1f}m/s, radar_conf={state1['radar_conf_t1']:.2f}")
    print(f"AEB triggered: {state1['aeb_trigger_t2']:.2f}, Brake force: {state1['brake_force_t2']:.2f}")
    print(f"Final state: dist={state1['dist_t3']:.1f}m, vel={state1['vel_t3']:.1f}m/s")
    print(f"Collision: {state1['collision']:.2f}, Severity: {state1['collision_severity']:.2f}")
    
    if state1['collision'] > 0.5:
        print("\n--- Finding causes of collision ---")
        causes1 = find_all_causes(system, context1, 'collision', '>', 0.5)
        pretty_print_causes(system, context1, causes1)
    
    print("\n" + "="*60 + "\n")
    
    # Scenario 2: Moderate speed with good radar but very close obstacle
    print("Scenario 2: Moderate speed with good radar but very close obstacle")
    context2 = {
        'dist_t1_u': 6.0,      # Very close initial distance
        'vel_t1_u': 12.0,      # Moderate speed (43 km/h)
        'radar_conf_t1_u': 0.9  # Good radar confidence
    }
    
    state2 = system.get_state(context2)
    print(f"Initial state: dist={state2['dist_t1']:.1f}m, vel={state2['vel_t1']:.1f}m/s, radar_conf={state2['radar_conf_t1']:.2f}")
    print(f"AEB triggered: {state2['aeb_trigger_t2']:.2f}, Brake force: {state2['brake_force_t2']:.2f}")
    print(f"Final state: dist={state2['dist_t3']:.1f}m, vel={state2['vel_t3']:.1f}m/s")
    print(f"Collision: {state2['collision']:.2f}, Severity: {state2['collision_severity']:.2f}")
    
    if state2['collision'] > 0.5:
        print("\n--- Finding causes of collision ---")
        causes2 = find_all_causes(system, context2, 'collision', '>', 0.5)
        pretty_print_causes(system, context2, causes2)
    
    print("\n" + "="*60 + "\n")

def analyze_severe_collision_scenario():
    """Analyze scenarios where collision severity is high."""
    print("=== High Severity Collision Analysis ===\n")
    
    system = load_aeb_system()
    
    # Scenario: High speed with radar failure
    print("Scenario: High speed with complete radar failure")
    context = {
        'dist_t1_u': 7.0,      # Moderate initial distance
        'vel_t1_u': 18.0,      # Very high speed (65 km/h)
        'radar_conf_t1_u': 0.2  # Very poor radar confidence (almost blind)
    }
    
    state = system.get_state(context)
    print(f"Initial state: dist={state['dist_t1']:.1f}m, vel={state['vel_t1']:.1f}m/s, radar_conf={state['radar_conf_t1']:.2f}")
    print(f"AEB triggered: {state['aeb_trigger_t2']:.2f}, Brake force: {state['brake_force_t2']:.2f}")
    print(f"Final state: dist={state['dist_t3']:.1f}m, vel={state['vel_t3']:.1f}m/s")
    print(f"Collision: {state['collision']:.2f}, Severity: {state['collision_severity']:.2f}")
    
    if state['collision_severity'] > 0.3:
        print("\n--- Finding causes of high collision severity ---")
        causes = find_all_causes(system, context, 'collision_severity', '>', 0.3)
        pretty_print_causes(system, context, causes)

def explore_parameter_space():
    """Explore different parameter combinations to understand system behavior."""
    print("=== Parameter Space Exploration ===\n")
    
    system = load_aeb_system()
    
    collision_cases = []
    safe_cases = []
    
    # Sample parameter space
    for dist in [6.0, 8.0, 10.0, 12.0]:
        for vel in [10.0, 15.0, 20.0]:
            for radar_conf in [0.3, 0.6, 0.9]:
                context = {
                    'dist_t1_u': dist,
                    'vel_t1_u': vel,
                    'radar_conf_t1_u': radar_conf
                }
                
                state = system.get_state(context)
                
                if state['collision'] > 0.5:
                    collision_cases.append((dist, vel, radar_conf, state['collision_severity']))
                else:
                    safe_cases.append((dist, vel, radar_conf))
    
    print(f"Found {len(collision_cases)} collision cases out of {len(collision_cases) + len(safe_cases)} scenarios")
    print("\nCollision cases (dist, vel, radar_conf, severity):")
    for case in collision_cases:
        print(f"  dist={case[0]:.1f}m, vel={case[1]:.1f}m/s, radar={case[2]:.1f}, severity={case[3]:.3f}")
    
    print(f"\nSafe cases: {len(safe_cases)}")

if __name__ == '__main__':
    try:
        analyze_collision_scenario()
        analyze_severe_collision_scenario()
        explore_parameter_space()
        
        print("\n=== Analysis Complete ===")
        print("The AEB system model demonstrates how:")
        print("1. High initial velocity increases collision risk")
        print("2. Poor radar confidence prevents effective braking")
        print("3. Close initial distance reduces reaction time")
        print("4. The combination of these factors determines collision severity")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
