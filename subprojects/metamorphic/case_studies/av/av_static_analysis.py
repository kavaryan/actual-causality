#!/usr/bin/env python3
"""
Static AEB (Automatic Emergency Braking) System Analysis

This script analyzes causes of collisions in a static AEB system model.
The model includes instantaneous decision making based on:
- Current distance to obstacle
- Current vehicle velocity  
- Radar detection confidence
- Physics-based critical distance calculation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from core.scm import read_system
from core.hp_modified import find_all_causes, pretty_print_causes
import numpy as np

def load_static_aeb_system():
    """Load the static AEB system from the configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'av_static.conf')
    return read_system(config_path)

def analyze_collision_scenarios():
    """Analyze scenarios where collision occurs in the static model."""
    print("=== Static AEB System Collision Analysis ===\n")
    
    system = load_static_aeb_system()
    
    # Scenario 1: High speed, close distance, good radar
    print("Scenario 1: High speed, close distance, good radar")
    context1 = {
        'dist_u': 3.0,         # Very close distance
        'vel_u': 15.0,         # High speed (54 km/h)
        'radar_conf_u': 0.9    # Good radar confidence
    }
    
    state1 = system.get_state(context1)
    print(f"Current state: dist={state1['dist']:.1f}m, vel={state1['vel']:.1f}m/s, radar_conf={state1['radar_conf']:.2f}")
    print(f"Critical distance: {state1['critical_dist']:.1f}m")
    print(f"Distance margin: {state1['dist_margin']:.1f}m")
    print(f"AEB should trigger: {state1['aeb_should_trigger']:.2f}")
    print(f"Collision risk: {state1['collision_risk']:.2f}")
    print(f"Collision: {state1['collision']:.2f}, Severity: {state1['collision_severity']:.2f}")
    
    if state1['collision'] > 0.5:
        print("\n--- Finding causes of collision ---")
        causes1 = find_all_causes(system, context1, 'collision', '>', 0.5)
        pretty_print_causes(system, context1, causes1)
    
    print("\n" + "="*60 + "\n")
    
    # Scenario 2: Moderate speed, very close distance, poor radar
    print("Scenario 2: Moderate speed, very close distance, poor radar")
    context2 = {
        'dist_u': 1.5,         # Extremely close distance
        'vel_u': 10.0,         # Moderate speed (36 km/h)
        'radar_conf_u': 0.3    # Poor radar confidence
    }
    
    state2 = system.get_state(context2)
    print(f"Current state: dist={state2['dist']:.1f}m, vel={state2['vel']:.1f}m/s, radar_conf={state2['radar_conf']:.2f}")
    print(f"Critical distance: {state2['critical_dist']:.1f}m")
    print(f"Distance margin: {state2['dist_margin']:.1f}m")
    print(f"AEB should trigger: {state2['aeb_should_trigger']:.2f}")
    print(f"Collision risk: {state2['collision_risk']:.2f}")
    print(f"Collision: {state2['collision']:.2f}, Severity: {state2['collision_severity']:.2f}")
    
    if state2['collision'] > 0.5:
        print("\n--- Finding causes of collision ---")
        causes2 = find_all_causes(system, context2, 'collision', '>', 0.5)
        pretty_print_causes(system, context2, causes2)
    
    print("\n" + "="*60 + "\n")

def analyze_high_collision_risk():
    """Analyze scenarios with high collision risk."""
    print("=== High Collision Risk Analysis ===\n")
    
    system = load_static_aeb_system()
    
    # Scenario: Within critical distance but not yet colliding
    print("Scenario: Within critical distance with good radar")
    context = {
        'dist_u': 8.0,         # Moderate distance
        'vel_u': 18.0,         # Very high speed (65 km/h)
        'radar_conf_u': 0.8    # Good radar confidence
    }
    
    state = system.get_state(context)
    print(f"Current state: dist={state['dist']:.1f}m, vel={state['vel']:.1f}m/s, radar_conf={state['radar_conf']:.2f}")
    print(f"Critical distance: {state['critical_dist']:.1f}m")
    print(f"Distance margin: {state['dist_margin']:.1f}m")
    print(f"AEB should trigger: {state['aeb_should_trigger']:.2f}")
    print(f"Collision risk: {state['collision_risk']:.2f}")
    print(f"Collision: {state['collision']:.2f}, Severity: {state['collision_severity']:.2f}")
    
    if state['collision_risk'] > 0.7:
        print("\n--- Finding causes of high collision risk ---")
        causes = find_all_causes(system, context, 'collision_risk', '>', 0.7)
        pretty_print_causes(system, context, causes)

def analyze_aeb_failure():
    """Analyze scenarios where AEB should trigger but doesn't."""
    print("=== AEB Failure Analysis ===\n")
    
    system = load_static_aeb_system()
    
    # Scenario: Dangerous situation but AEB doesn't trigger due to poor radar
    print("Scenario: Dangerous situation with radar failure")
    context = {
        'dist_u': 5.0,         # Close distance
        'vel_u': 16.0,         # High speed (58 km/h)
        'radar_conf_u': 0.2    # Very poor radar confidence
    }
    
    state = system.get_state(context)
    print(f"Current state: dist={state['dist']:.1f}m, vel={state['vel']:.1f}m/s, radar_conf={state['radar_conf']:.2f}")
    print(f"Critical distance: {state['critical_dist']:.1f}m")
    print(f"Distance margin: {state['dist_margin']:.1f}m")
    print(f"AEB should trigger: {state['aeb_should_trigger']:.2f}")
    print(f"Collision risk: {state['collision_risk']:.2f}")
    print(f"Collision: {state['collision']:.2f}, Severity: {state['collision_severity']:.2f}")
    
    # Analyze causes of AEB NOT triggering when it should
    if state['aeb_should_trigger'] < 0.5 and state['dist_margin'] < 0:
        print("\n--- Finding causes of AEB failure to trigger ---")
        # We want to find causes of aeb_should_trigger being LOW (< 0.5)
        causes = find_all_causes(system, context, 'aeb_should_trigger', '<', 0.5)
        pretty_print_causes(system, context, causes)

def explore_critical_distance_relationship():
    """Explore how velocity affects critical distance and collision risk."""
    print("=== Critical Distance vs Velocity Analysis ===\n")
    
    system = load_static_aeb_system()
    
    print("Velocity vs Critical Distance relationship:")
    print("Vel (m/s) | Vel (km/h) | Critical Dist (m) | Safe Dist for Collision")
    print("-" * 65)
    
    for vel in [5, 10, 15, 20]:
        context = {
            'dist_u': 10.0,  # Fixed distance for comparison
            'vel_u': float(vel),
            'radar_conf_u': 1.0  # Perfect radar
        }
        
        state = system.get_state(context)
        vel_kmh = vel * 3.6
        safe_dist = 2.0  # Distance below which collision occurs
        
        print(f"{vel:8.0f} | {vel_kmh:9.1f} | {state['critical_dist']:13.1f} | {safe_dist:20.1f}")
    
    print(f"\nNote: Critical distance = vel * 0.8 + vel² / 16")
    print(f"This accounts for reaction time (0.8s) + braking distance at 8 m/s² deceleration")

def parameter_sensitivity_analysis():
    """Analyze sensitivity to different parameters."""
    print("=== Parameter Sensitivity Analysis ===\n")
    
    system = load_static_aeb_system()
    
    collision_cases = []
    high_risk_cases = []
    safe_cases = []
    
    # Sample parameter space more systematically
    distances = [1.5, 3.0, 5.0, 8.0, 12.0, 20.0]
    velocities = [5.0, 10.0, 15.0, 20.0]
    radar_confs = [0.2, 0.5, 0.8, 1.0]
    
    for dist in distances:
        for vel in velocities:
            for radar_conf in radar_confs:
                context = {
                    'dist_u': dist,
                    'vel_u': vel,
                    'radar_conf_u': radar_conf
                }
                
                state = system.get_state(context)
                
                if state['collision'] > 0.5:
                    collision_cases.append((dist, vel, radar_conf, state['collision_severity']))
                elif state['collision_risk'] > 0.7:
                    high_risk_cases.append((dist, vel, radar_conf, state['collision_risk']))
                else:
                    safe_cases.append((dist, vel, radar_conf))
    
    total_cases = len(collision_cases) + len(high_risk_cases) + len(safe_cases)
    print(f"Parameter space analysis ({total_cases} total scenarios):")
    print(f"  Collision cases: {len(collision_cases)}")
    print(f"  High risk cases: {len(high_risk_cases)}")
    print(f"  Safe cases: {len(safe_cases)}")
    
    if collision_cases:
        print(f"\nWorst collision cases (dist, vel, radar_conf, severity):")
        sorted_collisions = sorted(collision_cases, key=lambda x: x[3], reverse=True)
        for case in sorted_collisions[:5]:
            print(f"  dist={case[0]:.1f}m, vel={case[1]:.1f}m/s, radar={case[2]:.1f}, severity={case[3]:.3f}")

if __name__ == '__main__':
    try:
        analyze_collision_scenarios()
        analyze_high_collision_risk()
        analyze_aeb_failure()
        explore_critical_distance_relationship()
        parameter_sensitivity_analysis()
        
        print("\n=== Static Analysis Complete ===")
        print("The static AEB system model demonstrates:")
        print("1. Critical distance depends quadratically on velocity")
        print("2. Radar confidence is crucial for AEB activation")
        print("3. Collision risk increases when distance < critical distance")
        print("4. AEB failure occurs when radar confidence is low despite danger")
        print("5. Collision severity depends on both velocity and proximity")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
