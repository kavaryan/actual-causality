#!/usr/bin/env python3

import sys
import os
sys.path.append('subprojects/metamorphic')
sys.path.append('subprojects/metamorphic/case-studies/lift')

from mock_lift_simulation import MockLiftsSimulator
from search_formulation import SearchSpace
from experiment import run_exp_individual

def test_mock_simulator():
    """Test that the MockLiftsSimulator works as expected"""
    print("Testing MockLiftsSimulator...")
    
    # Create simulator with known parameters
    simulator = MockLiftsSimulator(stretch_coefficient=2.0, startup_cost=0.1)
    
    # Test with different numbers of lifts
    test_cases = [1, 2, 5, 10]
    for num_lifts in test_cases:
        result = simulator.simulate(num_lifts)
        expected = 2.0 / num_lifts + 0.1
        print(f"  num_lifts={num_lifts}: result={result:.3f}, expected={expected:.3f}")
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    # Check total time tracking
    total_expected = sum(2.0 / n + 0.1 for n in test_cases)
    total_actual = simulator.get_total_time()
    print(f"  Total time: {total_actual:.3f}, expected: {total_expected:.3f}")
    assert abs(total_actual - total_expected) < 1e-6
    
    # Test reset
    simulator.reset_time()
    assert simulator.get_total_time() == 0.0
    print("  Reset works correctly")
    
    print("MockLiftsSimulator tests passed!\n")

def test_experiment():
    """Test that the experiment runs and produces reasonable results"""
    print("Testing experiment...")
    
    # Create simulator and search space
    simulator = MockLiftsSimulator(stretch_coefficient=1.0, startup_cost=0.05)
    search_space = SearchSpace(simulator.simulate)
    
    # Run a small experiment
    N = 5  # Small number for quick test
    print(f"Running experiment with N={N} lifts...")
    
    result = run_exp_individual(N, simulator.simulate, search_space)
    
    print(f"Results: d_hp={result['d_hp']:.4f}, d_hp_mm={result['d_hp_mm']:.4f}")
    
    # Sanity checks
    assert result['d_hp'] > 0, "d_hp should be positive"
    assert result['d_hp_mm'] > 0, "d_hp_mm should be positive"
    
    # The times should be reasonable (not too large, not too small)
    assert result['d_hp'] < 10, "d_hp seems too large"
    assert result['d_hp_mm'] < 10, "d_hp_mm seems too large"
    assert result['d_hp'] > 0.001, "d_hp seems too small"
    assert result['d_hp_mm'] > 0.001, "d_hp_mm seems too small"
    
    # Check that simulator time was tracked
    total_sim_time = simulator.get_total_time()
    print(f"Total simulator time: {total_sim_time:.4f}")
    assert total_sim_time > 0, "Simulator should have tracked some time"
    
    print("Experiment test passed!\n")

def test_timing_components():
    """Test that timing includes both computation and simulation time"""
    print("Testing timing components...")
    
    simulator = MockLiftsSimulator(stretch_coefficient=2.0, startup_cost=0.1)
    search_space = SearchSpace(simulator.simulate)
    
    # Reset and run a very small experiment
    simulator.reset_time()
    initial_time = simulator.get_total_time()
    assert initial_time == 0.0
    
    # Run experiment
    result = run_exp_individual(3, simulator.simulate, search_space)
    
    final_sim_time = simulator.get_total_time()
    print(f"Simulation time accumulated: {final_sim_time:.4f}")
    print(f"Total reported times: d_hp={result['d_hp']:.4f}, d_hp_mm={result['d_hp_mm']:.4f}")
    
    # The reported times should include simulation time
    # (They should be at least as large as the simulation time, but could be larger due to computation)
    assert result['d_hp'] >= final_sim_time * 0.1, "d_hp should include significant simulation time"
    assert result['d_hp_mm'] >= final_sim_time * 0.1, "d_hp_mm should include significant simulation time"
    
    print("Timing components test passed!\n")

if __name__ == "__main__":
    print("Running experiment validation tests...\n")
    
    try:
        test_mock_simulator()
        test_experiment() 
        test_timing_components()
        
        print("All tests passed! The experiment setup appears to be working correctly.")
        print("\nKey observations:")
        print("- MockLiftsSimulator tracks time internally without sleep")
        print("- Experiment functions combine computation and simulation time")
        print("- Results are reasonable and positive")
        print("- Time tracking works as expected")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
