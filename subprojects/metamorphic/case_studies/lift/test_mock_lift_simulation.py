import unittest
import random
from math import inf
from mock_lift_simulation import MockLiftsSimulator


class TestMockLiftsSimulator(unittest.TestCase):
    
    def test_basic_simulation_functionality(self):
        """Test basic simulation with different numbers of lifts."""
        simulator = MockLiftsSimulator(average_max_time=10.0, speed=1.0)
        
        # Test with 1 lift
        time_1_lift = simulator.simulate(1)
        self.assertGreater(time_1_lift, 0)
        
        # Test with 2 lifts - should be roughly half the time
        simulator_2 = MockLiftsSimulator(average_max_time=10.0, speed=1.0)
        time_2_lifts = simulator_2.simulate(2)
        self.assertGreater(time_2_lifts, 0)
        self.assertLess(time_2_lifts, time_1_lift)
        
        # Test with 5 lifts - should be even less
        simulator_5 = MockLiftsSimulator(average_max_time=10.0, speed=1.0)
        time_5_lifts = simulator_5.simulate(5)
        self.assertGreater(time_5_lifts, 0)
        self.assertLess(time_5_lifts, time_2_lifts)
    
    def test_speed_affects_simulation_time(self):
        """Test that different speeds produce different simulation times."""
        # Test with slow speed
        slow_simulator = MockLiftsSimulator(average_max_time=10.0, speed=0.5)
        slow_time = slow_simulator.simulate(2)
        
        # Test with fast speed
        fast_simulator = MockLiftsSimulator(average_max_time=10.0, speed=2.0)
        fast_time = fast_simulator.simulate(2)
        
        # Fast speed should result in shorter simulation time
        self.assertGreater(slow_time, fast_time)
        
        # Both should be positive
        self.assertGreater(slow_time, 0)
        self.assertGreater(fast_time, 0)
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases like zero lifts and invalid speed."""
        # Test with zero lifts - should return infinity
        simulator = MockLiftsSimulator(average_max_time=10.0)
        time_zero_lifts = simulator.simulate(0)
        self.assertEqual(time_zero_lifts, inf)
        
        # Test invalid speed raises ValueError
        with self.assertRaises(ValueError):
            MockLiftsSimulator(speed=0)
        
        with self.assertRaises(ValueError):
            MockLiftsSimulator(speed=-1.0)
        
        # Test that negative times are prevented
        simulator = MockLiftsSimulator(
            average_max_time=0.001,  # Very small base time
            fluctuation_scale=10.0   # Large negative fluctuations possible
        )
        time_result = simulator.simulate(1)
        self.assertGreater(time_result, 0)
    
    def test_time_tracking_and_reset(self):
        """Test total time tracking and reset functionality."""
        simulator = MockLiftsSimulator(average_max_time=5.0)
        
        # Initially total time should be 0
        self.assertEqual(simulator.get_total_time(), 0.0)
        
        # Run some simulations
        time1 = simulator.simulate(1)
        self.assertEqual(simulator.get_total_time(), time1)
        
        time2 = simulator.simulate(2)
        self.assertEqual(simulator.get_total_time(), time1 + time2)
        
        time3 = simulator.simulate(3)
        expected_total = time1 + time2 + time3
        self.assertEqual(simulator.get_total_time(), expected_total)
        
        # Test reset
        simulator.reset_time()
        self.assertEqual(simulator.get_total_time(), 0.0)
        
        # Verify simulation still works after reset
        time_after_reset = simulator.simulate(1)
        self.assertGreater(time_after_reset, 0)
        self.assertEqual(simulator.get_total_time(), time_after_reset)


if __name__ == '__main__':
    unittest.main()
