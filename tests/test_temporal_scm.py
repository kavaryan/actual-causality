import unittest
from core.scm import read_temporal_system_str


class TestTemporalSCMSystem(unittest.TestCase):
    def test_read_temporal_system_and_expand(self):
        """Test reading a temporal system config and expanding to SCM."""
        config_str = """
    [differential_equations]
    dX/dt = -X + Y

    [equations]
    Y = 2

    [domains]
    X: Float(-10.0, 10.0)
    Y: Float(-10.0, 10.0)
    """
        
        # Read the temporal system
        temporal_system = read_temporal_system_str(config_str, temporal_expansion_window_width=2, delta=0.1)
        
        # Check temporal system properties
        assert temporal_system.temporal_expansion_window_width == 2
        assert temporal_system.delta == 0.1
        assert "X" in temporal_system.differential_components
        assert "Y" in temporal_system.components
        
        # Expand to regular SCM
        expanded_scm = temporal_system.expand_to_scm()
        
        # Check expanded variables
        expected_vars = {"X_0", "X_1", "X_2", "Y"}
        assert set(expanded_scm.vars) == expected_vars
        
        # Check that X_0 is exogenous and others are endogenous
        assert "X_0" in expanded_scm.exogenous_vars
        assert "Y" in expanded_scm.exogenous_vars
        assert "X_1" in expanded_scm.endogenous_vars
        assert "X_2" in expanded_scm.endogenous_vars
        
        # Test evaluation
        context = {"X_0": 1.0}
        state = expanded_scm.get_state(context)
        
        # Y = 2 (constant)
        assert state["Y"] == 2
        
        # X_1 = X_0 + 0.1 * (-X_0 + Y) = 1.0 + 0.1 * (-1.0 + 2) = 1.1
        assert abs(state["X_1"] - 1.1) < 1e-10
        
        # X_2 = X_1 + 0.1 * (-X_1 + Y) = 1.1 + 0.1 * (-1.1 + 2) = 1.19
        assert abs(state["X_2"] - 1.19) < 1e-10

if __name__ == "__main__":
    unittest.main()
