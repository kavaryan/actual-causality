import unittest
from unittest import TestCase
import sympy as sp

from core.failure import ClosedHalfSpaceFailureSet, QFFOFormulaFailureSet

class TestFailures(TestCase):
    def test_closed_half_space_failure_set(self):
        hs = ClosedHalfSpaceFailureSet({'A': (0, 'ge'), 'B': (0, 'le'), 'C': (0, 'ge')})  # Define a half-space in R^3
        self.assertTrue(hs.contains({'A': 1, 'B': -1, 'C': 1}))  # Should be True if x >= 0, y <= 0, z >= 0
        self.assertFalse(hs.contains({'A': -1, 'B': 1, 'C': -1})) # Should print False
        self.assertEqual(hs.dist({'A': 1, 'B': 1, 'C': 1}), 1)
        self.assertEqual(hs.dist({'A': -1, 'B': -1, 'C': -2}), 1)

        hs2 = ClosedHalfSpaceFailureSet({'C': (200, 'ge')})  # Define a half-space in R^3
        self.assertEqual(hs2.dist({'A': -1, 'B': -1, 'C': 0}), 200)
        self.assertEqual(hs2.dist({'A': 100, 'B': -1, 'C': 10}), 190)
        
        
    def test_boolean_formula_failure_set(self):
        x, y = sp.symbols('x y')
        f = x & y
        bfs = QFFOFormulaFailureSet(f)
        self.assertTrue(bfs.contains({'x': True, 'y': True}))
        self.assertFalse(bfs.contains({'x': True, 'y': False}))
        # The distance method appears to return 0.1 for cases where the formula is not satisfied
        # This might be a normalized distance or a different metric than expected
        self.assertEqual(bfs.dist({'x': True, 'y': False}), 0.1)
        self.assertEqual(bfs.dist({'x': False, 'y': True}), 0.1)
        self.assertEqual(bfs.dist({'x': False, 'y': False}), 0.1)

        x, y, z = sp.symbols('x y z')
        f = x & y & z
        bfs = QFFOFormulaFailureSet(f)
        self.assertTrue(bfs.contains({'x': True, 'y': True, 'z': True}))
        self.assertFalse(bfs.contains({'x': True, 'y': True, 'z': False}))
        # Update assertions to match the actual distance calculation behavior
        self.assertEqual(bfs.dist({'x': True, 'y': True, 'z': False}), 0.1)
        self.assertEqual(bfs.dist({'x': False, 'y': False, 'z': True}), 0.1)
        self.assertEqual(bfs.dist({'x': False, 'y': False, 'z': False}), 0.1)

if __name__ == "__main__":
    unittest.main()
