"""Test k-leg liability calculation."""
import unittest
from unittest import TestCase
import numpy as np

from subprojects.liab.k_leg_liab import k_leg_liab
from core.failure import ClosedHalfSpaceFailureSet
from core.scm import SCMSystem, Component, BoundedIntInterval

class TestKLegLiab(TestCase):
    def test_k_leg_liab(self):
        # Create specification system components
        a_sp = Component('A = a')
        b_sp = Component('B = b')
        c_sp = Component('C = c')
        d_sp = Component('D = d + Max(A*B, A*C, B*C)')
        
        # Create implementation system components
        a_im = Component('A = a + 10')
        b_im = Component('B = b + 10')
        c_im = Component('C = c + 8')
        d_im = Component('D = d + Max(A*B, A*C, B*C) + 10')

        # Create domains
        domains = {
            'a': BoundedIntInterval(0, 20),
            'b': BoundedIntInterval(0, 20),
            'c': BoundedIntInterval(0, 20),
            'd': BoundedIntInterval(0, 20),
            'A': BoundedIntInterval(0, 30),  # a or a+10, so 0-30
            'B': BoundedIntInterval(0, 30),  # b or b+10, so 0-30
            'C': BoundedIntInterval(0, 28),  # c or c+8, so 0-28
            'D': BoundedIntInterval(0, 1000) # d + Max(A*B, A*C, B*C) + possibly 10, needs large range
        }

        S = SCMSystem([a_sp, b_sp, c_sp, d_sp], domains)
        T = SCMSystem([a_im, b_im, c_im, d_im], domains)

        u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
        F = ClosedHalfSpaceFailureSet({'D': (250, 'ge')})

        self.assertEqual(S.get_state(u), {'A': 10, 'B': 10, 'C': 10, 'D': 110, 'a': 10, 'b': 10, 'c': 10, 'd': 10})
        self.assertEqual(T.get_state(u), {'A': 20, 'B': 20, 'C': 18, 'D': 420, 'a': 10, 'b': 10, 'c': 10, 'd': 10})
        
        liabs = k_leg_liab(T, S, u, F, k=2)
        print("2-leg liabilities:", liabs)
        self.assertTrue(np.isclose(liabs['A'], 0.348, atol=0.01))
        self.assertTrue(np.isclose(liabs['A'], liabs['B']))
        self.assertTrue(np.isclose(liabs['C'], 0.302, atol=0.01))
        self.assertTrue(np.isclose(liabs['D'], 0, atol=0.01))



if __name__ == "__main__":
    unittest.main()
