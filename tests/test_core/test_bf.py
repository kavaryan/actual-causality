import unittest
from unittest import TestCase
from core.bf import bf
from core.scm import SCMSystem
from core.failure import ClosedHalfSpaceFailureSet
from core.scm import Component, BoundedIntInterval

class TestBF(TestCase):
    def test_bf(self):    
        # Create specification system components
        a_sp = Component('A = a')
        b_sp = Component('B = b')
        c_sp = Component('C = c + A*B')
        
        # Create implementation system components  
        a_im = Component('A = a + 10')
        b_im = Component('B = b + 10')
        c_im = Component('C = c + A*B + 10')

        # Create domains
        domains = {
            'a': BoundedIntInterval(0, 20),
            'b': BoundedIntInterval(0, 20), 
            'c': BoundedIntInterval(0, 20),
            'A': BoundedIntInterval(0, 40),  # a can be 0-20, so A = a or a+10 can be 0-30
            'B': BoundedIntInterval(0, 40),  # b can be 0-20, so B = b or b+10 can be 0-30  
            'C': BoundedIntInterval(0, 1000) # C depends on A*B + c + possibly 10, so needs larger range
        }

        S = SCMSystem([a_sp, b_sp, c_sp], domains)
        T = SCMSystem([a_im, b_im, c_im], domains)
        F = ClosedHalfSpaceFailureSet({'C': (250, 'ge')})
        
        self.assertTrue(bf(T, S, 'A', {'a': 10, 'b': 10, 'c': 10}, F))
        self.assertTrue(bf(T, S, 'B', {'a': 10, 'b': 10, 'c': 10}, F))
        self.assertFalse(bf(T, S, 'C', {'a': 10, 'b': 10, 'c': 10}, F))
        self.assertFalse(bf(T, S, ['A', 'C'], {'a': 10, 'b': 10, 'c': 10}, F))


if __name__ == "__main__":
    unittest.main()
