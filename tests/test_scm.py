import tempfile
import textwrap
import unittest
from unittest import TestCase
from core.scm import read_system, SCMSystem

class TestSystem(TestCase):
    def test_read_system(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(textwrap.dedent(""" \
                [equations]
                A = a
                B = A**2
                C = A + B
                """))
            f.flush()
            system = read_system(f.name)
            self.assertIsInstance(system, SCMSystem)
            actual_state = system.get_state({'a': 2})
            expected_state = {'A': 2, 'B': 4, 'C': 6, 'a': 2}
            # self.assertLessEqual(set(expected_state.items()), set(actual_state.items()))
            self.assertEqual(actual_state, expected_state)
        

# Example usage:
if __name__ == "__main__":
    unittest.main()