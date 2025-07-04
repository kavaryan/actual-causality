import tempfile
import unittest
from unittest import TestCase
from actual.scm import read_system, SCMSystem

class TestSystem(TestCase):
    def test_read_system(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"A = a\nB = A**2\nC = A + B")
            f.flush()
            system = read_system(f.name)
            self.assertIsInstance(system, SCMSystem)
            self.assertEqual(system.get_state({'a': 2}), {'A': 2, 'B': 4, 'C': 6})
        

# Example usage:
if __name__ == "__main__":
    unittest.main()