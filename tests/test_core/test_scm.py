import tempfile
import textwrap
import unittest
from unittest import TestCase
from core.scm import read_system, SCMSystem

class TestSCMSystem(TestCase):
    def test_read_system(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(textwrap.dedent("""\
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


    def test_suzy_billy(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(textwrap.dedent("""\
               [equations]
                ST = STu
                BT = BTu

                # Suzy's rock hits if she throws.
                SH = ST

                # Billy's rock hits only if he throws and Suzy doesn't throw.
                BH = BT * (1 - ST)

                # The bottle is shattered if either rock hits.
                BS = min(1, SH + BH)

                [domains]
                ST,BT,SH,BH,BS: Int(0,1)
                """))
            f.flush()
            system = read_system(f.name)

        for STu, BTu in [[0,0],[0,1],[1,0],[1,1]]:
            state = system.get_state({'STu': STu, 'BTu': BTu})
            self.assertEqual(state['BS'], min(1, STu + BTu))

        # Test interventions
        self.assertEqual(system.get_state({'STu': 0, 'BTu': 1}, interventions={'SH': 1})['BS'], 1)

        # Test replacement of equations
        system2 = system.replace('BT=1-ST')
        state2 = system2.get_state({'ST': 1})
        self.assertEqual(state2, {'ST': 1, 'BT': 0, 'SH': 1, 'BH': 0, 'BS': 1})


                

# Example usage:
if __name__ == "__main__":
    unittest.main()