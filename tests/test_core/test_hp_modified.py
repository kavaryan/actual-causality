import tempfile
import textwrap
import unittest
from unittest import TestCase
from core.scm import read_system, SCMSystem
from core.hp_modified import find_all_causes, pretty_print_causes


class TestHPModified(TestCase):
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
    
        causes = find_all_causes(system, {'STu': 0, 'BTu': 1}, 'BS', '==', 1)
        expected_results = [
            {'X_x_prime': {'BT': 0}, 'W': (), 'w': {}},
            {'X_x_prime': {'BT': 0}, 'W': ('SH',), 'w': {'SH': 0}},
            {'X_x_prime': {'BT': 0}, 'W': ('ST',), 'w': {'ST': 0}},
            {'X_x_prime': {'BT': 0}, 'W': ('SH', 'ST'), 'w': {'SH': 0, 'ST': 0}},
            {'X_x_prime': {'BH': 0}, 'W': (), 'w': {}},
            {'X_x_prime': {'BH': 0}, 'W': ('SH',), 'w': {'SH': 0}},
            {'X_x_prime': {'BH': 0}, 'W': ('ST',), 'w': {'ST': 0}},
            {'X_x_prime': {'BH': 0}, 'W': ('BT',), 'w': {'BT': 1}},
            {'X_x_prime': {'BH': 0}, 'W': ('SH', 'ST'), 'w': {'SH': 0, 'ST': 0}},
            {'X_x_prime': {'BH': 0}, 'W': ('SH', 'BT'), 'w': {'SH': 0, 'BT': 1}},
            {'X_x_prime': {'BH': 0}, 'W': ('ST', 'BT'), 'w': {'ST': 0, 'BT': 1}},
            {'X_x_prime': {'BH': 0}, 'W': ('SH', 'ST', 'BT'), 'w': {'SH': 0, 'ST': 0, 'BT': 1}},
            {'X_x_prime': {'ST': 1}, 'W': ('SH',), 'w': {'SH': 0}},
            {'X_x_prime': {'ST': 1}, 'W': ('SH', 'BT'), 'w': {'SH': 0, 'BT': 1}}
        ]

        for e in expected_results:
            self.assertIn(e, causes['results'])
                    

