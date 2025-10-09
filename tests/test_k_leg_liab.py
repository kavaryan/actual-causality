"""Test k-leg liability calculation."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sympy import Max
from subprojects.liab.k_leg_liab import k_leg_liab
from core.failure import ClosedHalfSpaceFailureSet
from core.scm import ComponentOrEquation, System


def test_k_leg_liab():
    a_sp = ComponentOrEquation(['a'], 'A', 'a')
    b_sp = ComponentOrEquation(['b'], 'B', 'b')
    c_sp = ComponentOrEquation(['c'], 'C', 'c')
    d_sp = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)')
    a_im = ComponentOrEquation(['a'], 'A', 'a+10')
    b_im = ComponentOrEquation(['b'], 'B', 'b+10')
    c_im = ComponentOrEquation(['c'], 'C', 'c+8')
    d_im = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)+10')

    S = System([a_sp, b_sp, c_sp, d_sp])
    T = System([a_im, b_im, c_im, d_im])

    u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
    F = ClosedHalfSpaceFailureSet({'D': (250, 'ge')})

    assert S.induced_scm().get_state(u)[0] == {'A': 10, 'B': 10, 'C': 10, 'D': 110}
    assert T.induced_scm().get_state(u)[0] == {'A': 20, 'B': 20, 'C': 18, 'D': 420}
    
    liabs = k_leg_liab(T, S, u, F, k=2)
    assert np.isclose(liabs['A'], 0.348, atol=0.01)
    assert np.isclose(liabs['A'], liabs['B'])
    assert np.isclose(liabs['C'], 0.302, atol=0.01)
    assert np.isclose(liabs['D'], 0, atol=0.01)

    print("All tests passed!")


if __name__ == "__main__":
    test_k_leg_liab()
