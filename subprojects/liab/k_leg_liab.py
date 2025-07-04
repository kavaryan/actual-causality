"""Calculate the k-leg liability."""
if __name__ == "__main__":
    import sys
    sys.path.append('src/')

from typing import Dict

from sympy import Max
import numpy as np
from liab.bf import bf
from liab.failure import ClosedHalfSpaceFailureSet, FailureSet
from liab.scm import ComponentOrEquation, GSym, System
from liab.utils import subsets_upto

def k_leg_liab(T: System, S: System, u: Dict[GSym, float], F: FailureSet, *, k: int):
    """Calculate the k-leg liability of each component in T.
    
    For more information see the paper.

    Args:
        T (System): The implementation system.
        S (System): The specification system.
        u (Dict[GSym, float]): The context as a dictionary with keys as the variable names.
        F (FailureSet): The failure set.
        k (int, always as keyword argument): The number of legs to consider.

    Return:
        k_leg liability of ENDOGENOUS components as a dict. The length of returned dict is
            not the same as the total number of system variables. 
    """
    if not u or not isinstance(u, dict):
        raise ValueError("The context must be a non-empty dictionary.")
    bf_dict = {}
    M = S.induced_scm()
    N = T.induced_scm(state_order=M.state_order)

    liabs = np.zeros(len(T.cs))
    for Xi, X in enumerate(T.cs):
        B = []
        for K in subsets_upto(T.cs, k-1):
            if bf(T, S, [X]+list(K), u, F, bf_dict):
                B.append(K)

        if len(B) == 0:
            liabs[Xi] = 0
            continue

        X_shares = np.zeros(len(B))
        for Ki, K in enumerate(B):
            rep_dict = {kc.O: M.cs_dict[kc.O] for kc in K}
            rep_dict2 = dict(rep_dict)
            rep_dict2[X.O] = M.cs_dict[X.O]
            d = max(0,  F.depth(T.get_replacement(rep_dict).induced_scm().get_state(u)[0]) -
                        F.depth(T.get_replacement(rep_dict2).induced_scm().get_state(u)[0]))
                        
            X_shares[Ki] = d
        liabs[Xi] = X_shares.mean()

    keys = [c.O for c in T.cs]
    if liabs.sum() != 0:
        liabs = liabs/liabs.sum()
    return dict(zip(keys, liabs))


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
    import sys
    sys.path.append('src/')
    test_k_leg_liab()