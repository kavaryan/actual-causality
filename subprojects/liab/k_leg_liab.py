"""Calculate the k-leg liability."""
if __name__ == "__main__":
    import sys
    sys.path.append('src/')

from typing import Dict

from sympy import Max
import numpy as np
from core.bf import bf
from core.failure import ClosedHalfSpaceFailureSet, FailureSet
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


if __name__ == "__main__":
    import sys
    sys.path.append('src/')
    from tests.test_k_leg_liab import test_k_leg_liab
    test_k_leg_liab()
