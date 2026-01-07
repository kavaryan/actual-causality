"""Calculate the k-leg liability."""
if __name__ == "__main__":
    import sys
    sys.path.append('src/')

import math
from typing import Dict
from itertools import chain, combinations

from sympy import Max
import numpy as np
from core.bf import bf
from core.failure import ClosedHalfSpaceFailureSet, FailureSet
from core.scm import SCMSystem, Component
from subprojects.liab.utils import subsets_upto

def shapley_liab(T: SCMSystem, S: SCMSystem, u: Dict[str, float], F: FailureSet, k: int=-1):
    """Calculate the Shapley liability of each component in T.
    
    For more information see the paper.

    Args:
        T (SCMSystem): The implementation system.
        S (SCMSystem): The specification system.
        u (Dict[str, float]): The context as a dictionary with keys as the variable names.
        F (FailureSet): The failure set.
        k (int, optional): The number of legs to consider. Defaults to -1.
    """
    if not u or not isinstance(u, dict):
        raise ValueError("The context must be a non-empty dictionary.")
    bf_dict = {}

    endogenous_vars = T.endogenous_vars
    liabs = np.zeros(len(endogenous_vars))
    
    for Xi, X in enumerate(endogenous_vars):
        B = []
        for K in subsets_upto(endogenous_vars, len(endogenous_vars)-1):
            if bf(T, S, [X]+list(K), u, F, bf_dict):
                B.append(K)

        if len(B) == 0:
            liabs[Xi] = 0
            continue

        X_shares = np.zeros(len(B))
        for Ki, K in enumerate(B):
            # Create modified system with K variables replaced by specification
            new_components_k = []
            for var, comp in T.components.items():
                if var in K:
                    if var in S.components:
                        new_components_k.append(S.components[var])
                    else:
                        new_components_k.append(comp)
                else:
                    new_components_k.append(comp)
            T_k = SCMSystem(new_components_k, T.domains)
            
            # Create modified system with both K and X variables replaced
            new_components_kx = []
            for var, comp in T.components.items():
                if var in K or var == X:
                    if var in S.components:
                        new_components_kx.append(S.components[var])
                    else:
                        new_components_kx.append(comp)
                else:
                    new_components_kx.append(comp)
            T_kx = SCMSystem(new_components_kx, T.domains)
            
            state_k = T_k.get_state(u)
            state_kx = T_kx.get_state(u)
            
            d = max(0, F.depth(state_k) - F.depth(state_kx))
            # Apply Shapley value weight
            d *= math.factorial(len(K))*math.factorial(len(endogenous_vars)-len(K)-1)/math.factorial(len(endogenous_vars))
            X_shares[Ki] = d
        liabs[Xi] = X_shares.sum()

    if liabs.sum() != 0:
        liabs = liabs/liabs.sum()
    return dict(zip(endogenous_vars, liabs))
