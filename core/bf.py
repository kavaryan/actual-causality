"""Contains the method to check but-for (BF) causality."""
if __name__ == "__main__":
    import sys
    sys.path.append('src/')

from typing import Union, List, Dict, Tuple, Callable
import numpy as np
from itertools import chain, combinations

from core.scm import SCMSystem, Component
from core.failure import FailureSet

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def bf(T: SCMSystem, S: SCMSystem, X: Union[str,List[str]], 
       u: Dict[str, float], F: FailureSet, bf_dict=None) -> bool:
    """Check but-for causality.
    
    Check if the variable(s) X is a but-for cause of T (with the specification S) 
        ending up in F under the context u.

    Args:
        T (SCMSystem): The implementation system.
        S (SCMSystem): The specification system.
        X (Union[str,List[str]]): The variable name(s) to check.
        u (Dict[str, float]): The context.
        F (FailureSet): The failure set.
        bf_dict (Dict[frozenset, bool], optional): A dictionary to reuse the results of the but-for analysis.
    """
    if not isinstance(X, list):
        assert isinstance(X, str)
        X = [X]
    X = frozenset(X)
    if bf_dict is None:
        bf_dict_ = {}
    else:
        bf_dict_ = bf_dict

    # BF1 axiom (in the paper)
    t_dict = T.get_state(u)
    if not F.contains(t_dict):
        return False
    
    def _rec_bf(Y):
        if Y in bf_dict_: return
        if len(Y) == 0: return
        for Z in powerset(Y):
            if len(Z) == 0 or len(Z) == len(Y):
                continue
            _rec_bf(Z)
            if bf_dict_[Z]:
                bf_dict_[Y] = False
        
        # BF3 axiom (in the paper)
        if Y in bf_dict_:
            assert bf_dict_[Y] == False
            return

        # BF2 axiom (in the paper)
        # Create a new system with variables from Z replaced by their specification values
        new_components = []
        for var, comp in T.components.items():
            if var in Z:
                # Replace with specification component
                if var in S.components:
                    new_components.append(S.components[var])
                else:
                    # If not in S.components, it's an exogenous variable, keep as is
                    new_components.append(comp)
            else:
                new_components.append(comp)
        
        T_modified = SCMSystem(new_components, T.domains)
        z_dict = T_modified.get_state(u)
        bf_dict_[Y] = not F.contains(z_dict)
        
    _rec_bf(X)
    return bf_dict_[X]
    
def test_bf():
    from core.failure import ClosedHalfSpaceFailureSet
    from core.scm import Component, BoundedIntInterval
    
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
        'c': BoundedIntInterval(0, 20)
    }

    S = SCMSystem([a_sp, b_sp, c_sp], domains)
    T = SCMSystem([a_im, b_im, c_im], domains)
    F = ClosedHalfSpaceFailureSet({'C': (250, 'ge')})
    
    assert bf(T, S, 'A', {'a': 10, 'b': 10, 'c': 10}, F)
    assert bf(T, S, 'B', {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, 'C', {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, ['A', 'C'], {'a': 10, 'b': 10, 'c': 10}, F)

    print("All tests passed!")

if __name__ == "__main__":
    test_bf()
