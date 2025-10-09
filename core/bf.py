"""Contains the method to check but-for (BF) causality."""
if __name__ == "__main__":
    import sys
    sys.path.append('src/')

from typing import Union, List, Dict, Tuple, Callable
import numpy as np
from itertools import chain, combinations

from core.scm import ComponentOrEquation, GSym, System, SCM
from core.failure import FailureSet

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def bf(T: System, S: System, X: Union[ComponentOrEquation,List[ComponentOrEquation]], 
       u: Dict[GSym, float], F: FailureSet, bf_dict=None) -> bool:
    """Check but-for causality.
    
    Check if the component(s) X is a but-for cause of T (with the specification S) 
        ending up in F under the context u.

    Args:
        T (System): The implementation system.
        S (System): The specification system.
        X (Union[ComponentOrEquation,List[ComponentOrEquation]]): The component(s) to check.
        u (np.ndarray): The context.
        F (FailureSet): The failure set.
        bf_dict (Dict[frozenset, bool], optional): A dictionary to reuse the results of the but-for analysis.
    """
    assert T.is_compatible(S)
    M = S.induced_scm()
    N = T.induced_scm(state_order=M.state_order)
    if not isinstance(X, list):
        assert isinstance(X, ComponentOrEquation)
        X = [X]
    X = frozenset(X)
    if bf_dict is None:
        bf_dict_ = {}
    else:
        bf_dict_ = bf_dict

    # BF1 axiom (in the paper)
    t_dict, _ = N.get_state(u)
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
        Z_rep_dict = {c.O: M.cs_dict[c.O] for c in Z}
        z_dict, _  = T.get_replacement(Z_rep_dict).induced_scm(state_order=M.state_order).get_state(u)
        bf_dict_[Y] = not F.contains(z_dict)
        
    _rec_bf(X)
    return bf_dict_[X]
    
def test_bf():
    from failure import ClosedHalfSpaceFailureSet
    a_sp = ComponentOrEquation(['a'], 'A', 'a')
    b_sp = ComponentOrEquation(['b'], 'B', 'b')
    c_sp = ComponentOrEquation(['c', 'A', 'B'], 'C', 'c+A*B')
    a_im = ComponentOrEquation(['a'], 'A', 'a+10')
    b_im = ComponentOrEquation(['b'], 'B', 'b+10')
    c_im = ComponentOrEquation(['c', 'A', 'B'], 'C', 'c+A*B+10')

    S = System([a_sp, b_sp, c_sp])
    T = System([a_im, b_im, c_im])
    F = ClosedHalfSpaceFailureSet({'C': (250, 'ge')})
    
    assert bf(T, S, a_im, {'a': 10, 'b': 10, 'c': 10}, F)
    assert bf(T, S, b_sp, {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, c_sp, {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, [a_sp, c_sp], {'a': 10, 'b': 10, 'c': 10}, F)

    print("All tests passed!")

if __name__ == "__main__":
    test_bf()