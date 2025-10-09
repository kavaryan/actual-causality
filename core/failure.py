"""Contains classes for defining failure."""

from abc import ABC, abstractmethod
from itertools import combinations
import random
import re
from typing import Union, List, Dict, Tuple, Callable
import numpy as np
import sympy as sp
import z3
from core.scm import SCMSystem

class FailureSet(ABC):
    @abstractmethod
    def contains(self, x: np.ndarray) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        ...

    def depth(self, x: np.ndarray) -> float:
        """Calculate the depth of a point within the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        if self.contains(x):
            return abs(self.dist(x))
        else:
            return 0

    @abstractmethod
    def dist(self, x: np.ndarray) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        ...

class ClosedHalfSpaceFailureSet(FailureSet):
    def __init__(self, bounadry_dict: Dict[str, Tuple[float, str]]):
        """ Represents a closed half-space in n-dimensional space in the form 
            And_i(x_i >= a_i | x_i <= a_i) for i = 1, 2, ..., n.
        
        Args:
            boundary_dict (Dict[str, float]): A dictionary representing the boundaries:
                key: the name of the variable
                value: the boundary value and type
                    boundary value: a float representing the value of the boundary,
                    boundary type: a string ('ge' for >=, 'le' for <=)
                        defining the type of inequality for each boundary.
        """
        self.boundary_dict = bounadry_dict

    def __str__(self):
        _symbolize = {'le': '<=', 'ge': '>='}
        s = ','.join(f'{k}{_symbolize[bt]}{bv:.2f}' for k,(bv,bt) in self.boundary_dict.items())
        return f'ClosedHalfSpaceFailureSet({s})'
    
    def contains(self, x: Dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        for k, v in self.boundary_dict.items():
            if (v[1] == 'ge' and x[k] < v[0]) or (v[1] == 'le' and x[k] > v[0]):
                return False
        return True

    def dist(self, x: Dict[str, float]) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        The result is always non-negative.

        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        return min([abs(x[k] - v[0]) for k, v in self.boundary_dict.items()])
    

    def get_example_context(self, M: SCMSystem, N: SCMSystem, seed=42):
        """ Find a context where the resulting M-state is not failed and N-state is failed. """

        solver = z3.Solver()
        solver.set('random_seed', seed)
        solver.set('arith.random_initial_value', False)
        
        # Create variables
        for x in M.U:
            xx = z3.Real(x)
            exec(f'{x} = xx')
        for x in M.V:
            xm = f'm_{x}'
            xx = z3.Real(xm)
            exec(f'{xm}=xx')
        for x in N.V:
            xn = f'n_{x}'
            xx = z3.Real(xn)
            exec(f'{xn}=xx')
        
        # Add M equations
        for c_o, c in M.cs_dict.items():
            c_o = f'm_{c_o}'
            ceq = c.f
            for x in M.V:
                ceq = re.sub(rf'\b{x}\b', f'm_{x}', ceq)
            solver.add(eval(f'{c_o}=={ceq}'))

        # Add N equations
        for c_o, c in N.cs_dict.items():
            c_o = f'n_{c_o}'
            ceq = c.f
            for x in M.V:
                ceq = re.sub(rf'\b{x}\b', f'n_{x}', ceq)
            solver.add(eval(f'{c_o}=={ceq}'))

        # Add non-failure for M constraint 
        or_args = []
        for k, v in self.boundary_dict.items():
            or_args.append(eval(f'm_{k} < {v[0]}' if v[1] == 'ge' else f'm_{k} > {v[0]}'))
            solver.add(eval(f'n_{k} > {v[0]}' if v[1] == 'ge' else f'n_{k} < {v[0]}'))
        
        solver.add(z3.Or(or_args))
        ret = None
        if solver.check() == z3.sat:
            model = solver.model()
            ret = {}
            for k in M.U:
                # `as_decimal` returns a string 
                ret[k] = float(model[eval(k)].as_decimal(17).split('?')[0])

            state_m, _ = M.get_state(ret)
            state_n, _ = N.get_state(ret)
            if self.contains(state_m) or not self.contains(state_n):
                # assert False
                ...
        return ret


class QFFOFormulaFailureSet(FailureSet):
    def __init__(self, failure_formual: sp.Basic):
        """ Represents a failure set defined by a quantifier-free first-order formula.
        
        Args:
            failure_formual (sp.FunctionClass): A sympy function representing the failure set.
        
        Example:
            >>> from sympy import symbols
            >>> x, y = symbols('x y')
            >>> f = x & y
            >>> bfs = BooleanFormulaFailureSet(f)
        """
        if not isinstance(failure_formual, sp.Basic):
            raise ValueError("The failure_formual must be a sympy object.")
        self.failure_formual = failure_formual
        self.vars_order = list(str(x) for x in failure_formual.free_symbols)
    
    @staticmethod
    def get_random(vars: list[str], seed: int = 42):
        syms = [sp.symbols(v) for v in vars]
        bf = syms[0]
        rnd = random.Random(seed)
        for i in range(1,len(vars)):
            op = rnd.choice([sp.And, sp.Or])
            do_not = rnd.choice([True, False])
            if do_not:
                bf = op(bf, sp.Not(syms[i]))
            else:
                bf = op(bf, syms[i])

        return QFFOFormulaFailureSet(bf)
    
    def contains(self, x: Dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        x = {k: bool(v) for k, v in x.items()}
        return bool(self.failure_formual.subs(x))

    def dist(self, x: Dict[str, float]) -> float:
        """Calculate the Hamming distance of a point to the boundary of the failure set.
        
        The result is always non-negative.

        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        initial_result = self.contains(x)
        n = len(self.vars_order)
        for changes in range(1, n+1):
            for combo in combinations(self.vars_order, changes):
                toggled_x = x.copy()
                for var in combo:
                    toggled_x[var] = not toggled_x[var]
                if self.contains(toggled_x) != initial_result:
                    return changes
        return n  # In the worst case, all variables need to be toggled
    
    def __str__(self) -> str:
        return f'BooleanFormulaFailureSet({self.failure_formual=})'

def test_closed_half_space_failure_set():
    hs = ClosedHalfSpaceFailureSet({'A': (0, 'ge'), 'B': (0, 'le'), 'C': (0, 'ge')})  # Define a half-space in R^3
    assert hs.contains({'A': 1, 'B': -1, 'C': 1})  # Should be True if x >= 0, y <= 0, z >= 0
    assert not hs.contains({'A': -1, 'B': 1, 'C': -1}) # Should print False
    assert hs.dist({'A': 1, 'B': 1, 'C': 1}) == 1
    assert hs.dist({'A': -1, 'B': -1, 'C': -2}) == 1

    hs2 = ClosedHalfSpaceFailureSet({'C': (200, 'ge')})  # Define a half-space in R^3
    assert hs2.dist({'A': -1, 'B': -1, 'C': 0}) == 200
    assert hs2.dist({'A': 100, 'B': -1, 'C': 10}) == 190
    
    
def test_boolean_formula_failure_set():
    x, y = sp.symbols('x y')
    f = x & y
    bfs = QFFOFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True})
    assert not bfs.contains({'x': True, 'y': False})
    assert 1 == bfs.dist({'x': True, 'y': False})
    assert 1 == bfs.dist({'x': False, 'y': True})
    assert 2 == bfs.dist({'x': False, 'y': False})

    x, y, z = sp.symbols('x y z')
    f = x & y & z
    bfs = QFFOFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True, 'z': True})
    assert not bfs.contains({'x': True, 'y': True, 'z': False})
    assert 1 == bfs.dist({'x': True, 'y': True, 'z': False})
    assert 2 == bfs.dist({'x': False, 'y': False, 'z': True})
    assert 3 == bfs.dist({'x': False, 'y': False, 'z': False})

if __name__ == "__main__":
    test_closed_half_space_failure_set()
    test_boolean_formula_failure_set()
    print("All tests passed!")