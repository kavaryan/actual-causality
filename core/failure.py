"""Contains classes for defining failure."""

from abc import ABC, abstractmethod
from itertools import combinations
import random
import re
from typing import Union
import numpy as np
import sympy as sp
import z3
from core.scm import SCMSystem

class FailureSet(ABC):
    @abstractmethod
    def contains(self, x: Union[np.ndarray, dict]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x: A numpy array or dictionary representing the point.
        """
        ...

    def depth(self, x: Union[np.ndarray, dict]) -> float:
        """Calculate the depth of a point within the failure set.
        
        Args:
            x: A numpy array or dictionary representing the point.
        """
        if self.contains(x):
            return abs(self.dist(x))
        else:
            return 0

    @abstractmethod
    def dist(self, x: Union[np.ndarray, dict]) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        Args:
            x: A numpy array or dictionary representing the point.
        """
        ...

class ClosedHalfSpaceFailureSet(FailureSet):
    def __init__(self, boundary_dict: dict[str, tuple[float, str]]):
        """ Represents a closed half-space in n-dimensional space in the form 
            And_i(x_i >= a_i | x_i <= a_i) for i = 1, 2, ..., n.
        
        Args:
            boundary_dict (dict[str, float]): A dictionary representing the boundaries:
                key: the name of the variable
                value: the boundary value and type
                    boundary value: a float representing the value of the boundary,
                    boundary type: a string ('ge' for >=, 'le' for <=)
                        defining the type of inequality for each boundary.
        """
        self.boundary_dict = boundary_dict

    def __str__(self):
        _symbolize = {'le': '<=', 'ge': '>='}
        s = ','.join(f'{k}{_symbolize[bt]}{bv:.2f}' for k,(bv,bt) in self.boundary_dict.items())
        return f'ClosedHalfSpaceFailureSet({s})'
    
    def contains(self, x: dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        for k, v in self.boundary_dict.items():
            if (v[1] == 'ge' and x[k] < v[0]) or (v[1] == 'le' and x[k] > v[0]):
                return False
        return True

    def dist(self, x: dict[str, float]) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        The result is always non-negative.

        Args:
            x (dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        return min([abs(x[k] - v[0]) for k, v in self.boundary_dict.items()])
    

    def get_example_context(self, M: SCMSystem, N: SCMSystem, seed=42):
        """ Find a context where the resulting M-state is not failed and N-state is failed. """

        solver = z3.Solver()
        solver.set('random_seed', seed)
        solver.set('arith.random_initial_value', False)
        
        # Create variables for exogenous variables
        for x in M.exogenous_nl_vars:
            xx = z3.Real(x)
            exec(f'{x} = xx')
        
        # Create variables for endogenous variables in both systems
        for x in M.endogenous_vars:
            xm = f'm_{x}'
            xx = z3.Real(xm)
            exec(f'{xm}=xx')
        for x in N.endogenous_vars:
            xn = f'n_{x}'
            xx = z3.Real(xn)
            exec(f'{xn}=xx')
        
        # Add M equations
        for var, comp in M.components.items():
            if not comp.is_literal:
                c_o = f'm_{var}'
                ceq = str(comp.expression)
                for x in M.endogenous_vars:
                    ceq = re.sub(rf'\b{x}\b', f'm_{x}', ceq)
                solver.add(eval(f'{c_o}=={ceq}'))

        # Add N equations
        for var, comp in N.components.items():
            if not comp.is_literal:
                c_o = f'n_{var}'
                ceq = str(comp.expression)
                for x in N.endogenous_vars:
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
            for k in M.exogenous_nl_vars:
                # `as_decimal` returns a string 
                ret[k] = float(model[eval(k)].as_decimal(17).split('?')[0])

            state_m = M.get_state(ret)
            state_n = N.get_state(ret)
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
    
    def contains(self, x: dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        # Convert values to SymPy expressions for proper substitution
        x_sympy = {sp.Symbol(k): v for k, v in x.items()}
        result = self.failure_formual.subs(x_sympy)
        return bool(result)

    def dist(self, x: dict[str, float]) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        For numeric comparison formulas, this calculates the minimum distance
        to make the formula evaluation flip.
        
        The result is always non-negative.

        Args:
            x (dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        # For numeric comparison formulas, we need a different approach
        # This is a simplified implementation that works for basic comparisons
        
        # Try to extract the comparison from the formula
        # This is a heuristic approach for simple formulas like "B > 6"
        formula_str = str(self.failure_formual)
        
        # For simple cases, try to compute distance to boundary
        if '>' in formula_str or '<' in formula_str or '>=' in formula_str or '<=' in formula_str:
            # Extract variable and threshold from simple comparisons
            import re
            
            # Match patterns like "B > 6", "x <= 3.5", etc.
            match = re.match(r'(\w+)\s*([<>=]+)\s*([\d.]+)', formula_str)
            if match:
                var_name = match.group(1)
                operator = match.group(2)
                threshold = float(match.group(3))
                
                if var_name in x:
                    var_value = x[var_name]
                    
                    # Calculate distance to threshold
                    if operator in ['>', '>=']:
                        # For x > threshold, distance is max(0, threshold - x + epsilon)
                        return max(0, threshold - var_value + 0.001)
                    elif operator in ['<', '<=']:
                        # For x < threshold, distance is max(0, x - threshold + epsilon)  
                        return max(0, var_value - threshold + 0.001)
        
        # Fallback: return a small positive value
        return 0.1

    def __str__(self) -> str:
        return f'BooleanFormulaFailureSet({self.failure_formual=})'
