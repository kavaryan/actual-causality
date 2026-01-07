import random
import networkx as nx
import numpy as np
import re
import sympy as sp
from sympy.logic import SOPform
from core.failure import FailureSet, ClosedHalfSpaceFailureSet, QFFOFormulaFailureSet
from core.scm import Component, SCMSystem, BoundedFloatInterval

def all_combs_b(n: int) -> list[tuple[int]]:
    """Generate all combinations of binary numbers with n digits."""
    ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
    ret = [tuple(bool(int(xi)) for xi in x) for x in ret]
    return ret

def all_combs_f(n: int) -> list[tuple[float]]:
    """Generate all combinations of binary numbers with n digits and convert them to floats."""
    ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
    ret = [tuple(float(xi) for xi in x) for x in ret]
    return ret

def get_rand_binary_vec_b(n: int, rnd: np.random.RandomState=None, seed=42):
    """ Get a random binary vector. """
    if not rnd:
        rnd = np.random.RandomState(seed)
    return rnd.choice(a=[0, 1], size=(1, n))[0]

def get_rand_binary_vec_f(n: int, rnd: np.random.RandomState=None, seed=42):
    """ Get a random binary vector. """
    if not rnd:
        rnd = np.random.RandomState(seed)
    return rnd.choice(a=[0.0, 1.0], size=(1, n))[0]

def get_rand_float_vec(n: int, low: float=-100, high: float=100, rnd: np.random.RandomState=None, seed=42):
    """ Get a random (uniformly distributed) float vector. """
    if not rnd:
        rnd = np.random.RandomState(seed)
    return rnd.uniform(low, high, n)

def get_rand_tt_b(n: int, rnd: np.random.RandomState=None, seed=42) -> dict:
    """ Get a random truth table (i.e., a binary function) over n variables. """
    fv = get_rand_binary_vec_b(2**n, rnd=rnd, seed=seed)
    ret = dict(zip(all_combs_b(n), list(fv)))
    return ret

def get_rand_tt_f(n: int, rnd: np.random.RandomState=None, seed=42) -> dict:
    """ Get a random truth table (i.e., a binary function) over n variables. """
    fv = get_rand_binary_vec_f(2**n, rnd=rnd, seed=seed)
    ret = dict(zip(all_combs_f(n), list(fv)))
    return ret

def change_not(input_string: str):
        result = re.sub(r'~(\w+)', r'(not \1)', input_string)
        result = re.sub(r'~\((.*?)\)', r'(not(\1))', result)
        return result

def get_rand_binary_eq(vars_list: list[str], rnd: np.random.RandomState=None, seed=42):
    rand_tt = get_rand_tt_b(len(vars_list), rnd=rnd, seed=seed)
    minterms = [list(k) for k,v in rand_tt.items() if v]
    eq = change_not(str(SOPform(vars_list, minterms)))
    # eq = str(SOPform(vars_list, minterms))
    return eq

def get_rand_linear_eq(vars_list: list[str], rnd: np.random.RandomState=None, seed=42):
    coeffs = get_rand_float_vec(len(vars_list), rnd=rnd, seed=seed)
    eq = '+'.join(f'{c:.2f}*{v}' for c, v in zip(coeffs,vars_list))
    eq = eq.replace('+-', '-')
    return eq


def get_rand_system(N: int, func_type: str, rnd: np.random.RandomState=None, seed=42):
    """ Create a random DAG. N = |U \\cup V|, the total number of variables

    Args:
        N: Number of system variables (includes both exogenous and endogenous)
        func_type: Type of equations: `binary` or `linear`
    
    Returns:
        SCMSystem: A randomly generated SCM system
    """
    assert func_type in ['binary', 'linear']
    G=nx.gnp_random_graph(N,0.5,directed=True,seed=rnd if rnd else seed)
    G=nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    assert nx.is_directed_acyclic_graph(G)

    components = []
    all_vars = set()
    endogenous_vars = set()
    
    for n in G.nodes:
        preds = list(G.predecessors(n))
        vars_list = [f'x_{k}' for k in preds]
        var_name = f'x_{n}'
        all_vars.add(var_name)
        all_vars.update(vars_list)
        
        if len(preds) == 0: # This is an exogenous variable
            continue
            
        endogenous_vars.add(var_name)
        
        if func_type == 'binary':            
            eq = get_rand_binary_eq(vars_list, rnd=rnd, seed=seed)
        elif func_type == 'linear':
            eq = get_rand_linear_eq(vars_list, rnd=rnd, seed=seed)

        component = Component(f"{var_name} = {eq}")
        components.append(component)
    
    # Create domains for all variables
    domains = {}
    for var in all_vars:
        if func_type == 'binary':
            domains[var] = BoundedFloatInterval(0.0, 1.0)
        else:  # linear
            domains[var] = BoundedFloatInterval(-100.0, 100.0)
    
    return SCMSystem(components, domains)

def rerand_system(S: SCMSystem, func_type: str, rnd: np.random.RandomState=None, seed=42):
    """
    Create a new random system with the same structure as S but different equations.
    
    Args:
        S: Original SCMSystem to use as template
        func_type: Type of equations: 'binary' or 'linear'
        rnd: Random state for reproducibility
        seed: Random seed
    
    Returns:
        SCMSystem: New system with same structure but different equations
    """
    assert func_type in ['binary', 'linear']
    
    new_components = []
    for comp in S.components.values():
        if func_type == 'binary':            
            eq = get_rand_binary_eq(comp.input_vars, rnd=rnd, seed=seed)
        elif func_type == 'linear':
            eq = get_rand_linear_eq(comp.input_vars, rnd=rnd, seed=seed)
        
        new_component = Component(f"{comp.output} = {eq}")
        new_components.append(new_component)

    return SCMSystem(new_components, S.domains)


def get_rand_prop(vars_list: list[str], num_syms: int, rnd: np.random.RandomState=None, seed=42):
    if not rnd:
        rnd = np.random.RandomState(seed)
    syms = rnd.choice(vars_list, size=num_syms, replace=False)
    rand_tt = get_rand_tt_b(len(syms), rnd=rnd, seed=seed)
    minterms = [list(k) for k,v in rand_tt.items() if v]
    ret = change_not(str(SOPform(syms, minterms)))
    return ret

def get_fully_qualified_name(cls):
    return f"{cls.__module__}.{cls.__name__}"

def get_rand_failure(vars_list: list[str], failure_type: FailureSet, rnd: np.random.RandomState=None, seed=42):
    if not rnd:
        rnd = np.random.RandomState(seed)

    # `failure_type` is not guaranteed to be the same object as `QFFOFormulaFailureSet` etc, presumably due to
    #   different imports, as testified by `id()`
    if get_fully_qualified_name(failure_type) == get_fully_qualified_name(QFFOFormulaFailureSet):
        for var in vars_list:
            locals()[var] = sp.symbols(var)
        prop = get_rand_prop(vars_list, num_syms=min(1,len(vars_list)//3), rnd=rnd, seed=seed)
        return QFFOFormulaFailureSet(eval(prop))
    
    elif get_fully_qualified_name(failure_type) == get_fully_qualified_name(ClosedHalfSpaceFailureSet):
        boundary_values = rnd.uniform(-90, 90, len(vars_list))
        boundary_types = rnd.choice(['le', 'ge'], len(vars_list), replace=True)
        return ClosedHalfSpaceFailureSet(dict(zip(vars_list, zip(boundary_values, boundary_types))))


def test_all_combs():
    assert all_combs_b(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all_combs_b(3) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

def test_all_combs_f():
    assert all_combs_f(2) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert all_combs_f(3) == [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]

if __name__ == "__main__":
    SEED = 42
    N = 5
    test_all_combs()
    test_all_combs_f()
    
    rnd = np.random.RandomState(SEED)
    S = get_rand_system(N, 'binary', rnd=rnd)
    print('S', S)
    T = rerand_system(S, 'binary', rnd=rnd)
    print('T', T)
    T2 = rerand_system(S, 'binary', rnd=rnd)
    print('T2', T2)

    all_vars = S.vars

    print(get_rand_failure(all_vars, QFFOFormulaFailureSet, rnd=rnd))

    S = get_rand_system(N, 'linear', rnd=rnd)
    print('S', S)
    T = rerand_system(S, 'linear', rnd=rnd)
    print('T', T)
    T2 = rerand_system(S, 'linear', rnd=rnd)
    print('T2', T2)

    print(get_rand_failure(all_vars[0:2], ClosedHalfSpaceFailureSet, rnd=rnd))
    
    print("random_system.py: All tests passed")
