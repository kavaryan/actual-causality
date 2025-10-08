from itertools import chain, combinations, product
import random
import time

from tqdm import tqdm
from core.scm import SCMSystem, read_system

'''TODO:
- [] Use Z3 for finding the counterexamples
- [] Use dynamic programming
- [] more realistic models
- [] time analysis
- [] attache models
- [] Shapley value's higher dimensions
'''

def find_all_causes(scm: SCMSystem, context: dict[str,object], Y: str, op: str , y_thr: object, include_exo=False):
    """
    TODO: support complex formulas for effect
    """
    results = {}
    
    start_time = time.time()
    all_causes_list = find_all_causes_ac1_and_ac2(scm, context, Y, op, y_thr, include_exo=include_exo)
    results["ac2_time"] = time.time() - start_time

    start_time = time.time()
    all_causes_list_checked_ac3 = check_ac3(all_causes_list)
    results["ac3_time"] = time.time() - start_time
    
    results['results'] = all_causes_list_checked_ac3
    return results

def check_ac3(all_causes_list):
    all_causes_set = {frozenset(d['X_x_prime'].keys()) for d in all_causes_list}
    all_causes = list(map(frozenset, sorted(all_causes_set, key=len)))
    found = True
    while found:
        found = False
        for i in range(len(all_causes)):
            for j in range(i+1, len(all_causes)):
                if all_causes[i] < all_causes[j]: # set comparision
                    all_causes.pop(j)
                    found = True
                    break
            if found:
                break
    final_all_causes_set = set(all_causes)
    return [d for d in all_causes_list if frozenset(d['X_x_prime'].keys()) in final_all_causes_set]

def check_op(y_actual, op, y_thr):
    """ Check if u op v holds, where op is one of the comparison operators."""
    assert op in ['==', '!=', '<=', '<', '>=', '>']
    return eval(f'{y_actual}{op}{y_thr}') 

def find_all_causes_ac1_and_ac2(scm: SCMSystem, context: dict[str,object], Y: str, op: str, y_thr: object, include_exo=False):
    """
    Find all sets of variables X (with their actual assignments x) that
    cause Y = y, in the sense that intervening on X to some x' (different
    from x in *all* coordinates) flips Y to a value != y.

    :param scm:    An SCM object with:
                     - scm.variables
                     - scm.domains[v]
                     - scm.get_state(context)
    :param context: dict of exogenous variable assignments (and possibly
                    partial interventions).
    :param Y:       The name of the variable in question (a string).
    :param y:       The value of Y we observe in the actual context.
    :return:        A list of (X, x) pairs, where X is a tuple of variable
                    names and x is their actual assignment in the given
                    context.
    """
    # 1. Compute the actual state (all endogenous variables) in the given context
    actual_state = scm.get_state(context)
    
    # If Y != y in the actual state, there are no "causes" for Y=y
    if not check_op(actual_state[Y], op, y_thr):
        return []
    
    # Helper to get all subsets of an iterable
    def all_subsets(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    if not include_exo:
        candidate_vars = [v for v in scm.endogenous_vars if v != Y]
    else:
        candidate_vars = [v for v in scm.vars if v != Y]
    
    all_causes = []
    all_causes_set = set()
    # 2. Enumerate all non-empty subsets of candidate_vars
    for X_subset in tqdm(list(all_subsets(candidate_vars))):
        if not X_subset:
            continue

        if frozenset(X_subset) in all_causes_set:
            continue
        
        # 2a. Get the actual assignment x of these variables in the actual state
        x = {v: actual_state[v] for v in X_subset}

        # 2b. Build all possible "new assignments" x' that differ
        #     in *every* variable of X_subset
        #     i.e. for each v in X_subset, choose a value from domain[v]
        #     that is != x[v].
        domain_options = []
        broke = False
        for v in X_subset:
            if v not in scm.domains:
                broke = True
                break
            domain_options.append([val for val in scm.domains[v].all if val != x[v]])
        if broke:
            continue

        # If any variable has no alternative domain values, skip
        # (this can happen if domain is 1-element or x[v] is the only possibility)
        if any(len(opts) == 0 for opts in domain_options):
            continue
        
        for W_subset in list(all_subsets(set(candidate_vars)-set(X_subset))):
            # if not W_subset:
            #     continue            
            
            w = {v: actual_state[v] for v in W_subset}
            
            # 2c. Check each product of these domain options
            for combo in product(*domain_options):
                x_prime_w = dict(zip(X_subset, combo))
                x_prime_w.update(w)
                
                # Evaluate the new state
                new_state = scm.get_state(context, interventions=x_prime_w)
                
                # Debug logging
                if frozenset(X_subset) == frozenset(['ST']):
                    print(f"    DEBUG ST: X_subset={X_subset}, combo={combo}, W_subset={W_subset}")
                    print(f"    DEBUG ST: x_prime_w={x_prime_w}")
                    print(f"    DEBUG ST: new_state={new_state}")
                    print(f"    DEBUG ST: Y={Y}, new_state[Y]={new_state[Y]}, op={op}, y_thr={y_thr}")
                    print(f"    DEBUG ST: check_op result={check_op(new_state[Y], op, y_thr)}")
                
                # Check if Y changed
                if not check_op(new_state[Y], op, y_thr):
                    # If Y is different from y, we found that X_subset=x is a cause
                    # (under the "change all variables in X_subset" definition).
                    print(f"    FOUND CAUSE: {X_subset} -> {dict(zip(X_subset, combo))} with W={W_subset}")
                    all_causes.append(
                        dict(
                            X_x_prime=dict(zip(X_subset, combo)),
                            W=W_subset,
                            w=w
                        )
                    )
                    all_causes_set.add(frozenset(X_subset))
                    break  # No need to keep checking more combos for this X_subset
        
    return all_causes

if __name__ == '__main__':
    system = read_system('examples/car-small.conf')
    THR = 2
    while True:
        context = system.get_random_context()
        state = system.get_state(context)
        if state['V_next'] <= THR:
            break
    
    a_results = find_all_causes(system, context, 'V_next', '<=', THR)
    print(a_results)


def pretty_print_a_cause(state: dict, cause: dict):
    xs = [f"{v}={state[v]} ({v}'={cause['X_x_prime'][v]})" for v in cause['X_x_prime']]
    ws = [f"{v}={state[v]}" for v in cause['w']]
    if ws:
        print(f"The fact that {','.join(xs)} under the contingency {','.join(ws)}")
    else:
        print(f"The fact that {','.join(xs)}")

def pretty_print_causes(system: SCMSystem, context: dict, causes: list[dict]):
    print(f"Times: AC1={causes['ac2_time']:.5f}, AC2={causes['ac3_time']:.5f}")
    state = system.get_state(context)
    for cause in causes['results']:
        pretty_print_a_cause(state, cause)
