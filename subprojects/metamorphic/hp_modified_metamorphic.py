import copy
from actual.hp_modified import find_all_causes
from actual.scm import BoundedIntInterval, SCMSystem

def expand_metamorphic_rels(scm: SCMSystem, metamorphic_rels: list[tuple], target: str):
    # Todo: improve propgatation
    ans = dict()
    for var1, var2, direction in metamorphic_rels:
        if var1 == target:
            ans[var2] = direction
        elif var2 == target:
            ans[var1] = -direction
    return ans

def find_all_causes_metamorphic(scm: SCMSystem, context: dict[str,object], Y: str, op: str , y: object,
    include_exo=False, metamorphic_rels: list[tuple]=None):
    """
        metamorphic_rels: list[tuple], each tuple
            ('<var-name-1>','<var-name-2>','<dir>(-1/+1)')
    """
    actual_state = scm.get_state(context)
    scm2 = copy.deepcopy(scm)
    
    if metamorphic_rels:
        metamorphic_rels_dict = expand_metamorphic_rels(scm, metamorphic_rels, target=Y)
        for var, direction in metamorphic_rels_dict.items():
            if (direction == 1 and (op == '<' or op == '<=') or
                direction == -1 and (op == '>' or op == '>=')):
                scm2.domains[var] = BoundedIntInterval(scm2.domains[var].a, actual_state[var])
            elif (direction == 1 and (op == '>' or op == '>=') or
                direction == -1 and (op == '<' or op == '<=')):
                scm2.domains[var] = BoundedIntInterval(actual_state[var], scm2.domains[var].b)


    return find_all_causes(scm2, context, Y, op, y, include_exo=include_exo)