
#%% --- cell (code) ---
import numpy as np

from core.scm import Component, SCMSystem, BoundedFloatInterval
from core.failure import ClosedHalfSpaceFailureSet
from subprojects.liab.k_leg_liab import k_leg_liab

a_sp = Component('A = a')
b_sp = Component('B = b')
c_sp = Component('C = c')
d_sp = Component('D = d + Max(A*B, A*C, B*C)')
a_im = Component('A = a + 10')
b_im = Component('B = b + 10')
c_im = Component('C = c + 8')
d_im = Component('D = d + Max(A*B, A*C, B*C) + 10')


#%% --- cell (code) ---
# Define domains for all variables
domains = {
    'a': BoundedFloatInterval(0, 100),
    'b': BoundedFloatInterval(0, 100), 
    'c': BoundedFloatInterval(0, 100),
    'd': BoundedFloatInterval(0, 100),
    'A': BoundedFloatInterval(0, 200),
    'B': BoundedFloatInterval(0, 200),
    'C': BoundedFloatInterval(0, 200),
    'D': BoundedFloatInterval(0, 1000)
}

S = SCMSystem([a_sp, b_sp, c_sp, d_sp], domains)
T = SCMSystem([a_im, b_im, c_im, d_im], domains)

print(f'S components: {list(S.components.keys())}')
print(f'T components: {list(T.components.keys())}')


#%% --- cell (code) ---
print(f'S exogenous vars: {S.exogenous_vars}')
print(f'S endogenous vars: {S.endogenous_vars}')
print(f'T exogenous vars: {T.exogenous_vars}')
print(f'T endogenous vars: {T.endogenous_vars}')


#%% --- cell (code) ---
S.display_dag()


#%% --- cell (code) ---
u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
F = ClosedHalfSpaceFailureSet({'D': (250, 'ge')})


#%% --- cell (code) ---
print('Specification state at u=', S.get_state(u))
print('Implementation state at u=', T.get_state(u))


#%% --- cell (code) ---
# Create modified systems by replacing individual components
for comp_name in ['A', 'B', 'C', 'D']:
    if comp_name in S.components:
        # Create a new system with the specification component replaced by implementation
        modified_components = list(T.components.values())
        for i, comp in enumerate(modified_components):
            if comp.output == comp_name:
                modified_components[i] = S.components[comp_name]
                break
        modified_system = SCMSystem(modified_components, domains)
        print(f'Implementation/fixed-{comp_name} state at u=', modified_system.get_state(u))


#%% --- cell (code) ---
assert S.get_state(u) == {'a': 10, 'b': 10, 'c': 10, 'd': 10, 'A': 10, 'B': 10, 'C': 10, 'D': 110}
assert T.get_state(u) == {'a': 10, 'b': 10, 'c': 10, 'd': 10, 'A': 20, 'B': 20, 'C': 18, 'D': 420}


#%% --- cell (code) ---
liabs = k_leg_liab(T, S, u, F, k=2)
print(f'{liabs=}')
