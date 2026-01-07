

#%% --- cell (code) ---
import sys
sys.path.append('../src')


#%% --- cell (code) ---
import numpy as np

from liab.scm import ComponentOrEquation, GSym, System
from liab.failure import ClosedHalfSpaceFailureSet
from liab.k_leg_liab import k_leg_liab

a_sp = ComponentOrEquation(['a'], 'A', 'a')
b_sp = ComponentOrEquation(['b'], 'B', 'b')
c_sp = ComponentOrEquation(['c'], 'C', 'c')
d_sp = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)')
a_im = ComponentOrEquation(['a'], 'A', 'a+10')
b_im = ComponentOrEquation(['b'], 'B', 'b+10')
c_im = ComponentOrEquation(['c'], 'C', 'c+8')
d_im = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)+10')


#%% --- cell (code) ---
S = System([a_sp, b_sp, c_sp, d_sp])
T = System([a_im, b_im, c_im, d_im])

print(f'{S=}')
print(f'{T=}')


#%% --- cell (code) ---
M = S.induced_scm()
N = T.induced_scm(state_order=M.state_order)
print(f'{M=}')
print(f'{N=}')


#%% --- cell (code) ---
M.draw()


#%% --- cell (code) ---
u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
F = ClosedHalfSpaceFailureSet({'D': (250, 'ge')})


#%% --- cell (code) ---
print('Specification state at u=', M.get_state(u)[0])
print('Implementation state at u=', N.get_state(u)[0])


#%% --- cell (code) ---
for X in S.cs:
    print(f'Implementation/fixed-{X.O} state at u=', 
          T.get_replacement({X.O: X}).induced_scm().get_state(u)[0])


#%% --- cell (code) ---
assert S.induced_scm().get_state(u)[0] == {'A': 10, 'B': 10, 'C': 10, 'D': 110}
assert T.induced_scm().get_state(u)[0] == {'A': 20, 'B': 20, 'C': 18, 'D': 420}


#%% --- cell (code) ---
liabs = k_leg_liab(T, S, u, F, k=2)
print(f'{liabs=}')
