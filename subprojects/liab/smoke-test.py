#!/usr/bin/env python3
"""
Smoke test for k-leg liability computation.
"""

import numpy as np

from core.scm import Component, SCMSystem, BoundedFloatInterval
from core.failure import ClosedHalfSpaceFailureSet
from subprojects.liab.k_leg_liab import k_leg_liab


def test_k_leg_liability():
    """Test k-leg liability computation with a simple example."""
    
    # Define specification components
    a_sp = Component('A = a')
    b_sp = Component('B = b')
    c_sp = Component('C = c')
    d_sp = Component('D = d + Max(A*B, A*C, B*C)')
    
    # Define implementation components
    a_im = Component('A = a + 10')
    b_im = Component('B = b + 10')
    c_im = Component('C = c + 8')
    d_im = Component('D = d + Max(A*B, A*C, B*C) + 10')
    
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
    
    # Create SCM systems
    S = SCMSystem([a_sp, b_sp, c_sp, d_sp], domains)
    T = SCMSystem([a_im, b_im, c_im, d_im], domains)
    
    # Print system details
    print("Specification System (S):")
    for comp in S.components:
        print(f"  {comp.equation}")
    print(f"  Exogenous vars: {S.exogenous_vars}")
    print(f"  Endogenous vars: {S.endogenous_vars}")
    
    print("\nImplementation System (T):")
    for comp in T.components:
        print(f"  {comp.equation}")
    print(f"  Exogenous vars: {T.exogenous_vars}")
    print(f"  Endogenous vars: {T.endogenous_vars}")
    
    print(f"\nContext: {u}")
    print(f"Failure set: {F}")
    
    # Test context and failure set
    u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
    F = ClosedHalfSpaceFailureSet({'D': (250, 'ge')})
    
    # Verify expected states
    s_state = S.get_state(u)
    t_state = T.get_state(u)
    
    expected_s = {'a': 10, 'b': 10, 'c': 10, 'd': 10, 'A': 10, 'B': 10, 'C': 10, 'D': 110}
    expected_t = {'a': 10, 'b': 10, 'c': 10, 'd': 10, 'A': 20, 'B': 20, 'C': 18, 'D': 420}
    
    assert s_state == expected_s, f"Specification state mismatch: {s_state} != {expected_s}"
    assert t_state == expected_t, f"Implementation state mismatch: {t_state} != {expected_t}"
    
    # Compute k-leg liability
    liabs = k_leg_liab(T, S, u, F, k=2)
    
    print(f"Specification state: {s_state}")
    print(f"Implementation state: {t_state}")
    print(f"2-leg liabilities: {liabs}")
    
    print("Smoke test passed!")
    return liabs


if __name__ == "__main__":
    test_k_leg_liability()
