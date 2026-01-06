#!/usr/bin/env python3
"""Quick demonstration of the liability framework."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from core.scm import SCMSystem
from core.failure import ClosedHalfSpaceFailureSet
from subprojects.liab.k_leg_liab import k_leg_liab

def main():
    print("=== K-leg Liability Framework Demo ===\n")
    
    # AEB system from paper
    print("Creating AEB system (Example from paper):")
    components = [
        "A=a+10",      # Speedometer with +10 error
        "B=b+10",      # Radar with +10 error  
        "C=A*B+c+10"   # ECU with +10 error
    ]
    domains = {"a": [-100, 100], "b": [-100, 100], "c": [-100, 100]}
    
    T = SCMSystem(components, domains)  # Implementation
    S = SCMSystem(["A=a", "B=b", "C=A*B+c"], domains)  # Specification
    
    context = {"a": 10, "b": 10, "c": 10}
    failure = ClosedHalfSpaceFailureSet({"C": (250, "ge")})  # Braking force too high
    
    print(f"Context: {context}")
    print(f"Failure condition: C >= 250")
    
    # Calculate states
    impl_state = T.get_state(context)
    spec_state = S.get_state(context)
    
    print(f"\nImplementation state: {impl_state}")
    print(f"Specification state: {spec_state}")
    print(f"System in failure: {failure.contains(impl_state)}")
    
    # Calculate liabilities
    print("\nCalculating k-leg liabilities...")
    liabilities = k_leg_liab(T, S, context, failure, k=2)
    
    print("\nLiability Results:")
    for component, liability in liabilities.items():
        print(f"  Component {component}: {liability:.3f}")
    
    print(f"\nTotal liability: {sum(liabilities.values()):.3f}")
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
