#!/usr/bin/env python3
"""Smoke test to verify artifact installation and basic functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from core.scm import SCMSystem
        from core.failure import ClosedHalfSpaceFailureSet
        from subprojects.liab.k_leg_liab import k_leg_liab
        from subprojects.liab.shapley_liab import shapley_liab
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic k-leg liability calculation."""
    try:
        from core.scm import SCMSystem
        from core.failure import ClosedHalfSpaceFailureSet
        from subprojects.liab.k_leg_liab import k_leg_liab
        
        # Create simple test system
        components = ["A=a+10", "B=b+10", "C=A*B+c+10"]
        domains = {"a": [-100, 100], "b": [-100, 100], "c": [-100, 100]}
        
        T = SCMSystem(components, domains)  # Implementation
        S = SCMSystem(["A=a", "B=b", "C=A*B+c"], domains)  # Specification
        
        context = {"a": 10, "b": 10, "c": 10}
        failure = ClosedHalfSpaceFailureSet({"C": (250, "ge")})
        
        liabilities = k_leg_liab(T, S, context, failure, k=2)
        
        if len(liabilities) == 3 and sum(liabilities.values()) > 0.99:
            print("✓ Basic k-leg liability calculation working")
            return True
        else:
            print("✗ K-leg liability calculation failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("Running artifact smoke tests...")
    
    tests = [test_imports, test_basic_functionality]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    if passed == len(tests):
        print("✓ All tests passed! Artifact ready.")
        return 0
    else:
        print(f"✗ {len(tests) - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
