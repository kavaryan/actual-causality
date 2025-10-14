#!/usr/bin/env python3
"""
Liability experiment script that runs k-leg and Shapley liability calculations
on SCM systems with failure conditions.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.scm import read_system_str, read_temporal_system_str, SCMSystem
from core.failure import QFFOFormulaFailureSet
from subprojects.liab.k_leg_liab import k_leg_liab
from subprojects.liab.shapley_liab import shapley_liab
import sympy as sp


def parse_args():
    parser = argparse.ArgumentParser(description='Run liability experiments on SCM systems')
    parser.add_argument('spec_file', help='Path to specification SCM configuration file')
    parser.add_argument('impl_file', help='Path to implementation SCM configuration file')
    parser.add_argument('--temporal', action='store_true', 
                       help='Treat the files as temporal SCM systems')
    parser.add_argument('--temporal_expansion_window_width', type=int, default=1,
                       help='Window width for temporal expansion (default: 1)')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='Delta parameter for temporal expansion (default: 0.1)')
    parser.add_argument('--failure', required=True,
                       help='Failure formula (e.g., "V > 2")')
    parser.add_argument('--k', type=int, required=True,
                       help='k parameter for k-leg liability')
    parser.add_argument('--num-exps', type=int, default=10,
                       help='Number of experiments to run (default: 10)')
    parser.add_argument('--output', default='liability_results.json',
                       help='Output JSON file (default: liability_results.json)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


def parse_failure_formula(failure_str: str) -> sp.Basic:
    """Parse a failure formula string into a sympy expression."""
    try:
        return sp.sympify(failure_str)
    except Exception as e:
        raise ValueError(f"Failed to parse failure formula '{failure_str}': {e}")


def find_failure_context(spec_scm: SCMSystem, impl_scm: SCMSystem, failure_formula: sp.Basic, max_attempts: int = 1000) -> Dict[str, float]:
    """
    Find a random context where the specification doesn't fail but the implementation does fail.
    
    Args:
        spec_scm: The specification SCM system
        impl_scm: The implementation SCM system
        failure_formula: Sympy expression representing the failure condition
        max_attempts: Maximum number of random attempts
        
    Returns:
        A context dictionary where spec doesn't fail but impl fails
        
    Raises:
        RuntimeError: If no satisfying context is found within max_attempts
    """
    failure_set = QFFOFormulaFailureSet(failure_formula)
    
    for _ in range(max_attempts):
        # Use the implementation system's exogenous variables for context generation
        # (assuming both systems have the same exogenous variables)
        context = impl_scm.get_random_context()
        
        spec_state = spec_scm.get_state(context)
        impl_state = impl_scm.get_state(context)
        
        # We want: spec doesn't fail AND impl fails
        if not failure_set.contains(spec_state) and failure_set.contains(impl_state):
            return context
    
    raise RuntimeError(f"Could not find a context where spec doesn't fail but impl fails within {max_attempts} attempts")


def single_subject(spec_scm: SCMSystem, impl_scm: SCMSystem, failure_formula: sp.Basic, k: int) -> Dict[str, Any]:
    """
    Run a single experiment subject: find a failure context and compute liabilities.
    
    Args:
        spec_scm: Specification SCM system
        impl_scm: Implementation SCM system
        failure_formula: Failure condition
        k: k parameter for k-leg liability
        
    Returns:
        Dictionary with context and liability results
    """
    import traceback
    
    try:
        # Find a context where spec doesn't fail but impl fails
        context = find_failure_context(spec_scm, impl_scm, failure_formula)
        
        # Create failure set
        failure_set = QFFOFormulaFailureSet(failure_formula)
        
        # Compute k-leg liability
        k_leg_result = k_leg_liab(impl_scm, spec_scm, context, failure_set, k=k)
        
        # Compute Shapley liability
        shapley_result = shapley_liab(impl_scm, spec_scm, context, failure_set)
        
        # Verify the states for debugging
        spec_state = spec_scm.get_state(context)
        impl_state = impl_scm.get_state(context)
        
        # Debug output
        print(f"Context: {context}")
        print(f"Spec state: {spec_state}")
        print(f"Impl state: {impl_state}")
        print(f"Spec fails: {failure_set.contains(spec_state)}")
        print(f"Impl fails: {failure_set.contains(impl_state)}")
        print(f"Failure formula: {failure_formula}")
        
        return {
            'context': context,
            'spec_state': spec_state,
            'impl_state': impl_state,
            'spec_fails': failure_set.contains(spec_state),
            'impl_fails': failure_set.contains(impl_state),
            'k-leg': k_leg_result,
            'shapley': shapley_result,
            'k-of-k-leg': k
        }
    except Exception as e:
        print(f"Full traceback for single_subject error:")
        traceback.print_exc()
        raise e


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Read the specification configuration file
    try:
        spec_config_content = Path(args.spec_file).read_text()
    except FileNotFoundError:
        print(f"Error: Specification file '{args.spec_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading specification file '{args.spec_file}': {e}")
        sys.exit(1)
    
    # Read the implementation configuration file
    try:
        impl_config_content = Path(args.impl_file).read_text()
    except FileNotFoundError:
        print(f"Error: Implementation file '{args.impl_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading implementation file '{args.impl_file}': {e}")
        sys.exit(1)
    
    # Parse the specification SCM system
    try:
        if args.temporal:
            spec_temporal_system = read_temporal_system_str(spec_config_content)
            spec_scm = spec_temporal_system.expand_to_scm(
                temporal_expansion_window_width=args.temporal_expansion_window_width,
                delta=args.delta
            )
        else:
            # Check if file contains temporal sections
            if '[differential_equations]' in spec_config_content:
                print("Error: Specification file contains differential equations but --temporal flag not provided")
                sys.exit(1)
            spec_scm = read_system_str(spec_config_content)
    except Exception as e:
        print(f"Error parsing specification SCM system: {e}")
        sys.exit(1)
    
    # Parse the implementation SCM system
    try:
        if args.temporal:
            impl_temporal_system = read_temporal_system_str(impl_config_content)
            impl_scm = impl_temporal_system.expand_to_scm(
                temporal_expansion_window_width=args.temporal_expansion_window_width,
                delta=args.delta
            )
        else:
            # Check if file contains temporal sections
            if '[differential_equations]' in impl_config_content:
                print("Error: Implementation file contains differential equations but --temporal flag not provided")
                sys.exit(1)
            impl_scm = read_system_str(impl_config_content)
    except Exception as e:
        print(f"Error parsing implementation SCM system: {e}")
        sys.exit(1)
    
    # Parse failure formula
    try:
        failure_formula = parse_failure_formula(args.failure)
        
        # Check that failure formula only references endogenous variables from both systems
        formula_vars = {str(s) for s in failure_formula.free_symbols}
        spec_endogenous_vars = set(spec_scm.endogenous_vars)
        impl_endogenous_vars = set(impl_scm.endogenous_vars)
        
        # Formula should reference variables that exist in both systems
        common_endogenous_vars = spec_endogenous_vars & impl_endogenous_vars
        invalid_vars = formula_vars - common_endogenous_vars
        if invalid_vars:
            print(f"Error: Failure formula references variables not common to both systems: {invalid_vars}")
            print(f"Common endogenous variables: {common_endogenous_vars}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error parsing failure formula: {e}")
        sys.exit(1)
    
    # Run experiments
    results = []
    print(f"Running {args.num_exps} experiments...")
    
    for i in range(args.num_exps):
        try:
            result = single_subject(spec_scm, impl_scm, failure_formula, args.k)
            results.append(result)
            print(f"Completed experiment {i+1}/{args.num_exps}")
        except Exception as e:
            import traceback
            print(f"Error in experiment {i+1}: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
            continue
    
    if not results:
        print("Error: No experiments completed successfully")
        sys.exit(1)
    
    # Save results
    output_data = {
        'config': {
            'spec_file': args.spec_file,
            'impl_file': args.impl_file,
            'temporal': args.temporal,
            'temporal_expansion_window_width': args.temporal_expansion_window_width,
            'delta': args.delta,
            'failure': args.failure,
            'k': args.k,
            'num_exps': args.num_exps,
            'seed': args.seed
        },
        'system_info': {
            'spec_endogenous_vars': spec_scm.endogenous_vars,
            'spec_exogenous_vars': spec_scm.exogenous_vars,
            'impl_endogenous_vars': impl_scm.endogenous_vars,
            'impl_exogenous_vars': impl_scm.exogenous_vars,
            'spec_total_vars': len(spec_scm.vars),
            'impl_total_vars': len(impl_scm.vars)
        },
        'results': results
    }
    
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
        print(f"Successfully completed {len(results)} out of {args.num_exps} experiments")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
