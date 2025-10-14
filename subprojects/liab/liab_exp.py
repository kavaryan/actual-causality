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


def parse_failure_formula(failure_str: str) -> sp.Basic:
    """Parse a failure formula string into a sympy expression."""
    try:
        return sp.sympify(failure_str)
    except Exception as e:
        raise ValueError(f"Failed to parse failure formula '{failure_str}': {e}")


def find_failure_context(scm: SCMSystem, failure_formula: sp.Basic, max_attempts: int = 1000) -> Dict[str, float]:
    """
    Find a random context that satisfies the failure condition.
    
    Args:
        scm: The SCM system
        failure_formula: Sympy expression representing the failure condition
        max_attempts: Maximum number of random attempts
        
    Returns:
        A context dictionary that satisfies the failure condition
        
    Raises:
        RuntimeError: If no satisfying context is found within max_attempts
    """
    failure_set = QFFOFormulaFailureSet(failure_formula)
    
    for _ in range(max_attempts):
        context = scm.get_random_context()
        state = scm.get_state(context)
        
        if failure_set.contains(state):
            return context
    
    raise RuntimeError(f"Could not find a context satisfying the failure condition within {max_attempts} attempts")


def single_subject(scm: SCMSystem, spec_scm: SCMSystem, failure_formula: sp.Basic, k: int) -> Dict[str, Any]:
    """
    Run a single experiment subject: find a failure context and compute liabilities.
    
    Args:
        scm: Implementation SCM system
        spec_scm: Specification SCM system (same as implementation for now)
        failure_formula: Failure condition
        k: k parameter for k-leg liability
        
    Returns:
        Dictionary with context and liability results
    """
    # Find a context that satisfies the failure condition
    context = find_failure_context(scm, failure_formula)
    
    # Create failure set
    failure_set = QFFOFormulaFailureSet(failure_formula)
    
    # Compute k-leg liability
    k_leg_result = k_leg_liab(scm, spec_scm, context, failure_set, k=k)
    
    # Compute Shapley liability
    shapley_result = shapley_liab(scm, spec_scm, context, failure_set)
    
    return {
        'context': context,
        'k-leg': k_leg_result,
        'shapley': shapley_result,
        'k-of-k-leg': k
    }


def main():
    parser = argparse.ArgumentParser(description='Run liability experiments on SCM systems')
    parser.add_argument('file', help='Path to SCM configuration file')
    parser.add_argument('--temporal', action='store_true', 
                       help='Treat the file as a temporal SCM system')
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
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Read the configuration file
    try:
        config_content = Path(args.file).read_text()
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{args.file}': {e}")
        sys.exit(1)
    
    # Parse the SCM system
    try:
        if args.temporal:
            temporal_system = read_temporal_system_str(config_content)
            scm = temporal_system.expand_to_scm(
                temporal_expansion_window_width=args.temporal_expansion_window_width,
                delta=args.delta
            )
        else:
            # Check if file contains temporal sections
            if '[differential_equations]' in config_content:
                print("Error: File contains differential equations but --temporal flag not provided")
                sys.exit(1)
            scm = read_system_str(config_content)
    except Exception as e:
        print(f"Error parsing SCM system: {e}")
        sys.exit(1)
    
    # Parse failure formula
    try:
        failure_formula = parse_failure_formula(args.failure)
        
        # Check that failure formula only references endogenous variables
        formula_vars = {str(s) for s in failure_formula.free_symbols}
        endogenous_vars = set(scm.endogenous_vars)
        
        invalid_vars = formula_vars - endogenous_vars
        if invalid_vars:
            print(f"Error: Failure formula references non-endogenous variables: {invalid_vars}")
            print(f"Available endogenous variables: {endogenous_vars}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error parsing failure formula: {e}")
        sys.exit(1)
    
    # For now, use the same system as both implementation and specification
    spec_scm = scm
    
    # Run experiments
    results = []
    print(f"Running {args.num_exps} experiments...")
    
    for i in range(args.num_exps):
        try:
            result = single_subject(scm, spec_scm, failure_formula, args.k)
            results.append(result)
            print(f"Completed experiment {i+1}/{args.num_exps}")
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
            continue
    
    if not results:
        print("Error: No experiments completed successfully")
        sys.exit(1)
    
    # Save results
    output_data = {
        'config': {
            'file': args.file,
            'temporal': args.temporal,
            'temporal_expansion_window_width': args.temporal_expansion_window_width,
            'delta': args.delta,
            'failure': args.failure,
            'k': args.k,
            'num_exps': args.num_exps,
            'seed': args.seed
        },
        'system_info': {
            'endogenous_vars': scm.endogenous_vars,
            'exogenous_vars': scm.exogenous_vars,
            'total_vars': len(scm.vars)
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
