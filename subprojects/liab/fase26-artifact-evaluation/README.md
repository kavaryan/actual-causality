# Causal Liability in Autonomous Systems - Artifact

This artifact implements the k-leg liability framework from the FASE 2026 paper "Causal Liability in Autonomous Systems."

## Getting Started Guide

### Installation & Smoke Test
Please see `REQUIREMENTS` for hardware and software requirements.

The following command sideloads the provided Docker image and runs the smoke test:
```bash
make smoke-test
```

Expected output: a small example of k-leg liability computation ending with a line stating "Smoke test passed!". (The full make options are listed in the `make help` output.)

## Step-by-Step Instructions

### Reproduce Paper Results
| RQ | Figure/Table | Page | How to Reproduce | Expected Output |
|----|--------------|------|------------------|-----------------|
| RQ1 (Accuracy) | Figure 3 (left) | 18 | `docker run --rm -v $(pwd)/images:/artifact/images liab-artifact:latest python subprojects/liab/case-study.py` | Box plot showing liability differences between Shapley and k-leg methods saved as image in `./images` directory |
|  | Table 4 | 17 | Same command as above | p-values table printed to console showing statistical significance of differences between k-leg and Shapley liabilities |
| RQ2 (Efficiency) | Figure 3 (right) | 18 | Same command as above | Box plot showing computational time differences between Shapley and k-leg methods saved as image in `./images` directory |

All results reported in the paper can be reproduced with a single invocation of
 `docker run --rm -v ./images:/artifact/images liab-artifact:latest python subprojects/liab/case-study.py`.

For a faster, approximate run that provides a high-level view of the results, `--num-samples` can be reduced (e.g., `--num-samples 100` instead of the default 1000). All configuration options are listed in the `--help` output.

### Paper Claims Supported
- Empirical comparison with Shapley values showing k-leg approximates Shapley well
- Polynomial complexity vs exponential Shapley computation time (Proposition 1)
- Case study demonstrating practical applicability

### Paper Claims NOT Supported
- Formal theorem proofs (mathematical, not computational)

## About Our Project

This artifact provides a comprehensive library for causal liability analysis in autonomous systems:

- **`core/scm.py`**: Core structural causal model implementation with component definitions and system evaluation
- **`core/failure.py`**: Failure set definitions including closed half-spaces
- **`core/bf.py`**: But-for causality checking implementation
- **`core/random_system.py`**: Random system generation utilities for experiments
- **`subprojects/liab/k_leg_liab.py`**: Main k-leg liability computation algorithm
- **`subprojects/liab/shapley_liab.py`**: Shapley value-based liability computation for comparison

Moreover, two driver scripts are provided:
- **`subprojects/liab/smoke-test.py`**: Lightweight sanity check that verifies correct installation and execution of the core pipeline on a small, fast-running configuration.
- **`subprojects/liab/case-study.py`**: Complete experimental framework reproducing the results reported in the paper.

Also, the provided `Makefile` and `Dockerfile` streamline the build and execution process by automating image construction, loading, and validation through a small set of self-documented targets.

### Reusability

This library can be easily extended and reused for other causal analysis applications. The modular design allows researchers to define
custom failure conditions, implement new causality measures, or apply the framework to different domains beyond autonomous vehicles.

**Core Class Hierarchies:**
- **`FailureSet`** (abstract base class): Extensible hierarchy for defining failure conditions
  - `ClosedHalfSpaceFailureSet`: Conjunctions of linear inequalities (e.g., `x ≥ a ∧ y ≤ b`)
  - `QFFOFormulaFailureSet`: Quantifier-free first-order formulas using SymPy expressions
  - Custom failure sets can be added by implementing `contains()` and `dist()` methods

- **`SCMSystem`**: Core structural causal model with component-based architecture
  - `Component`: Individual causal equations with symbolic expression support
  - `TemporalSCMSystem`: Extension for differential equations and temporal dynamics
  - Domain specifications via `BoundedIntInterval` and `BoundedFloatInterval`

- **Liability Algorithms**: Pluggable causality measures
  - `k_leg_liab()`: Our polynomial-time k-leg liability framework
  - `shapley_liab()`: Shapley value-based liability for comparison
  - `bf()`: But-for causality checking with memoization

The SCM framework supports both deterministic and probabilistic models, temporal systems with differential equations, and interval
specifications, making it suitable for a wide range of real-world systems requiring liability analysis.

**Active Research Use:** These core classes are currently being used in our other papers under review, demonstrating their practical
utility across multiple research projects in causal analysis, metamorphic testing, and formal verification domains.
