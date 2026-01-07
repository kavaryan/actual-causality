# Causal Liability in Autonomous Systems - Artifact

This artifact implements the k-leg liability framework from the FASE 2026 paper "Causal Liability in Autonomous Systems".

## Getting Started Guide

### Installation & Smoke Test
Please see `REQUIREMENTS` for hardware and software requirements.

The following commanf sideloads the provided docker image and runs the smoke test:
```bash
make smoke-test
```

Expected output: A small example of k-leg liability computation ending with "Smoke test passed!" line.

## Step-by-Step Instructions

### Reproduce Paper Results

**Case Study (Section 7)**:
```bash
docker run --rm -v $(pwd)/images:/artifact/images liab-artifact:latest python subprojects/liab/random-system-example.py
```

Expected output:
- The p-value table (Table 4 from the paper)
- Two images corresponding to Figure 3 (left and right plots) saved in the `./images` directory

### Paper Claims Supported
- Theoretical properties (Theorems 4.1-4.3)
- Empirical comparison with Shapley values showing k-leg approximates Shapley well
- Polynomial complexity vs exponential Shapley computation time
- Case study demonstrating practical applicability

### Paper Claims NOT Supported
- Formal theorem proofs (mathematical, not computational)

## About Our Project

This artifact provides a comprehensive library for causal liability analysis in autonomous systems. The key files include:

- **`core/scm.py`**: Core structural causal model implementation with component definitions and system evaluation
- **`core/failure.py`**: Failure set definitions including closed half-spaces and quantifier-free first-order formulas
- **`core/bf.py`**: But-for causality checking implementation
- **`core/random_system.py`**: Random system generation utilities for experiments
- **`subprojects/liab/k_leg_liab.py`**: Main k-leg liability computation algorithm
- **`subprojects/liab/shapley_liab.py`**: Shapley value-based liability computation for comparison
- **`subprojects/liab/random-system-example.py`**: Complete experimental framework reproducing paper results

### Reusability

This library can be easily extended and reused for other causal analysis applications. The modular design allows researchers to define
custom failure conditions, implement new causality measures, or apply the framework to different domains beyond autonomous vehicles.
The SCM framework supports both deterministic and probabilistic models, temporal systems with differential equations, and interval
specifications, making it suitable for a wide range of real-world systems requiring liability analysis.