# Causal Liability in Autonomous Systems - Artifact

This artifact implements the k-leg liability framework from the FASE 2026 paper "Causal Liability in Autonomous Systems".

## Getting Started Guide

### Installation & Smoke Test

```bash
docker build -t liability-artifact .
docker run --rm liability-artifact python smoke_test.py
```

Expected output: "✓ All tests passed! Artifact ready."

### Quick Demo
```bash
docker run --rm liability-artifact python demo.py
```

## Step-by-Step Instructions

### Reproduce Paper Results

1. **Basic Examples (Section 3)**:
   ```bash
   docker run --rm liability-artifact python examples.py
   ```

2. **Empirical Evaluation (Section 7)**:
   ```bash
   docker run --rm liability-artifact python experiments.py
   ```
   Generates comparison with Shapley values, runtime ~5 minutes.

3. **Case Studies**:
   ```bash
   docker run --rm liability-artifact python case_studies.py
   ```

### Paper Claims Supported
- Theoretical properties (Theorems 4.1-4.3)
- Empirical comparison with Shapley values
- AEB system case study
- Polynomial complexity vs exponential Shapley

### Paper Claims NOT Supported
- Formal theorem proofs (mathematical, not computational)
- Legal framework integration (conceptual)
