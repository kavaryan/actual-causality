import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

# !pip install ecos

def diffalgo(W, b, c, thr, lmbda=10.0, return_all=False):
    r"""
    Solves the convex relaxed causal search:
        min_{delta} ||delta||_1 + lambda * (thr - c^T x + epsilon)_+
    where x = (I - W)^{-1} (b + delta), c is effect vector, thr is threshold.
    Returns delta_star, x_star, and optionally more diagnostics.
    """
    n = W.shape[0]
    I = np.eye(n)
    inv = np.linalg.inv(I - W)
    delta = cp.Variable(n)
    x = inv @ (b + delta)
    robustness = c @ x - thr
    # Objective: sparse minimal intervention & ensure robustness negative
    objective = cp.Minimize(cp.norm1(delta) + lmbda * cp.pos(-(robustness)))
    prob = cp.Problem(objective)
    t0 = time.time()
    prob.solve(solver=cp.ECOS)
    t1 = time.time()
    result = {
        'delta_star': delta.value,
        'x_star': x.value,
        'robustness': robustness.value,
        'runtime': t1 - t0,
        'cause_size': np.sum(np.abs(delta.value) > 1e-6),  # count nonzeros
        'delta_norm': np.linalg.norm(delta.value, 1)
    }
    if return_all:
        result['prob'] = prob
    return result