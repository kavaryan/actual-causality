from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
import time

from subprojects.differentiable.scm import SCM, EffectFunction

# ----------------------------
# Proximal gradient (ISTA) for ||delta||_1 + lambda * hinge(xi(delta))
# ----------------------------

def soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def diffalgo(
    scm: SCM,
    effect: EffectFunction,
    lmbda: float,
    step_size: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-6,
    delta0: Optional[np.ndarray] = None,
    delta_inf_bound: Optional[float] = None,
    backtracking: bool = True,
    bt_shrink: float = 0.5,
    bt_max_tries: int = 20,
    return_history: bool = False,
    intervenable: Optional[np.ndarray] = None
) -> dict:
    """
    Minimise:  ||delta||_1 + lmbda * max(xi(x(delta)), 0)

    Uses proximal gradient (ISTA): smooth part is lmbda*hinge(xi),
    nonsmooth part is ||delta||_1 handled by soft-thresholding.

    Notes:
    - hinge is nonsmooth at xi=0; we use subgradient: d/dxi hinge = 1 if xi>0 else 0.
    - If backtracking=True, we shrink step size until objective decreases.
    """
    start_time = time.time()
    n = scm.n
    if intervenable is None:
        intervenable = np.ones(n, dtype=bool)
    else:
        intervenable = np.asarray(intervenable, dtype=bool).reshape(-1)
        assert intervenable.shape == (n,)

    if delta0 is None:
        delta = np.zeros(n, dtype=float)
    else:
        delta = np.asarray(delta0, dtype=float).reshape(-1).copy()
        if delta.shape != (n,):
            raise ValueError(f"delta0 must have shape ({n},), got {delta.shape}")

    def objective_and_grad(delta_vec: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        x, xi, dxi_dd = scm.effect_and_grad_delta(delta_vec, effect)
        hinge = max(xi, 0.0)
        # obj = float(np.linalg.norm(delta_vec, 1) + lmbda * hinge)
        l1 = float(np.linalg.norm(delta_vec[intervenable], 1))
        obj = l1 + lmbda * hinge
        # subgradient of hinge wrt delta
        if xi > 0.0:
            grad_smooth = lmbda * dxi_dd
        else:
            grad_smooth = np.zeros_like(delta_vec)
        grad_smooth[~intervenable] = 0.0
        return obj, grad_smooth, x, xi

    hist = {"obj": [], "l1": [], "xi": []} if return_history else None

    obj, grad_smooth, x, xi = objective_and_grad(delta)

    for it in range(max_iter):
        if return_history:
            hist["obj"].append(obj)
            hist["l1"].append(float(np.linalg.norm(delta, 1)))
            hist["xi"].append(float(xi))

        # One proximal gradient step (with optional backtracking)
        alpha = step_size
        delta_old = delta.copy()
        obj_old = obj

        for _try in range(bt_max_tries if backtracking else 1):
            # gradient step on smooth part
            z = delta_old - alpha * grad_smooth
            delta_new = delta_old.copy()
            delta_new[intervenable] = soft_threshold(z[intervenable], alpha)
            delta_new[~intervenable] = 0.0
            # prox for L1
            #delta_new = soft_threshold(z, alpha)

            # optional infinity-norm bound
            if delta_inf_bound is not None:
                B = float(delta_inf_bound)
                delta_new = np.clip(delta_new, -B, B)

            obj_new, grad_smooth_new, x_new, xi_new = objective_and_grad(delta_new)

            if (not backtracking) or (obj_new <= obj_old + 1e-12):
                delta, obj, grad_smooth, x, xi = delta_new, obj_new, grad_smooth_new, x_new, xi_new
                break
            alpha *= bt_shrink
        else:
            # backtracking failed to find a decrease; stop
            break

        # stopping criterion: small change in delta and objective
        if np.linalg.norm(delta - delta_old, 2) <= tol * (1.0 + np.linalg.norm(delta_old, 2)):
            if abs(obj - obj_old) <= tol * (1.0 + abs(obj_old)):
                break

    out = {
        "delta_star": delta,
        "x_star": x,
        "robustness": float(xi),
        "delta_norm_l1": float(np.linalg.norm(delta, 1)),
        "cause_size": int(np.sum(np.abs(delta) > 1e-6)),
        "status": "converged" if it < max_iter - 1 else "max_iter",
        "iters": it + 1,
        'time': time.time() - start_time,
    }
    if return_history:
        out["history"] = hist
    return out

