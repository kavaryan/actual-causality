import numpy as np
import cvxpy as cp
from typing import Any, Dict, Optional


def diffalgo2(
    W: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    thr: float,
    lmbda: float = 10.0,
    delta_inf_bound: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Differentiable HP-style causal search for linear SCMs, faithful to the paper.

    This implements the *hinge-penalised* relaxation described in the paper:

        x(δ) = (I - W)^{-1} (b + δ)
        ξ(x(δ), φ) = c^T x(δ) - thr

        J_λ(δ) = ||δ||_1 + λ · [ ξ(x(δ), φ) ]_+,
        where [t]_+ = max(t, 0).

    We solve the convex optimisation problem

        minimise_δ   ||δ||_1 + λ · [ ξ(x(δ), φ) ]_+,

    which is the unconstrained hinge-penalised version of

        minimise_δ   ||δ||_1
        subject to   ξ(x(δ), φ) ≤ 0.

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Strictly lower-triangular weight matrix of the linear SCM (A in the paper).
        The SCM is x = W x + b, so (I - W) must be invertible.
    b : np.ndarray, shape (n,)
        Exogenous offset vector.
    c : np.ndarray, shape (n,)
        Coefficients defining the linear effect O = c^T x.
    thr : float
        Threshold in the effect formula φ: (O > thr).
    lmbda : float, default 10.0
        Trade-off parameter λ between sparsity and robustness violation.
        Larger λ enforces the constraint ξ ≤ 0 more strongly.
    delta_inf_bound : float or None, default None
        Optional bound on ||δ||_∞. If None, no such bound is imposed.
        This is *not* part of the theoretical formulation in the paper, but can
        be useful in practice. To stay strictly faithful to the paper, leave as None.

    Returns
    -------
    result : dict
        A dictionary with the following keys:
        - "delta_star": np.ndarray, optimal intervention vector δ*
        - "x_star": np.ndarray, resulting state x(δ*)
        - "robustness": float, ξ(x(δ*), φ)
        - "delta_norm_l1": float, ||δ*||_1
        - "cause_size": int, number of coordinates with |δ_i| > 1e-6
        - "status": str, solver status

    Notes
    -----
    - This function assumes that the factual robustness ξ(x(0), φ) is positive,
      i.e., the effect φ holds in the factual state, as in the paper.
    - The optimisation is convex because:
        * x(δ) is affine in δ,
        * ξ(x(δ), φ) is affine in δ,
        * [·]_+ is convex and non-decreasing,
        * ||δ||_1 is convex,
      and the sum of convex functions is convex.
    """
    # Basic shape checks
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix of shape (n, n).")
    n = W.shape[0]

    if b.shape != (n,):
        raise ValueError(f"b must have shape ({n},), got {b.shape}.")
    if c.shape != (n,):
        raise ValueError(f"c must have shape ({n},), got {c.shape}.")

    # Compute (I - W)^{-1} as in the linear SCM formulation
    I = np.eye(n)
    try:
        inv = np.linalg.inv(I - W)
    except np.linalg.LinAlgError as e:
        raise ValueError("Matrix (I - W) is not invertible; SCM is ill-posed.") from e

    # CVXPY variable for additive interventions δ
    delta = cp.Variable(n)

    # x(δ) = (I - W)^{-1} (b + δ)
    x = inv @ (b + delta)

    # Robustness ξ(x, φ) = c^T x - thr
    robustness = c @ x - thr

    # Hinge: [ξ]_+ = max(ξ, 0)
    hinge_robustness = cp.pos(robustness)

    # Objective: J_λ(δ) = ||δ||_1 + λ · [ξ]_+
    objective = cp.Minimize(cp.norm1(delta) + lmbda * hinge_robustness)

    # Constraints: only those explicitly requested by the caller.
    constraints = []
    if delta_inf_bound is not None:
        constraints.append(cp.norm_inf(delta) <= float(delta_inf_bound))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed: status={prob.status}")

    delta_star = delta.value
    x_star = (inv @ (b + delta_star)).reshape(-1)
    robustness_val = float((c @ x_star) - thr)
    delta_norm_l1 = float(np.linalg.norm(delta_star, 1))
    cause_size = int(np.sum(np.abs(delta_star) > 1e-6))

    return {
        "delta_star": delta_star,
        "x_star": x_star,
        "robustness": robustness_val,
        "delta_norm_l1": delta_norm_l1,
        "cause_size": cause_size,
        "status": prob.status,
    }
