from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np


# ----------------------------
# Function interfaces
# ----------------------------

class NodeFunction(Protocol):
    """A differentiable structural equation x_i = f_i(x_pa) + b_i + delta_i."""
    def value_and_grad(self, x_pa: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns (f(x_pa), grad w.r.t. x_pa).
        grad has shape (len(x_pa),).
        """
        ...


class EffectFunction(Protocol):
    """Returns robustness xi(x) and grad wrt x."""
    def value_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns (xi(x), d xi / d x).
        """
        ...


@dataclass
class LinearEffect:
    """xi(x) = c^T x - thr."""
    c: np.ndarray
    thr: float

    def value_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        c = np.asarray(self.c, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float).reshape(-1)
        if c.shape != x.shape:
            raise ValueError(f"c and x must have same shape, got {c.shape} vs {x.shape}")
        return float(c @ x - self.thr), c.copy()


# ----------------------------
# SCM (DAG, assumes parents < i)
# ----------------------------

@dataclass
class SCM:
    """
    A general (differentiable) DAG SCM in topological order 0..n-1.

    Structural equations:
        x_i = f_i(x_parents(i)) + b_i + delta_i
    where delta is an additive intervention vector (same dimension as x).
    """
    parents: List[np.ndarray]          # parents[i] is array of parent indices for node i
    fns: List[NodeFunction]            # fns[i] maps parent values -> scalar
    b: np.ndarray                      # shape (n,)

    def __post_init__(self) -> None:
        n = len(self.parents)
        if len(self.fns) != n:
            raise ValueError("parents and fns must have the same length")
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {self.b.shape}")
        # basic DAG/topological sanity: all parents indices < i
        for i, pa in enumerate(self.parents):
            pa = np.asarray(pa, dtype=int).reshape(-1)
            if np.any(pa >= i):
                raise ValueError(f"parents of node {i} must be < {i} (topological order)")
            self.parents[i] = pa

    @property
    def n(self) -> int:
        return len(self.parents)

    def forward(self, delta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute x(delta) in topological order."""
        n = self.n
        if delta is None:
            delta = np.zeros(n, dtype=float)
        delta = np.asarray(delta, dtype=float).reshape(-1)
        if delta.shape != (n,):
            raise ValueError(f"delta must have shape ({n},), got {delta.shape}")

        x = np.zeros(n, dtype=float)
        for i in range(n):
            pa_idx = self.parents[i]
            x_pa = x[pa_idx] if pa_idx.size else np.empty((0,), dtype=float)
            fi, _ = self.fns[i].value_and_grad(x_pa)
            x[i] = fi + self.b[i] + delta[i]
        return x

    def forward_do(self, do: dict[int, float]) -> np.ndarray:
      """
      Forward-evaluate counterfactual under do-interventions:
      for i in do: x_i = do[i] (override structural equation)
      otherwise:   x_i = f_i(x_pa) + b_i
      """
      n = self.n
      x = np.zeros(n, dtype=float)
      for i in range(n):
          if i in do:
              x[i] = float(do[i])
              continue
          pa_idx = self.parents[i]
          x_pa = x[pa_idx] if pa_idx.size else np.empty((0,), dtype=float)
          fi, _ = self.fns[i].value_and_grad(x_pa)
          x[i] = float(fi + self.b[i])
      return x

    def effect_and_grad_delta(
        self,
        delta: np.ndarray,
        effect: EffectFunction,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Returns (x, xi, d xi / d delta) using reverse-mode backprop on the DAG.
        """
        n = self.n
        delta = np.asarray(delta, dtype=float).reshape(-1)
        if delta.shape != (n,):
            raise ValueError(f"delta must have shape ({n},), got {delta.shape}")

        # Forward pass: compute x, and cache local grads wrt parents
        x = np.zeros(n, dtype=float)
        local_grads: List[np.ndarray] = [np.empty((0,), dtype=float) for _ in range(n)]
        for i in range(n):
            pa_idx = self.parents[i]
            x_pa = x[pa_idx] if pa_idx.size else np.empty((0,), dtype=float)
            fi, dfi_dpa = self.fns[i].value_and_grad(x_pa)
            if dfi_dpa.shape != (pa_idx.size,):
                raise ValueError(
                    f"Node {i}: grad has shape {dfi_dpa.shape}, expected ({pa_idx.size},)"
                )
            local_grads[i] = dfi_dpa
            x[i] = fi + self.b[i] + delta[i]

        xi, dxi_dx = effect.value_and_grad(x)
        dxi_dx = np.asarray(dxi_dx, dtype=float).reshape(-1)
        if dxi_dx.shape != (n,):
            raise ValueError(f"effect grad must have shape ({n},), got {dxi_dx.shape}")

        # Reverse pass: backprop from xi to delta.
        # Start with adjoints of x as dxi/dx
        adj = dxi_dx.copy()
        grad_delta = np.zeros(n, dtype=float)

        for i in range(n - 1, -1, -1):
            # x_i depends on delta_i with coefficient 1
            grad_delta[i] += adj[i]

            # propagate to parents through f_i
            pa_idx = self.parents[i]
            if pa_idx.size:
                dfi_dpa = local_grads[i]  # shape (len(pa),)
                # adj[parent_k] += adj[i] * d x_i / d x_parent_k
                # and d x_i / d x_parent_k = d f_i / d x_parent_k
                adj[pa_idx] += adj[i] * dfi_dpa

        return x, float(xi), grad_delta


# ----------------------------
# Random polynomial node
# ----------------------------

@dataclass
class RandomPolynomial:
    """
    Polynomial in m variables (parent values).
    Supports degree up to 3 with sparse-ish terms:
      const + linear + squares + pairwise products + cubes
    """
    const: float
    lin: np.ndarray               # (m,)
    sq: np.ndarray                # (m,)
    cross_pairs: List[Tuple[int, int, float]]  # (i,j,coeff) for i<j term coeff*x_i*x_j
    cube: np.ndarray              # (m,)

    def value_and_grad(self, x_pa: np.ndarray) -> Tuple[float, np.ndarray]:
        x = np.asarray(x_pa, dtype=float).reshape(-1)
        m = x.shape[0]
        if self.lin.shape != (m,) or self.sq.shape != (m,) or self.cube.shape != (m,):
            raise ValueError("Polynomial parameter shapes do not match input size")

        # value
        val = self.const
        val += float(self.lin @ x)
        val += float(self.sq @ (x ** 2))
        val += float(self.cube @ (x ** 3))
        for i, j, a in self.cross_pairs:
            val += a * x[i] * x[j]

        # grad
        grad = np.zeros(m, dtype=float)
        grad += self.lin
        grad += 2.0 * self.sq * x
        grad += 3.0 * self.cube * (x ** 2)
        for i, j, a in self.cross_pairs:
            grad[i] += a * x[j]
            grad[j] += a * x[i]

        return float(val), grad

    def to_z3(self, x_pa_z3):
      """
      Build a Z3 expression for the polynomial.
      x_pa_z3: list of Z3 ArithRef expressions (parents in the same order as value_and_grad)
      """
      import z3

      m = len(x_pa_z3)
      # convert floats to exact-ish reals via decimal strings
      rv = lambda a: z3.RealVal(str(float(a)))

      expr = rv(self.const)

      for k in range(m):
          xk = x_pa_z3[k]
          expr += rv(self.lin[k]) * xk
          expr += rv(self.sq[k]) * (xk * xk)
          expr += rv(self.cube[k]) * (xk * xk * xk)

      for i, j, a in self.cross_pairs:
          expr += rv(a) * x_pa_z3[i] * x_pa_z3[j]

      return expr


def random_polynomial_scm_from_mask(
    mask_lower: np.ndarray,
    b: Optional[np.ndarray] = None,
    degree: int = 3,
    coeff_scale: float = 0.5,
    cross_density: float = 0.3,
    seed: Optional[int] = None,
) -> SCM:
    """
    Given a strictly lower-triangular {0,1} mask (i,j)=1 meaning j->i,
    return an SCM whose node i is a random polynomial of its parents.

    Assumes the natural order 0..n-1 is topological (mask is lower-triangular).
    """
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask_lower, dtype=int)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask_lower must be square (n,n)")
    n = mask.shape[0]
    if np.any(np.triu(mask, k=0) != 0):
        raise ValueError("mask_lower must be strictly lower-triangular (zeros on/above diagonal)")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask_lower must be binary (0/1)")
    if degree > 3:
        raise ValueError("RandomPolynomial supports degree <= 3")

    if b is None:
        b = np.zeros(n, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.shape != (n,):
        raise ValueError(f"b must have shape ({n},), got {b.shape}")

    parents: List[np.ndarray] = []
    fns: List[NodeFunction] = []

    for i in range(n):
        pa = np.where(mask[i, :i] == 1)[0].astype(int)
        parents.append(pa)
        m = pa.size

        # If no parents: allow a constant polynomial (still differentiable).
        const = float(rng.normal(0.0, coeff_scale))

        lin = rng.normal(0.0, coeff_scale, size=m) if degree >= 1 else np.zeros(m)
        sq = rng.normal(0.0, coeff_scale, size=m) if degree >= 2 else np.zeros(m)
        cube = rng.normal(0.0, coeff_scale, size=m) if degree >= 3 else np.zeros(m)

        cross_pairs: List[Tuple[int, int, float]] = []
        if degree >= 2 and m >= 2 and cross_density > 0:
            # choose random subset of pairs
            for a in range(m):
                for b_ in range(a + 1, m):
                    if rng.random() < cross_density:
                        coeff = float(rng.normal(0.0, coeff_scale))
                        cross_pairs.append((a, b_, coeff))

        fns.append(RandomPolynomial(const=const, lin=lin, sq=sq, cross_pairs=cross_pairs, cube=cube))

    return SCM(parents=parents, fns=fns, b=b)


