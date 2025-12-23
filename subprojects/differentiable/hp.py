import itertools
import numpy as np
import time

def hp(
    scm,
    effect,                      # assumed LinearEffect: xi(x)=c^T x - thr ; phi is xi>0
    intervenable=None,           # boolean mask of vars allowed in cause X (optional)
    outcome_idx=None,            # default: last variable
    max_cause_size=2,            # cap for HP3 search
    max_witness_size=2,          # cap for W search
    delta_inf_bound=2.0,         # IMPORTANT: bounds for x' via |x'_i - x*_i| <= delta_inf_bound
    require_change=True,         # enforce at least one var in X changes
    z3_timeout_ms=2000,
):
    """
    Returns a dict:
      status: "found" or "not_found"
      X, W: lists of indices
      x_factual: factual state
      x_prime: dict {i: x'_i} for i in X (model)
      do: dict interventions applied (X->x', W->w*)
      x_cf: counterfactual state under do
      robustness: xi(x_cf)
      delta: intervention vector (nonzero only on X) = x' - x*
      delta_norm_l1: L1 norm of that delta on X
      cause_size: |X|
    """
    start_time = time.time()
    try:
        import z3
    except ImportError as e:
        raise ImportError(
            "Z3 is required for hp(). Install with: pip install z3-solver"
        ) from e

    n = scm.n
    if max_cause_size is None:
        max_cause_size = n
    if max_witness_size is None:
        max_witness_size = n
    if outcome_idx is None:
        outcome_idx = n - 1

    if intervenable is None:
        intervenable = np.ones(n, dtype=bool)
    else:
        intervenable = np.asarray(intervenable, dtype=bool).reshape(-1)
        if intervenable.shape != (n,):
            raise ValueError(f"intervenable must have shape ({n},)")

    # Factual world
    x_star = scm.forward(delta=np.zeros(n))
    xi_star, _ = effect.value_and_grad(x_star)
    if not (xi_star > 0):
        print('failed')
        return {
            "status": "hp1_failed",
            "x_factual": x_star,
            "robustness": float(xi_star),
            "delta_norm_l1": 0.0,
            "cause_size": 0,
        }

    # Only search among ancestors of outcome (huge pruning for n=50)
    parents = scm.parents
    anc = set()
    stack = [outcome_idx]
    while stack:
        v = stack.pop()
        for p in parents[v]:
            if p not in anc:
                anc.add(int(p))
                stack.append(int(p))

    # print(anc)
    candidate_vars = [i for i in sorted(anc) if intervenable[i]]
    # heuristic: try strongest first (uses grad wrt delta at factual)
    _, _, grad0 = scm.effect_and_grad_delta(np.zeros(n), effect)
    candidate_vars.sort(key=lambda i: -abs(grad0[i]))
    # print(candidate_vars)

    rv = lambda a: z3.RealVal(str(float(a)))

    def solve_for_X_W(X, W):
        """
        Try to find x'_X within bounds s.t. ¬phi under do(X=x', W=w*).
        """
        X = list(X)
        W = list(W)

        # Z3 vars for X
        xprime_vars = {i: z3.Real(f"xprime_{i}") for i in X}

        s = z3.Solver()
        s.set("timeout", int(z3_timeout_ms))

        # bounds: |x'_i - x*_i| <= delta_inf_bound
        B = float(delta_inf_bound)
        for i in X:
            lo = float(x_star[i] - B)
            hi = float(x_star[i] + B)
            s.add(xprime_vars[i] >= rv(lo))
            s.add(xprime_vars[i] <= rv(hi))

        if require_change:
            s.add(z3.Or([xprime_vars[i] != rv(x_star[i]) for i in X]))

        # Build expressions for all nodes under interventions
        expr = [None] * n
        for i in range(n):
            if i in xprime_vars:
                expr[i] = xprime_vars[i]
            elif i in W:
                expr[i] = rv(x_star[i])  # witness holds W at factual values
            else:
                # structural equation
                pa = parents[i]
                pa_exprs = [expr[int(p)] for p in pa]
                fi = scm.fns[i]
                if not hasattr(fi, "to_z3"):
                    raise TypeError(
                        f"Node function at i={i} has no to_z3(); hp() currently "
                        f"supports RandomPolynomial (or any fn with to_z3)."
                    )
                expr[i] = fi.to_z3(pa_exprs) + rv(scm.b[i])

        # robustness xi(x) = c^T x - thr ; want ¬phi => xi <= 0
        c = np.asarray(effect.c, dtype=float).reshape(-1)
        thr = float(effect.thr)

        xi_expr = rv(0.0)
        for i in range(n):
            if c[i] != 0.0:
                xi_expr += rv(c[i]) * expr[i]
        xi_expr -= rv(thr)

        s.add(xi_expr <= rv(0.0))

        if s.check() != z3.sat:
            return None

        m = s.model()
        xprime = {}
        for i in X:
            val = m.eval(xprime_vars[i], model_completion=True)
            # convert Z3 number to float robustly
            if hasattr(val, "as_fraction"):
                xprime[i] = float(val.as_fraction())
            else:
                # decimal string fallback
                xprime[i] = float(val.as_decimal(30).replace("?", ""))

        return xprime

    # HP3: search minimal |X|
    for k in range(1, max_cause_size + 1):
        # print(f'{k=}')
        for X in itertools.combinations(candidate_vars, k):
            # print(f'{X=}')
            # HP2: search W subsets (disjoint from X)
            remaining = [i for i in range(n) if i not in X]
            # (optional pruning) only consider W among ancestors too (often enough)
            remaining = [i for i in remaining if (i in anc) or (i == outcome_idx)]

            for wsize in range(0, max_witness_size + 1):
                # print(f'{wsize=}')
                for W in itertools.combinations(remaining, wsize):
                    # print(f'{W=}')
                    xprime = solve_for_X_W(X, W)
                    if xprime is None:
                        continue

                    # Build do-map and evaluate counterfactual numerically
                    do = {i: xprime[i] for i in X}
                    do.update({i: float(x_star[i]) for i in W})

                    x_cf = scm.forward_do(do)
                    xi_cf, _ = effect.value_and_grad(x_cf)

                    # delta only on X (for comparison with diff method)
                    delta = np.zeros(n, dtype=float)
                    for i in X:
                        delta[i] = float(xprime[i] - x_star[i])
                    delta_l1 = float(np.linalg.norm(delta, 1))

                    return {
                        "status": "found",
                        "X": list(X),
                        "W": list(W),
                        "x_factual": x_star,
                        "x_prime": xprime,
                        "do": do,
                        "x_cf": x_cf,
                        "robustness": float(xi_cf),
                        "delta": delta,
                        "delta_norm_l1": delta_l1,
                        "cause_size": len(X),
                        "time": time.time() - start_time
                    }

    return {
        "status": "not_found",
        "x_factual": x_star,
        "robustness": float(xi_star),
        "delta_norm_l1": 0.0,
        "cause_size": 0,
        "time": time.time() - start_time
    }

