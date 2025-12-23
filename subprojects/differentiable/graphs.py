from typing import Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# (Optional) helper to create a random lower-triangular mask
# ----------------------------
def erdos_renyi_dag(
    n: int,
    p: float,
    n_output_min_parents: int = 3,
    seed: Optional[int] = None
) -> np.ndarray:
    if n_output_min_parents > n - 1:
        raise ValueError("n_output_parents cannot exceed n-1")

    rng = np.random.default_rng(seed)
    M = np.zeros((n, n), dtype=int)

    # fill lower triangular part
    for i in range(n):
        for j in range(i):
            M[i, j] = 1 if rng.random() < p else 0

    # enforce minimum parents for last row
    last_row = n - 1
    parents = np.flatnonzero(M[last_row, :last_row])

    if len(parents) < n_output_min_parents:
        missing = n_output_min_parents - len(parents)
        candidates = np.setdiff1d(np.arange(last_row), parents)
        chosen = rng.choice(candidates, size=missing, replace=False)
        M[last_row, chosen] = 1

    return M


def barabasi_albert_dag_outpref(
    n: int,
    m: int,
    n_output_min_parents: int = 3,
    seed: Optional[int] = None
) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be >= 1")
    if n_output_min_parents > n - 1:
        raise ValueError("n_output_min_parents cannot exceed n-1")

    rng = np.random.default_rng(seed)
    M = np.zeros((n, n), dtype=int)

    outdeg = np.zeros(n, dtype=float)

    for i in range(1, n):
        k = min(m, i)

        weights = outdeg[:i] + 1.0
        probs = weights / weights.sum()

        parents = rng.choice(i, size=k, replace=False, p=probs)
        M[i, parents] = 1
        outdeg[parents] += 1  # being chosen as parent adds an outgoing edge

    # enforce minimum parents for last row
    last_row = n - 1
    parents = np.flatnonzero(M[last_row, :last_row])
    if len(parents) < n_output_min_parents:
        missing = n_output_min_parents - len(parents)
        candidates = np.setdiff1d(np.arange(last_row), parents)

        weights = outdeg[candidates] + 1.0
        probs = weights / weights.sum()
        chosen = rng.choice(candidates, size=missing, replace=False, p=probs)
        M[last_row, chosen] = 1

    return M


def plot_degree_ccdf(degrees, label=None, color=None):
    degrees = np.asarray(degrees)
    k_values = np.arange(degrees.min(), degrees.max() + 2)  # include max+1 for full step
    ccdf = [(degrees >= k).mean() for k in k_values]
    plt.step(k_values, ccdf, where="post", label=label, color=color)

    plt.xlabel("Degree $k$")
    plt.ylabel("P(degree ≥ k)")
    plt.ylim(0, 1.05)
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)

def compare_degree_ccdf(M_er, M_ba, mode="in", figsize=(6, 4)):
    import networkx as nx

    def get_degrees(M, mode):
        n = M.shape[0]
        G = nx.DiGraph()
        for i in range(n):
            for j in range(i):
                if M[i, j]:
                    G.add_edge(j, i)
        if mode == "in":
            return [d for _, d in G.in_degree()]
        elif mode == "out":
            return [d for _, d in G.out_degree()]
        else:
            raise ValueError

    plt.figure(figsize=figsize)
    deg_er = get_degrees(M_er, mode)
    deg_ba = get_degrees(M_ba, mode)
    plot_degree_ccdf(deg_er, label="Erdős–Rényi")
    plot_degree_ccdf(deg_ba, label="Barabási–Albert")
    plt.legend()
    plt.tight_layout()
    plt.show()
