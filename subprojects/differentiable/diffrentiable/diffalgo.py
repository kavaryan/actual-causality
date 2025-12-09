import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

# !pip install ecos

def algo(W, b, c, thr, lmbda=10.0, return_all=False):
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

def plot_runtime_vs_n(n_list, p=0.3, lmbda=10.0, repeats=10):
    runtimes = []
    for n in n_list:
        t_all = []
        for _ in range(repeats):
            W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
            b = np.random.randn(n)
            c = np.zeros(n)
            c[-1] = 1  # sink node
            thr = np.random.uniform(-1, 1)
            result = algo(W, b, c, thr, lmbda)
            t_all.append(result['runtime'])
        runtimes.append(np.mean(t_all))
    plt.figure()
    plt.plot(n_list, runtimes, marker='o')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Average runtime (s)')
    plt.title('Runtime of Causal Search vs n')
    plt.grid(True)
    plt.show()

def plot_cause_size_vs_lambda(W, b, c, thr, lambdas):
    cause_sizes = []
    delta_norms = []
    for lmbda in lambdas:
        result = algo(W, b, c, thr, lmbda)
        cause_sizes.append(result['cause_size'])
        delta_norms.append(result['delta_norm'])
    # plt.figure()
    # plt.plot(lambdas, cause_sizes, marker='o', label=r'Nonzero $\delta_j$')
    # plt.xlabel(r'Sparsity parameter $\lambda$')
    # plt.ylabel('Size of discovered cause (nonzero interventions)')
    # plt.title(r'Causal Set Size vs $\lambda$')
    # plt.xscale('log')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # Optionally also plot L1 norm:
    plt.figure()
    plt.plot(lambdas, delta_norms, marker='s', label=r'$\ell_1$-norm')
    plt.xlabel(r'Sparsity parameter $\lambda$')
    plt.ylabel(r'Intervention norm ($\|\delta\|_1$)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title(r'$\ell_1$-norm of Intervention vs $\lambda$')
    plt.show()

def plot_robustness_vs_lambda(W, b, c, thr, lambdas):
    robustnesses = []
    for lmbda in lambdas:
        result = algo(W, b, c, thr, lmbda)
        robustnesses.append(result['robustness'])
    plt.figure()
    plt.plot(lambdas, robustnesses, marker='o')
    plt.xlabel(r'Sparsity parameter $\lambda$')
    plt.ylabel('Robustness $c^T x - thr$')
    plt.xscale('log')
    plt.grid(True)
    plt.title(r'Discovered Cause Robustness vs $\lambda$')
    plt.show()

# Example erdos_renyi_dag function:
def erdos_renyi_dag(n, p=0.3, seed=None, draw=True):
    """
    Generates a random lower-triangular adjacency matrix for a DAG.
    """
    rng = np.random.default_rng(seed)
    W = np.tril(rng.random((n, n)) < p, k=-1).astype(float)
    W *= rng.uniform(0.5, 1.5, size=(n, n))  # random edge weights
    if draw:
        import networkx as nx
        G = nx.DiGraph(W)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
    return W

# ================== Example usage ======================
if __name__ == "__main__":
    # --- Plot 1: Runtime vs n
    n_list = [5, 10, 15, 20, 25, 30]
    plot_runtime_vs_n(n_list, p=0.3, lmbda=10.0, repeats=5)

    # --- Plot 2: Cause size vs lambda (single DAG)
    n = 10
    W = erdos_renyi_dag(n, p=0.3, seed=123, draw=False)
    b = np.random.randn(n)
    c = np.zeros(n)
    c[-1] = 1
    thr = np.random.uniform(-1, 1)
    lambdas = np.logspace(-2, 2, 15)
    plot_cause_size_vs_lambda(W, b, c, thr, lambdas)

    # --- Plot 3: Robustness vs lambda (same DAG)
    plot_robustness_vs_lambda(W, b, c, thr, lambdas)