import numpy as np

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
