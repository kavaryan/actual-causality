import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import heapq
from collections import deque, defaultdict

class MonotoQual:
    def __init__(self, var: int):
        self.var = var

class RevMonotoQual:
    def __init__(self, var: int):
        self.var = var

class SearchSpace:
    def __init__(self, simulate_lifts_func):
        self.simulate_lifts_func = simulate_lifts_func

    def neighbors(self, X, V):
        X = set(X) & set(V)
        
        # Flip each element of V once: remove if present, add if absent.
        for v in V:
            if v in X:
                yield frozenset(X - {v})   # remove v ####### why go backward? no matter, we have a seen queue
            else:
                yield frozenset(X | {v})   # add v

    def is_goal(self, X, v, awt_thr):
        x = [v[i] for i in X]
        rest = [v[i] for i in range(len(v)) if i not in X]
        x_prime = [0 if x_i else 1 for x_i in x]
        num_lifts = sum(rest) + sum(x_prime)
        # awt = run_lift_simulation_for_lifts(num_lifts)
        awt = self.simulate_lifts_func(num_lifts)
        return awt <= awt_thr

def hp_cause_bfs(V, v, awt_thr, search_space: SearchSpace):
    start = frozenset()                     # begin with no flips
    if search_space.is_goal(start, v, awt_thr):
        return start

    frontier = deque([start])
    visited  = {start}

    while frontier:
        X = frontier.popleft()
        for Y in search_space.neighbors(X, V):
            if Y not in visited:
                if search_space.is_goal(Y, v, awt_thr):  # goal test on generation
                    return Y
                visited.add(Y)
                frontier.append(Y)
    return None


def hp_cause_mm(V, v, awt_thr, mms, search_space: SearchSpace):
    Q_f = set(mn.var for mn in mms if isinstance(mn, MonotoQual))
    Q_r = set(mn.var for mn in mms if isinstance(mn, RevMonotoQual))
    def h(X):
        s = 0
        for i in X:
            if i in Q_f:
                s += (1 - v[i])
        return s
    
    start = frozenset()
    if search_space.is_goal(start, v, awt_thr):
        return start

    # priority queue stores (f = g + h, g, subset)
    pq = [(h(start), 0, start)]
    g_cost = defaultdict(lambda: float('inf'))
    g_cost[start] = 0

    while pq:
        f_curr, g_curr, X = heapq.heappop(pq)

        # Early exit if this entry is stale
        if g_curr != g_cost[X]:
            continue

        for Y in search_space.neighbors(X, V):
            tentative_g = g_curr + 1          # each edge has unit cost
            if tentative_g < g_cost[Y]:
                g_cost[Y] = tentative_g
                f_Y = tentative_g + h(Y)
                if search_space.is_goal(Y, v, awt_thr):    # goal found
                    return Y
                heapq.heappush(pq, (f_Y, tentative_g, Y))

    return None   # no solution within search space
