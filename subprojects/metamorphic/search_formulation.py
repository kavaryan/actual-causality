import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import abstractmethod
from tqdm import tqdm
import seaborn as sns
import heapq
from collections import deque, defaultdict

# FIXME: remove - removed circular import

class MonotoQual:
    def __init__(self, var: int):
        self.var = var

class RevMonotoQual:
    def __init__(self, var: int):
        self.var = var

class AdditiveBundle:
    def __init__(self, vars: list[int]):
        self.vars = vars

class SearchSpace:
    def __init__(self, V):
        self.V = V

    def neighbors(self, X):
        if not X.issubset(self.V):
            raise ValueError("X must be a subset of V")
        X = set(X) & set(self.V)

        # Flip each element of V once: remove if present, add if absent.
        for v in self.V:
            if v in X:
                yield frozenset(X - {v})   # remove v ####### why go backward? no matter, we have a seen queue
            else:
                yield frozenset(X | {v})   # add v

    @abstractmethod
    def is_goal(self, X, v):
        pass

def hp_cause_bfs(v, search_space: SearchSpace):
    start = frozenset()                     # begin with no flips
    if search_space.is_goal(start, v):
        return start

    frontier = deque([start])
    visited  = {start}

    while frontier:
        X = frontier.popleft()
        for Y in search_space.neighbors(X):
            if Y not in visited:
                if search_space.is_goal(Y, v):  # goal test on generation
                    yield Y
                visited.add(Y)
                frontier.append(Y)
    return None


def hp_cause_mm(v, awt_thr, mms, search_space):
    Q_f = set(mn.var for mn in mms if isinstance(mn, MonotoQual))
    Q_r = set(mn.var for mn in mms if isinstance(mn, RevMonotoQual))
    def h1(X):
        s = 0
        for i in X & Q_f:
            s += (1 - v[i])
        return -s
    
    def h2(X):
        s = 0
        for i in X & Q_f:
            s += -1 if v[i] else 1 
        return -s
    
    h = h2
    
    start = frozenset()
    if search_space.is_goal(start, v):
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

        for Y in search_space.neighbors(X):
            tentative_g = g_curr + 1          # each edge has unit cost
            if tentative_g < g_cost[Y]:
                g_cost[Y] = tentative_g
                f_Y = tentative_g + h(Y)
                if search_space.is_goal(Y, v):    # goal found
                    return Y
                heapq.heappush(pq, (f_Y, tentative_g, Y))

    return None   # no solution within search space


def hp_cause_mm_bundled(V, v, awt_thr, mms, bundles, search_space):
    """
    A* search that flips variables in bundles together to reduce simulator calls.
    
    :param bundles: List of AdditiveBundle objects, each containing vars to flip together
    """
    # Extract individual vars from monotone qualitative constraints
    Q_f = set(mn.var for mn in mms if isinstance(mn, MonotoQual))
    Q_r = set(mn.var for mn in mms if isinstance(mn, RevMonotoQual))
    
    # Create a mapping from variable to its bundle
    var_to_bundle = {}
    for bundle in bundles:
        for var in bundle.vars:
            var_to_bundle[var] = bundle
    
    def h(X):
        # Count minimum bundle flips still needed
        bundles_needed = set()
        for i in Q_f:
            if i not in X and v[i] == 1:
                # This variable needs to be flipped but hasn't been yet
                if i in var_to_bundle:
                    bundles_needed.add(id(var_to_bundle[i]))
                else:
                    bundles_needed.add(f"single_{i}")
        return len(bundles_needed)
    
    def bundle_neighbors(X):
        """Generate neighbors by flipping entire bundles at once"""
        X_set = set(X)
        
        # Try flipping each bundle
        seen_bundles = set()
        for bundle in bundles:
            bundle_id = id(bundle)
            if bundle_id in seen_bundles:
                continue
            seen_bundles.add(bundle_id)
            
            # Check if all vars in bundle are in same state (all in X or all not in X)
            vars_in_X = [var in X_set for var in bundle.vars]
            
            if all(vars_in_X):
                # Remove all vars in bundle
                yield frozenset(X_set - set(bundle.vars))
            elif not any(vars_in_X):
                # Add all vars in bundle
                yield frozenset(X_set | set(bundle.vars))
            else:
                # Bundle is partially flipped - try both adding all and removing all
                yield frozenset(X_set - set(bundle.vars))
                yield frozenset(X_set | set(bundle.vars))
        
        # Also try flipping individual variables not in any bundle
        bundled_vars = set()
        for bundle in bundles:
            bundled_vars.update(bundle.vars)
        
        for v in V:
            if v not in bundled_vars:
                if v in X_set:
                    yield frozenset(X_set - {v})
                else:
                    yield frozenset(X_set | {v})
    
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

        for Y in bundle_neighbors(X):
            tentative_g = g_curr + 1          # each edge has unit cost
            if tentative_g < g_cost[Y]:
                g_cost[Y] = tentative_g
                f_Y = tentative_g + h(Y)
                if search_space.is_goal(Y, v, awt_thr):    # goal found
                    return Y
                heapq.heappush(pq, (f_Y, tentative_g, Y))

    return None   # no solution within search space
