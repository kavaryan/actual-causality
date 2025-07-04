from linear_scm import LinearSCM
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from z3 import RealVector, Real, RealVal, Solver, Sum, sat, is_rational_value, is_algebraic_value
import itertools

def all_subsets(iterable):
    s = list(iterable)
    for r in range(len(s)+1):
        for subset in itertools.combinations(s, r):
            yield set(subset)

def generate_random_dag_adjacency_matrix(n, w_min=-5, w_max=5, p=0.3):
    adj_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i):
            if random.random() < p:
                adj_matrix[i][j] = random.random() * (w_max - w_min) + w_min

    return adj_matrix


def draw_dag_from_adjacency_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    dag = nx.DiGraph()

    # Add edges based on adjacency matrix
    for u in range(n):
        for v in range(n):
            if adj_matrix[u][v] != 0:
                dag.add_edge(u, v, weight=adj_matrix[u][v])

    # Try using graphviz layout
    try:
        # pos = nx.nx_pydot.graphviz_layout(dag, prog='dot', args='-Grankdir=LR')
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot", args='-Grankdir=LR')
    except Exception as e:
        # Fallback to spring layout if graphviz layout is not available
        pos = nx.spring_layout(dag)
        print("Graphviz layout not available; using spring layout instead.")

    plt.figure(figsize=(5, 3)) 

    # Draw the graph
    edge_labels = nx.get_edge_attributes(dag, 'weight')
    nx.draw(dag, pos, with_labels=True, node_size=700, node_color="lightblue", arrowsize=20)
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels)

    plt.title("Random DAG")
    plt.show()

def get_float_from_z3_val(val):
    if is_rational_value(val):
        return float(val.as_fraction())
    elif is_algebraic_value(val):
        # Use a decimal approximation
        return float(val.approx(10))  # 10-digit precision
    else:
        return float(val)

def find_u_with_z3(W, o_func, o_thr, distincts, max_val=1000):
    n = len(W)
    u = RealVector('u', n)
    solver = Solver()

    v = np.linalg.inv(np.eye(n) - W) @ u
    o = o_func(v)
    solver.add(o > o_thr)
    
    # for i in range(len(distincts)):
    #     for j in range(i+1, len(distincts)):
    #         solver.add(u[distincts[i]] != u[distincts[j]])

    for i in range(n):
        solver.add(u[i] < max_val)

    solver.add(o < max_val)
    
    if solver.check() == sat:
        model = solver.model()
        u_vals = [get_float_from_z3_val(model.evaluate(u[i], model_completion=True)) for i in range(n)]
        return np.array(u_vals)
    else:
        return None

def generate_er_dag_adj(n, p, w_min, w_max):
    # Erdos-Renyi
    G = nx.erdos_renyi_graph(n, p)
    adj = np.zeros((n, n))

    # rng = np.random.default_rng(seed)
    rng = np.random

    for u, v in G.edges:
        if u > v:
            weight = rng.uniform(w_min, w_max)
            adj[u, v] = weight
        elif v < u:
            weight = rng.uniform(w_min, w_max)
            adj[v, u] = weight

    return adj

def generate_ba_dag_adj(n, m, w_min, w_max):
    # Barabási–Albert
    G = nx.barabasi_albert_graph(n, m)
    adj = np.zeros((n, n))

    # rng = np.random.default_rng(seed)
    rng = np.random

    for u, v in G.edges:
        if u > v:
            weight = rng.uniform(w_min, w_max)
            adj[u, v] = weight
        elif v < u:
            weight = rng.uniform(w_min, w_max)
            adj[v, u] = weight

    return adj


def get_random_linear_scm_and_u(n, dag_method, dag_method_arg, o_func, w_min=-5, w_max=5, n_distincts=4, device=None):
    if dag_method not in ['er', 'ba']:
        raise ValueError("Method must be 'er' or 'ba'")
    
    while True:
        if dag_method == 'er':
            adj = generate_er_dag_adj(n, dag_method_arg, w_min, w_max)
        elif dag_method == 'ba':
            adj = generate_ba_dag_adj(n, dag_method_arg, w_min, w_max)
        
        distincts = np.random.choice(list(range(n)), n_distincts, replace=False)
        o_thr = np.random.uniform(90, 110)
        print('get_random_adj: ', o_thr)
        u = find_u_with_z3(adj, o_func, o_thr, distincts)        
        
        if u is None:
            print(f"get_random_adj: No solution found for {dag_method=}, {dag_method_arg=}, {o_thr=}")
            continue

        scm = LinearSCM(adj, device)
        print(len(set(u)))
        v = np.linalg.inv(np.eye(len(adj)) - adj) @ u
        o = o_func(v)
        # if u is not None and len(set(u)) > 3 and all(abs(u) < 1000) and all((abs(v) > .1) | (abs(v) == 0)):
        #    print('get_random_adj: Found valid u')
        return dict(scm=scm, u=u, v=v, o=o, o_thr=o_thr, distincts=distincts)


# def get_random_adj(n, dag_method, dag_method_arg, o_func, o_thr, w_min=-5, w_max=5, seed=0):
#     if dag_method not in ['er', 'ba']:
#         raise ValueError("Method must be 'er' or 'ba'")
    
#     if dag_method == 'er':
#         adj = generate_er_dag_adj(n, dag_method_arg, w_min, w_max, seed=seed)
#     elif dag_method == 'ba':
#         adj = generate_ba_dag_adj(n, dag_method_arg, w_min, w_max, seed=seed)
    
#     o_thr = 10
#     while True:
#         print(f'{o_thr=}')
#         u = find_u_with_z3(adj, o_func, o_thr)
#         if u is None:            
#             continue

#         print(u)
#         print(len(set(u)))
#         v = np.linalg.inv(np.eye(len(adj)) - adj) @ u
#         o = o_func(v)
#         if u is not None and len(set(u)) > 3 and all(abs(u) < 1000) and all((abs(v) > .1) | (abs(v) == 0)):
#             print('get_random_adj: Found valid u')
#             return dict(adj=adj, u=u, v=v, o=o)
        
#         o_thr += 10

def hp_cause_linear(adj, u, o_func, o_thr): # dict(cf_X=cf, W=dict(zip(W, w)), o=o_func(v), o_cf=o_cf)
    n = len(adj)
    V = set(range(n))
    M = np.eye(n) - adj
    v = np.linalg.inv(M) @ u
    for X in all_subsets(V):
        if not len(X):
            continue
        for W in all_subsets(V - X):
            M_int = M.copy()
            for i in W|X:
                M_int[i,:] = 0
                M_int[i,i] = 1
            
            if np.linalg.det(M_int) == 0:
                # print(f"det=0 for X={X}, W={W}")
                continue
            M_int_inv = np.linalg.inv(M_int)
            solver = Solver()
            u_vars = {i: Real(f'u_int_{i}') for i in X}
            u_int = []
            for i in range(n):
                if i in W:
                    u_int.append(float(v[i]))
                elif i in X:
                    u_int.append(u_vars[i])
                else:
                    u_int.append(float(u[i]))
            
            u_int = np.array(u_int)

            # form counterfactual v_expr = M_int_inv @ u_int
            # this is a list of Z3 expressions
            v_expr = []
            for j in range(n):
                acc = None
                for k in range(n):
                    term = M_int_inv[j,k] * u_int[k]
                    acc = term if acc is None else acc + term
                v_expr.append(acc)

            obj_z3 = o_func(v_expr)      
            solver.add(obj_z3 <= RealVal(o_thr))

            for i in X:
                solver.add(u_vars[i] != RealVal(str(float(v[i]))))
                solver.add(u_vars[i] < 1000)
                solver.add(u_vars[i] > .1)

            if solver.check() == sat:
                model = solver.model()
                # pull out the u_int values for X
                cf = {i: get_float_from_z3_val(model.evaluate(u_vars[i], model_completion=True)) for i in X}
                # u values determine endogenous variables, because even if in the in the oriignal model, Xi
                #  has a corresponding u_i, it is not in the intervention model
                # print(f"Counterfactual for X={X}, W={W}, w={v[list(W)]} → x' = {cf}")
                u_int2 = u_int.copy()
                u_int2[list(cf.keys())] = np.array(list(cf.values()))
                o_cf = o_func(M_int_inv @ u_int2)
                w = map(float, v[list(W)])
                yield dict(cf_X=cf, W=dict(zip(W, w)), o=o_func(v), o_cf=o_cf)
                break

def compute_v_and_o_lower_tri(W, u, ints=None, o_func=None):
    n = len(u)
    v = np.zeros(n)
    ints = ints or {}

    for i in range(n):
        if i in ints:
            v[i] = ints[i]
        else:
            # sum over all j<i such that W[i,j] is nonzero
            s = u[i]
            for j in range(i):
                if W[i,j] != 0:
                    s += W[i,j] * v[j]
            v[i] = s

    return v, (o_func(v) if o_func else None)


