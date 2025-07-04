import hashlib
import multiprocessing as mp
from collections import defaultdict
import random
import time
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
import datetime

from exps import get_random_adj, hp_cause
from exps_nn import hp_cause_nn

def stable_hash(s: str) -> int:
    full_hash = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(full_hash, byteorder='big') % (2**32)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# o_func = lambda v: sum(v)**2 + v[3]**2 + v[4]**2
# def o_func(v):
#      return sum(v)**2 + v[3]**2 + v[4]**2

# class RandomLinearOutput:
#     def __init__(self, n, w_min=-5, w_max=5, seed=42):
#         self.n = n
#         rng = np.random.default_rng(seed)
#         self.w = torch.tensor(rng.uniform(w_min, w_max, (n)), dtype=torch.float32)
#         self.b = rng.uniform(w_min, w_max)
#         print(f"RandomLinearOutput ({seed=}): {self.w=}, {self.b=}")

#     def __call__(self, v):
#         return torch.inner(self.w, v) + self.b


import numpy as np

class RandomLinearOutput:
    def __init__(self, n, w_min=-5, w_max=5, seed=42):
        raise Exception("RandomLinearOutput results in singletone causes, so please use other output functions")
        self.n = n
        rng = np.random.default_rng(seed)
        self.w = np.zeros(n)
        self.w[0:2] = rng.uniform(w_min, w_max, 2)
        self.b = rng.uniform(w_min, w_max)
        print(f"RandomLinearOutput ({seed=}): {self.w=}, {self.b=}")

    def __call__(self, v):
        return np.inner(self.w, v) + self.b


class RandomQuadraticOutput:
    def __init__(self, n, w_min=-5, w_max=5, seed=42):
        self.n = n
        rng = np.random.default_rng(seed)
        self.w = np.zeros(n)
        # self.w[3:6] = rng.uniform(w_min, w_max, 3)
        self.w = rng.uniform(w_min, w_max, n)
        self.b = rng.uniform(w_min, w_max)
        print(f"RandomLinearOutput ({seed=}): {self.w=}, {self.b=}")

    def __call__(self, v):
        # w = torch.from_numpy(self.w).to(v.device)
        # return w @ (v * v) + self.b
        #return sum(v)**2 + v[3]**2 + v[4]**2

        return sum(v) ** 2 + sum(wi * vi * vi for wi, vi in zip(self.w, v))# + self.b



def one_expr(n, dag_method, dag_method_arg, o_func, expr_name):
    adj_res = get_random_adj(n, dag_method, dag_method_arg, o_func)
    adj_np = adj_res['adj']
    u_np = adj_res['u']
    o_thr = adj_res['o_thr']
    # v_np = adj_res['v']
    o_np = adj_res['o']

    if n <= 10:
        tic = time.time()
        a_cause_hp = next(hp_cause(adj_np, u_np, o_func, o_thr))
        toc = time.time()
        delta_hp = toc - tic
    else:
        a_cause_hp = dict(cf_X={}, W={}, o=100, o_cf=100)
        delta_hp = 100

    tic = time.time()
    a_cause_hp_nn = hp_cause_nn(adj_np, u_np, o_func, o_thr, expr_name=expr_name)
    toc = time.time()
    time_hp_nn = toc - tic

    return dict(
        adj=adj_np,
        u=u_np,
        o_thr=o_thr,
        #v=v_np,
        o=o_np,
        a_cause_hp=a_cause_hp,
        delta_hp=delta_hp,
        a_cause_hp_nn=a_cause_hp_nn,
        delta_hp_nn=time_hp_nn,
    )


def run_one_expr(args):
    n, dag_method, dag_method_arg, o_func, expr_name = args

    seed_everything(stable_hash(expr_name))

    return one_expr(n, dag_method, dag_method_arg, o_func, expr_name=expr_name)

# Multiprocessing setup
def main():
    # Parameters
    # n_values = [15, 20, 50, 100]
    n_values = [100]
    num_repeats = 4
    results = defaultdict(list)


    os.makedirs('results', exist_ok=True)

    # seed = 42
    seed = int(time.time())
    seed_everything(seed)
    o_func = RandomQuadraticOutput(n_values[0], w_min=0.9, w_max=1.1, seed=seed)

    args_list = []
    for n in n_values:
        args_list.extend([(n, 'er', 0.4, o_func, f'er/{n=}/{i}') for i in range(num_repeats)])

    with mp.Pool(mp.cpu_count()) as pool:
        all_results = list(tqdm(pool.imap(run_one_expr, args_list), total=len(args_list)))

    # all_results = [run_one_expr(args_list[0])]
    # num_repeats = 1

    idx = 0
    for n in n_values:
        for _ in range(num_repeats):
            results[n].append(all_results[idx])
            idx += 1

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'results/results_{current_time}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {filepath}")

    from parse_pickle import parse_pickle_2
    parse_pickle_2(filepath)


if __name__ == '__main__':
    main()