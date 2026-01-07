import sys
from pathlib import Path
import __main__
import os
    
import time
import pickle
from collections import defaultdict

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import sympy as sp
import warnings
import matplotlib.pyplot as plt
from subprojects.liab.k_leg_liab import k_leg_liab
from subprojects.liab.shapley_liab import shapley_liab
from core.random_system import get_rand_system, rerand_system, get_rand_float_vec, get_rand_failure
from core.failure import ClosedHalfSpaceFailureSet
from IPython.display import display, clear_output
from scipy.stats import mannwhitneyu
from scipy.stats import entropy

SEED = 42
NUM_WORKERS = 8
pickle_dir = Path('.')
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
# S = get_rand_system(6, 'linear', seed=SEED)
# T = rerand_system(S, seed=SEED+1)
# print(S)
# print(T)
# print(S.func_type, T.func_type)

# %%
# M = S.induced_scm()
# N = T.induced_scm()
# print(M)
# print(N)
# print(M.func_type, N.func_type)

# %%
def get_exp_unit(args):
    num_vars, seed = args
    rnd = np.random.RandomState(seed)
    while True:
        S = get_rand_system(num_vars, 'linear', rnd=rnd)
        T = rerand_system(S, rnd=rnd)
        # Get all variables from the systems
        all_vars = list(S.U) + list(S.V)
        F = get_rand_failure(all_vars[:2], ClosedHalfSpaceFailureSet, rnd=rnd)
        
        # Generate a random context for exogenous variables
        u = {}
        for var in S.U:
            u[var] = rnd.uniform(-10, 10)  # Random values for exogenous variables
        
        # Convert systems to SCMSystem format
        M = S.induced_scm()
        N = T.induced_scm()
        
        state_m = M.get_state(u)
        state_n = N.get_state(u)
        if not F.contains(state_m) and F.contains(state_n):
            return N, M, u, F
            
def do_exp(args):
    T, S, u, F, ks = args
    k_leg_values, shapley_values, k_leg_times, shapley_times = {}, {}, {}, {}
    start_time = time.time()
    shapley = shapley_liab(T, S, u, F, k=-1)
    shapley_time = time.time() - start_time
    for k in ks:
        start_time = time.time()
        k_leg = k_leg_liab(T, S, u, F, k=k)
        k_leg_times[k] = time.time() - start_time
        shapley_times[k] = shapley_time
        k_leg_values[k], shapley_values[k] = [], []
        for var in k_leg:
            k_leg_values[k].append(k_leg[var])
            if var in shapley:
                shapley_values[k].append(shapley[var])
            else:
                print(f'Warning: {var} not found in Shapley liability')

    return {'k_leg_values': k_leg_values, 'shapley_values': shapley_values,
        'k_leg_times': k_leg_times, 'shapley_times': shapley_times}

def experiment(num_vars, ks=[1,2], num_samples=20):
    print(f'Doing experiments ({num_samples=}) ...')
    units = []
    pbar = tqdm(total=num_samples)
    def update_progress_get_unit(unit):
        if unit:
            units.append(unit)
            pbar.update(1)
    def error_get_unit(e):
        raise e
    if NUM_WORKERS == 1:
        tasks = [(num_vars, i) for i in range(num_samples)]
        for task in tasks:
            try:
                unit = get_exp_unit(task)
                update_progress_get_unit(unit)
            except Exception as e:
                error_get_unit(e)
    else:
        with Pool(NUM_WORKERS) as pool:
            tasks = [(num_vars, i) for i in range(num_samples)]
            for task in tasks:
                pool.apply_async(get_exp_unit, args=(task,), callback=update_progress_get_unit, error_callback=error_get_unit)
            pool.close()
            pool.join()
    pbar.close()
    
    print(f'Processing results for k in {ks} ...')
    exp_results = {'k_leg_values':  defaultdict(list), 'shapley_values':  defaultdict(list),
        'k_leg_times': defaultdict(list), 'shapley_times': defaultdict(list)}
    pbar = tqdm(total=num_samples)
    def update_progress_do_exp(exp_result):
        if exp_result:
            for k in ks:
                for key in exp_result:
                    exp_results[key][k].append(exp_result[key][k])
            pbar.update(1)
    def error_do_exp(e):
        raise e
    if NUM_WORKERS == 1:
        tasks = []
        for unit in tqdm(units):
            T, S, u, F = unit
            tasks.append((T, S, u, F, ks))
        for task in tasks:
            sys.stdout.flush()
            try:
                exp_result = do_exp(task)
                update_progress_do_exp(exp_result)
            except Exception as e:
                error_do_exp(e)
    else:
        with Pool(NUM_WORKERS) as pool:
            tasks = []
            for unit in tqdm(units):
                T, S, u, F = unit
                tasks.append((T, S, u, F, ks))
            for task in tasks:
                sys.stdout.flush()
                pool.apply_async(do_exp, args=(task,), callback=update_progress_do_exp, error_callback=error_do_exp)
            pool.close()
            pool.join()
    pbar.close()
    
    return exp_results

def get_vargha_delaney(n1, n2, U):
    # Calculate Vargha and Delaney A effect size
    A = U / (n1 * n2)

    # Determine the effect size description
    if A >= 0.71 or A == 0:
        return f"large effect"
    elif A >= 0.64:
        return f"medium effect"
    elif A >= 0.56:
        return f"small effect"
    elif A >= 0.44:
        return f"negligible effect"
    else:
        return f"no effect"

def experiment_and_plot(num_vars, ks=[1,2], num_samples=20, use_pickle=False, pickle_dir='.'):
    # Create images directory if it doesn't exist
    images_dir = Path('images')
    images_dir.mkdir(exist_ok=True)
    
    pickle_fn = Path(pickle_dir) / f'{num_vars=}_{ks=}.pickle'
    print(pickle_fn)
    if use_pickle:
        with open(pickle_fn, 'rb') as pickle_fd:
            exp_results = pickle.load(pickle_fd)
    else:
        exp_results = experiment(num_vars, ks=ks, num_samples=num_samples)
        with open(pickle_fn, 'wb') as pickle_fd:
            pickle.dump(exp_results, pickle_fd)
    
    k_leg_values, shapley_values, k_leg_times, shapley_times = tuple(exp_results.values())
    
    ents = defaultdict(list)
    for k in ks:
        for sublist1, sublist2 in zip(k_leg_values[k], shapley_values[k]):
            # try:
            #     mut = max(entropy(sublist1, sublist2), entropy(sublist2, sublist1))
            # except RuntimeWarning:
            #     print(sublist1)
            #     print(sublist2)
            #     print('**', entropy(sublist1, sublist2))
            #     print('==', entropy(sublist2, sublist1))
            #     return
            mut = abs(np.array(sublist1) - sublist2).sum()
            ents[k].append(mut)

    fig, ax = plt.subplots(1, len(k_leg_times), figsize=(4*len(k_leg_times), 4))
    for ki, k in enumerate(k_leg_times):
        ax[ki].hist(ents[k], bins=50, alpha=0.7)
        # ax[ki].hist(shapley_falttened[k], bins=20, alpha=0.7, label=f'Shapley')
        # ax[ki].legend()
        # U, p_value = mannwhitneyu(shapley_falttened[k], k_leg_flattened[k], alternative='two-sided')
        # effect_size = get_vargha_delaney(len(k_leg_flattened[k]), len(shapley_falttened[k]), U)
        # ax[ki].set_title(f'p-vale={p_value:.3f}, effect={effect_size}')
        ax[ki].set_yscale('log')
        X = np.array(ents[k])
        mx = max(X)
        p_value = (len(X) - len(X[X<.2])) / len(X) # .2 of max mean difference which is 1
        print(f'{p_value=}')
        # ax[1, ki].set_ylim([min(muts[k]), max(muts[k])])
    fig.suptitle(f'Liability difference (M={num_vars})', y=0.95)  # Position at the bottom
    fig.tight_layout()
    
    # Save the liability difference plot
    liability_plot_name = f'liability_difference_M{num_vars}_k{"-".join(map(str, ks))}_n{num_samples}.png'
    fig.savefig(images_dir / liability_plot_name, dpi=300, bbox_inches='tight')
    print(f'Saved liability difference plot: {images_dir / liability_plot_name}')

    fig, ax = plt.subplots(1, len(k_leg_times), figsize=(4*len(k_leg_times), 4))
    for ki, k in enumerate(k_leg_times):
        ax[ki].hist(k_leg_times[k], bins=20, alpha=0.7, label=f'{k}-leg')
        ax[ki].hist(shapley_times[k], bins=20, alpha=0.7, label=f'Shapley')
        ax[ki].legend()
        U, p_value = mannwhitneyu(shapley_times[k], k_leg_times[k], alternative='greater')
        effect_size = get_vargha_delaney(len(k_leg_times[k]), len(shapley_times[k]), U)
        ax[ki].set_title(f'p-vale={p_value:.3f}, effect={effect_size}')
    fig.suptitle(f'Computational time (seconds, M={num_vars})', y=0.95)  # Position at the bottom
    fig.tight_layout()
    
    # Save the computational time plot
    time_plot_name = f'computational_time_M{num_vars}_k{"-".join(map(str, ks))}_n{num_samples}.png'
    fig.savefig(images_dir / time_plot_name, dpi=300, bbox_inches='tight')
    print(f'Saved computational time plot: {images_dir / time_plot_name}')

    return exp_results

def reproduce_paper_plots(Ms=[4,5,6,7,8,9,10], ks=[1,2,3], num_samples=100, use_pickle=True):
    """
    Reproduce the box plots from the paper showing liability differences and time differences
    across different numbers of components (M) and k values.
    """
    # Create images directory if it doesn't exist
    images_dir = Path('images')
    images_dir.mkdir(exist_ok=True)
    
    # Collect all experimental results
    all_liability_diffs = {k: [] for k in ks}
    all_time_diffs = {k: [] for k in ks}
    all_Ms = []
    
    for M in Ms:
        print(f"\nProcessing M={M}...")
        
        # Run or load experiment for this M
        pickle_fn = Path(pickle_dir) / f'num_vars={M}_ks={ks}.pickle'
        
        if use_pickle and pickle_fn.exists():
            print(f"Loading from {pickle_fn}")
            with open(pickle_fn, 'rb') as pickle_fd:
                exp_results = pickle.load(pickle_fd)
        else:
            print(f"Running experiment for M={M}")
            exp_results = experiment(M, ks=ks, num_samples=num_samples)
            with open(pickle_fn, 'wb') as pickle_fd:
                pickle.dump(exp_results, pickle_fd)
        
        k_leg_values, shapley_values, k_leg_times, shapley_times = tuple(exp_results.values())
        
        # Calculate liability differences (sum of absolute differences)
        for k in ks:
            liability_diffs_k = []
            time_diffs_k = []
            
            for sublist1, sublist2 in zip(k_leg_values[k], shapley_values[k]):
                # Sum of absolute liability differences
                liability_diff = abs(np.array(sublist1) - np.array(sublist2)).sum()
                liability_diffs_k.append(liability_diff)
            
            for shapley_time, k_leg_time in zip(shapley_times[k], k_leg_times[k]):
                # Shapley time - k-leg time
                time_diff = shapley_time - k_leg_time
                time_diffs_k.append(time_diff)
            
            # Store results with M labels for box plotting
            all_liability_diffs[k].extend([(M, diff) for diff in liability_diffs_k])
            all_time_diffs[k].extend([(M, diff) for diff in time_diffs_k])
    
    # Create the liability difference box plots (like the paper's left plot)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, k in enumerate(ks):
        ax = axes[i]
        
        # Organize data by M for box plotting
        data_by_M = {}
        for M, diff in all_liability_diffs[k]:
            if M not in data_by_M:
                data_by_M[M] = []
            data_by_M[M].append(diff)
        
        # Create box plot
        Ms_sorted = sorted(data_by_M.keys())
        box_data = [data_by_M[M] for M in Ms_sorted]
        
        bp = ax.boxplot(box_data, positions=Ms_sorted, widths=0.6, patch_artist=True)
        
        # Style the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Sum of abs liability diff')
        ax.set_title(f'k={k}')
        ax.set_xticks(Ms_sorted)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 and use appropriate scale
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Comparison of Shapley and k-leg methods in terms of liability difference', fontsize=14)
    plt.tight_layout()
    
    # Save the liability difference plot
    liability_plot_name = f'all_liabs_M{min(Ms)}-{max(Ms)}_k{"-".join(map(str, ks))}_n{num_samples}.png'
    fig.savefig(images_dir / liability_plot_name, dpi=300, bbox_inches='tight')
    print(f'Saved liability difference plot: {images_dir / liability_plot_name}')
    
    # Create the computational time box plots (like the paper's right plot)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, k in enumerate(ks):
        ax = axes[i]
        
        # Organize data by M for box plotting
        data_by_M = {}
        for M, diff in all_time_diffs[k]:
            if M not in data_by_M:
                data_by_M[M] = []
            data_by_M[M].append(diff)
        
        # Create box plot
        Ms_sorted = sorted(data_by_M.keys())
        box_data = [data_by_M[M] for M in Ms_sorted]
        
        bp = ax.boxplot(box_data, positions=Ms_sorted, widths=0.6, patch_artist=True)
        
        # Style the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Shapley time - k-leg time')
        ax.set_title(f'k={k}')
        ax.set_xticks(Ms_sorted)
        ax.grid(True, alpha=0.3)
        
        # Add a dashed line at y=0 for reference
        ax.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    
    fig.suptitle('Comparison of Shapley and k-leg methods in terms of computational time', fontsize=14)
    plt.tight_layout()
    
    # Save the computational time plot
    time_plot_name = f'all_times_M{min(Ms)}-{max(Ms)}_k{"-".join(map(str, ks))}_n{num_samples}.png'
    fig.savefig(images_dir / time_plot_name, dpi=300, bbox_inches='tight')
    print(f'Saved computational time plot: {images_dir / time_plot_name}')
    
    return all_liability_diffs, all_time_diffs

# %%
if __name__ == '__main__':
    # Reproduce the paper plots
    reproduce_paper_plots(Ms=[4,5,6,7,8,9,10], ks=[1,2,3], num_samples=10, use_pickle=True)

# %%
# mannwhitneyu problem (fixed)
# for d in [1,2,3,10,300]:
#     x = np.random.normal(loc=6, scale=0.5, size=1000)
#     y = np.random.normal(loc=6+d, scale=0.5, size=1000)
#     U_lt, p_value = mannwhitneyu(x, y, alternative='less')
#     U_neq, p_value = mannwhitneyu(x, y, alternative='two-sided')

#     print(f'{d=}, {U_lt=}, {U_neq=}')


