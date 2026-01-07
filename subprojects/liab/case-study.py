import sys
from pathlib import Path
import __main__
import os
import argparse
    
import time
import pickle
from collections import defaultdict

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from subprojects.liab.k_leg_liab import k_leg_liab
from subprojects.liab.shapley_liab import shapley_liab
from core.random_system import get_rand_system, rerand_system, get_rand_failure
from core.failure import ClosedHalfSpaceFailureSet

# %%
def get_exp_unit(args):
    num_vars, seed = args
    rnd = np.random.RandomState(seed)
    while True:
        S = get_rand_system(num_vars, 'linear', rnd=rnd, seed=seed)
        T = rerand_system(S, 'linear', rnd=rnd, seed=seed)
        # Get all variables from the systems
        all_vars = S.vars
        F = get_rand_failure(all_vars[:2], ClosedHalfSpaceFailureSet, rnd=rnd, seed=seed)
        
        # Generate a random context for exogenous variables
        u = {}
        for var in S.exogenous_nl_vars:
            u[var] = rnd.uniform(-10, 10)  # Random values for exogenous variables
        
        # Systems are already SCMSystem objects
        M = S
        N = T
        
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

def experiment(num_vars, ks, num_samples, num_workers):
    print(f'Doing experiments ({num_samples=}) ...')
    units = []
    pbar = tqdm(total=num_samples)
    def update_progress_get_unit(unit):
        if unit:
            units.append(unit)
            pbar.update(1)
    def error_get_unit(e):
        raise e
    if num_workers == 1:
        tasks = [(num_vars, i) for i in range(num_samples)]
        for task in tasks:
            try:
                unit = get_exp_unit(task)
                update_progress_get_unit(unit)
            except Exception as e:
                error_get_unit(e)
    else:
        with Pool(num_workers) as pool:
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
    if num_workers == 1:
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
        with Pool(num_workers) as pool:
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


def reproduce_paper_plots(Ms, ks, num_samples=1000, use_pickle=True, pickle_dir=Path('.'), num_workers=8):
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
        pickle_fn = Path(pickle_dir) / f'num_vars={M}_ks={ks}_n{num_samples}.pickle'
        
        if use_pickle and pickle_fn.exists():
            print(f"Loading from {pickle_fn}")
            with open(pickle_fn, 'rb') as pickle_fd:
                exp_results = pickle.load(pickle_fd)
        else:
            print(f"Running experiment for M={M}")
            exp_results = experiment(M, ks=ks, num_samples=num_samples, num_workers=num_workers)
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
        
        # Set y-axis to start slightly below 0 so zero values are visible
        ax.set_ylim(bottom=-0.05)
    
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
    
    # Calculate and print p-values table (Table 4 from the paper)
    print("\n" + "="*60)
    print("Table 4: p-values for the difference between k-leg and Shapley liabilities")
    print(f"for M={min(Ms)}-{max(Ms)} and k={','.join(map(str, ks))}")
    print("="*60)
    
    # Create the table header
    header = "M".ljust(8)
    for k in ks:
        header += f"k={k}".ljust(12)
    print(header)
    print("-" * len(header))
    
    # Calculate p-values for each M and k combination
    for M in Ms:
        row = str(M).ljust(8)
        
        for k in ks:
            # Get liability differences for this M and k
            liability_diffs_for_M_k = [diff for m, diff in all_liability_diffs[k] if m == M]
            
            if liability_diffs_for_M_k:
                # Calculate p-value as proportion of cases where difference > 0.2 * 2 = 0.4
                # (0.2 of the maximum possible L1 norm difference, which is 2)
                threshold = 0.4
                num_above_threshold = sum(1 for diff in liability_diffs_for_M_k if diff > threshold)
                p_value = num_above_threshold / len(liability_diffs_for_M_k)
                row += f"{p_value:.3f}".ljust(12)
            else:
                row += "N/A".ljust(12)
        
        print(row)
    
    print("="*60)
    print("Note: p-values represent the proportion of cases where the L1 norm")
    print("difference between k-leg and Shapley apportionments exceeds 0.4")
    print("(which is 0.2 times the maximum possible difference of 2.0)")
    
    return all_liability_diffs, all_time_diffs

def parse_args():
    """Parse command line arguments with paper's default values."""
    
    parser = argparse.ArgumentParser(
        description='Reproduce paper results for k-leg liability framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Use paper defaults (M=4-10, k=1-3, 1000 samples)
  %(prog)s --Ms 4 5 6 --ks 1 2 --num-samples 100  # Quick test with fewer components and samples
  %(prog)s --no-pickle --num-workers 4       # Disable caching, use 4 workers
  %(prog)s --Ms 4 --ks 1 --num-samples 10    # Minimal test for debugging
        '''.strip()
    )
    
    parser.add_argument(
        '--Ms', 
        type=int, 
        nargs='+', 
        default=[4, 5, 6, 7, 8, 9, 10],
        help='List of component numbers to test'
    )
    
    parser.add_argument(
        '--ks', 
        type=int, 
        nargs='+', 
        default=[1, 2, 3],
        help='List of k values for k-leg liability'
    )
    
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=1000,
        help='Number of samples per experiment'
    )
    
    parser.add_argument(
        '--no-pickle', 
        action='store_true',
        help='Disable pickle caching (always recompute)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of worker processes for parallel computation'
    )
    
    return parser.parse_args()

# %%
if __name__ == '__main__':
    args = parse_args()
    
    reproduce_paper_plots(
        Ms=args.Ms, 
        ks=args.ks, 
        num_samples=args.num_samples, 
        use_pickle=not args.no_pickle,
        pickle_dir=Path('.'),
        num_workers=args.num_workers
    )
