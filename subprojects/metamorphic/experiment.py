import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from search_formulation import MonotoQual, RevMonotoQual, SearchSpace, AdditiveBundle, hp_cause_bfs, hp_cause_mm, hp_cause_mm_bundled

def run_exp_individual(N, simulator_func, search_space):
    V = list(range(N))
    v = np.random.randint(0, 2, size=N)
    awt = simulator_func(sum(v))
    awt_thr = awt * 0.9
    print(f"Initially active lifts: {np.sum(v)}/{N}, initial AWT: {awt}, aiming for: {awt_thr}")

    tic = time.time()
    X = hp_cause_bfs(V, v, awt_thr, search_space)
    toc = time.time()
    d_hp = toc - tic
    print('hp done')

    mms = [MonotoQual(i) for i in V]

    tic = time.time()
    hp_cause_mm(V, v, awt_thr, mms, search_space)
    toc = time.time()
    d_hp_mm = toc - tic
    print('hp mm done')

    return {'d_hp': d_hp, 'd_hp_mm': d_hp_mm}


def run_exp_bundled(N, bundle_size, simulator_func, search_space):
    V = list(range(N))
    v = np.random.randint(0, 2, size=N)
    awt = simulator_func(sum(v))
    awt_thr = awt * 0.9
    print(f"N={N}, bundle_size={bundle_size}")
    print(f"Initially active lifts: {np.sum(v)}/{N}, initial AWT: {awt}, aiming for: {awt_thr}")

    # Create bundles
    bundles = []
    for i in range(0, N, bundle_size):
        bundle_vars = list(range(i, min(i + bundle_size, N)))
        bundles.append(AdditiveBundle(bundle_vars))
    
    # All variables are monotone qualitative
    mms = [MonotoQual(i) for i in V]
    
    # Run standard A*
    tic = time.time()
    X_standard = hp_cause_mm(V, v, awt_thr, mms, search_space)
    toc = time.time()
    d_standard = toc - tic
    print(f'Standard A* done in {d_standard:.3f}s')
    
    # Run bundled A*
    tic = time.time()
    X_bundled = hp_cause_mm_bundled(V, v, awt_thr, mms, bundles, search_space)
    toc = time.time()
    d_bundled = toc - tic
    print(f'Bundled A* done in {d_bundled:.3f}s')
    
    return {
        'N': N,
        'bundle_size': bundle_size,
        'd_standard': d_standard,
        'd_bundled': d_bundled,
        'speedup': d_standard / d_bundled if d_bundled > 0 else float('inf')
    }


def draw_hists(results):
    d_hp_list = [res['d_hp'] for res in results]
    d_hp_mm_list = [res['d_hp_mm'] for res in results]

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].hist([d_hp_list, d_hp_mm_list], bins=20, alpha=0.5, label=['d_hp', 'd_hp_mm'], density=True)
    ax[0].legend()
    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Histogram of d_hp and d_hp_mm')
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'

    # Plot KDE for d_hp_list on ax1 (left y-axis), restrict to non-negative values
    sns.kdeplot(d_hp_list, ax=ax[1], color=color1, label='d_hp', fill=True, alpha=0.5, clip=(0, None))
    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_ylabel('Density (d_hp)', color=color1)
    ax[1].tick_params(axis='y', labelcolor=color1)

    # Create a second y-axis for d_hp_mm_list, restrict to non-negative values
    ax2 = ax[1].twinx()
    sns.kdeplot(d_hp_mm_list, ax=ax2, color=color2, label='d_hp_mm', fill=True, alpha=0.5, clip=(0, None))
    ax2.set_ylabel('Density (d_hp_mm)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax[1].set_title('KDE of d_hp and d_hp_mm (with twin axes)')
    
    fig.tight_layout()
    plt.show()
