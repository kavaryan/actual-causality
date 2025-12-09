from diffalgo import diffalgo
from graphs import erdos_renyi_dag

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_runtime_vs_n(n_list, p=0.3, lmbda=10.0, repeats=4):
    runtimes_mean = []
    runtimes_std = []
    for n in n_list:
        t_all = []
        for _ in range(repeats):
            W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
            # Make reversion not easy: choose b and c so that robustness0 is large
            b = np.random.uniform(2, 4, size=n)  # larger b
            c = np.zeros(n)
            c[-1] = 1  # sink node
            # Ensure property is satisfied at initial state, then set thr so that reverting is required
            x0 = np.linalg.inv(np.eye(n) - W) @ b
            robustness0 = c @ x0
            margin = np.abs(robustness0) * 0.8 + 1.0  # large margin, so reversion is hard
            thr = robustness0 - margin  # property is satisfied at start (robustness0 > thr), intervention must revert
            result = diffalgo(W, b, c, thr, lmbda)
            t_all.append(result['runtime'])
        runtimes_mean.append(np.mean(t_all))
        runtimes_std.append(np.std(t_all))
    
    plt.figure()
    plt.errorbar(n_list, runtimes_mean, yerr=runtimes_std, marker='o', capsize=5)
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Average runtime (s)')
    plt.title('Runtime of Causal Search vs n')
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs('images/differentiable', exist_ok=True)
    plt.savefig('images/differentiable/1_runtime_vs_n.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cause_size_vs_lambda(n=100, p=0.3, lambdas=None, repeats=4, lmbda_label=r'$\lambda$'):
    if lambdas is None:
        lambdas = np.logspace(-2, 2, 15)
    lambdas = np.asarray(lambdas)

    cause_sizes_all = []
    delta_norms_all = []

    for _ in range(repeats):
        try:
            W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
            # Make reversion not easy: choose b and c so that robustness0 is large
            b = np.random.uniform(2, 4, size=n)  # larger b
            c = np.zeros(n); c[-1] = 1
            # Ensure property is satisfied at initial state, then set thr so that reverting is required
            x0 = np.linalg.inv(np.eye(n) - W) @ b
            robustness0 = c @ x0
            margin = np.abs(robustness0) * 0.8 + 1.0  # large margin, so reversion is hard
            thr = robustness0 - margin  # property is satisfied at start (robustness0 > thr), intervention must revert

            cause_sizes_rep = []
            delta_norms_rep = []

            for lmbda in lambdas:
                res = diffalgo(W, b, c, thr, lmbda)
                cause_sizes_rep.append(res['cause_size'])
                # Fix: recalculate delta_norm from res['delta_star'] to avoid bug
                delta = res.get('delta_star', None)
                if delta is not None:
                    delta_norms_rep.append(np.linalg.norm(delta, 1))
                else:
                    delta_norms_rep.append(np.nan)

            cause_sizes_all.append(cause_sizes_rep)
            delta_norms_all.append(delta_norms_rep)
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            continue

    cause_sizes_all = np.array(cause_sizes_all)
    delta_norms_all = np.array(delta_norms_all)

    cause_sizes_mean = cause_sizes_all.mean(axis=0)
    cause_sizes_std  = cause_sizes_all.std(axis=0)
    delta_norms_mean = delta_norms_all.mean(axis=0)
    delta_norms_std  = delta_norms_all.std(axis=0)

    outdir = 'images/differentiable'
    os.makedirs(outdir, exist_ok=True)

    # plt.figure()
    # plt.errorbar(lambdas, cause_sizes_mean, yerr=cause_sizes_std, marker='o', capsize=5, label=r'Nonzero $\delta_j$')
    # plt.xlabel(fr'Sparsity parameter {lmbda_label}')
    # plt.ylabel('Size of discovered cause (nonzero interventions)')
    # plt.title(fr'Causal Set Size vs {lmbda_label}')
    # plt.xscale('log')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f'{outdir}/2_cause_size_vs_lambda.png', dpi=300, bbox_inches='tight')
    # plt.close()

    plt.figure()
    plt.plot(lambdas, delta_norms_mean, marker='s', label=r'$\ell_1$-norm')
    plt.fill_between(
        lambdas,
        delta_norms_mean - delta_norms_std,
        delta_norms_mean + delta_norms_std,
        color='C0',
        alpha=0.3,
        label='Std. dev.'
    )
    plt.xlabel(fr'Sparsity parameter {lmbda_label}')
    plt.ylabel(r'Intervention norm ($\|\delta\|_1$)')
    # plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title(r'$\ell_1$-norm of Intervention vs $\lambda$')
    plt.savefig(f'{outdir}/3_l1_norm_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'lambdas': lambdas,
        'cause_sizes_mean': cause_sizes_mean,
        'cause_sizes_std': cause_sizes_std,
        'delta_norms_mean': delta_norms_mean,
        'delta_norms_std': delta_norms_std,
        'outdir': outdir
    }

def plot_robustness_vs_lambda(n=10, p=0.3, lambdas=None, repeats=4):
    if lambdas is None:
        lambdas = np.logspace(-2, 2, 15)
    
    robustnesses_mean = []
    robustnesses_std = []
    
    for lmbda in lambdas:
        robustnesses_exp = []
        
        for _ in range(repeats):
            W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
            # Make reversion not easy: choose b and c so that robustness0 is large
            b = np.random.uniform(2, 4, size=n)  # larger b
            c = np.zeros(n)
            c[-1] = 1  # sink node
            # Ensure property is satisfied at initial state, then set thr so that reverting is required
            x0 = np.linalg.inv(np.eye(n) - W) @ b
            robustness0 = c @ x0
            margin = np.abs(robustness0) * 0.8 + 1.0  # large margin, so reversion is hard
            thr = robustness0 - margin  # property is satisfied at start (robustness0 > thr), intervention must revert
            result = diffalgo(W, b, c, thr, lmbda)
            robustnesses_exp.append(result['robustness'])
        
        robustnesses_mean.append(np.mean(robustnesses_exp))
        robustnesses_std.append(np.std(robustnesses_exp))
    
    plt.figure()
    plt.plot(lambdas, robustnesses_mean, marker='o', label='Mean robustness')
    plt.fill_between(
        lambdas,
        np.array(robustnesses_mean) - np.array(robustnesses_std),
        np.array(robustnesses_mean) + np.array(robustnesses_std),
        color='C0',
        alpha=0.3,
        label='Std. dev.'
    )
    plt.xlabel(r'Sparsity parameter $\lambda$')
    plt.ylabel('Robustness $c^T x - thr$')
    # plt.xscale('log')
    plt.grid(True)
    plt.title(r'Discovered Cause Robustness vs $\lambda$')
    plt.legend()
    
    # Create directory if it doesn't exist
    os.makedirs('images/differentiable', exist_ok=True)
    plt.savefig('images/differentiable/4_robustness_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

# ================== Example usage ======================
if __name__ == "__main__":
    # --- Plot 1: Runtime vs n
    n_list = [5, 10, 15, 20, 25, 30]
    plot_runtime_vs_n(n_list, p=0.3, lmbda=10.0, repeats=10)

    # --- Plot 2: Cause size vs lambda (4 experiments per lambda)
    # lambdas = np.logspace(-3, 3, 15)
    lambdas = np.linspace(0, 1000, 25)
    plot_cause_size_vs_lambda(n=100, p=0.3, lambdas=lambdas, repeats=25)

    # --- Plot 3: Robustness vs lambda (4 experiments per lambda)
    plot_robustness_vs_lambda(n=10, p=0.3, lambdas=lambdas, repeats=10)
