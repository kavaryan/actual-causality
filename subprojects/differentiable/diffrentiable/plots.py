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
            b = np.random.randn(n)
            c = np.zeros(n)
            c[-1] = 1  # sink node
            thr = np.random.uniform(-1, 1)
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
    os.makedirs('images/diffrentiable', exist_ok=True)
    plt.savefig('images/diffrentiable/1_runtime_vs_n.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cause_size_vs_lambda(n=100, p=0.3, lambdas=None, repeats=4):
    if lambdas is None:
        lambdas = np.logspace(-2, 2, 15)
    
    # Store results for each repeat and lambda
    cause_sizes_all = []
    delta_norms_all = []
    
    for _ in range(repeats):
        # Generate one random DAG per repeat
        W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
        b = np.random.randn(n)
        c = np.zeros(n)
        c[-1] = 1  # sink node
        thr = np.random.uniform(-1, 1)
        
        cause_sizes_rep = []
        delta_norms_rep = []
        
        # Run through full lambda spectrum for this DAG
        for lmbda in lambdas:
            result = diffalgo(W, b, c, thr, lmbda)
            cause_sizes_rep.append(result['cause_size'])
            delta_norms_rep.append(result['delta_norm'])
        
        cause_sizes_all.append(cause_sizes_rep)
        delta_norms_all.append(delta_norms_rep)
    
    # Calculate means and stds across repeats
    cause_sizes_all = np.array(cause_sizes_all)
    delta_norms_all = np.array(delta_norms_all)
    
    cause_sizes_mean = np.mean(cause_sizes_all, axis=0)
    cause_sizes_std = np.std(cause_sizes_all, axis=0)
    delta_norms_mean = np.mean(delta_norms_all, axis=0)
    delta_norms_std = np.std(delta_norms_all, axis=0)
    
    # Plot cause sizes
    # plt.figure()
    # plt.errorbar(lambdas, cause_sizes_mean, yerr=cause_sizes_std, marker='o', 
    #              capsize=5, label=r'Nonzero $\delta_j$')
    # plt.xlabel(r'Sparsity parameter $\lambda$')
    # plt.ylabel('Size of discovered cause (nonzero interventions)')
    # plt.title(r'Causal Set Size vs $\lambda$')
    # plt.xscale('log')
    # plt.grid(True)
    # plt.legend()
    
    # Create directory if it doesn't exist
    os.makedirs('images/diffrentiable', exist_ok=True)
    # plt.savefig('images/diffrentiable/2_cause_size_vs_lambda.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # Plot L1 norm
    plt.figure()
    plt.errorbar(lambdas, delta_norms_mean, yerr=delta_norms_std, marker='s', 
                 capsize=5, label=r'$\ell_1$-norm')
    plt.xlabel(r'Sparsity parameter $\lambda$')
    plt.ylabel(r'Intervention norm ($\|\delta\|_1$)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title(r'$\ell_1$-norm of Intervention vs $\lambda$')
    
    # Create directory if it doesn't exist
    os.makedirs('images/diffrentiable', exist_ok=True)
    plt.savefig('images/diffrentiable/3_l1_norm_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_robustness_vs_lambda(n=10, p=0.3, lambdas=None, repeats=4):
    if lambdas is None:
        lambdas = np.logspace(-2, 2, 15)
    
    robustnesses_mean = []
    robustnesses_std = []
    
    for lmbda in lambdas:
        robustnesses_exp = []
        
        for _ in range(repeats):
            W = erdos_renyi_dag(n, p=p, seed=None, draw=False)
            b = np.random.randn(n)
            c = np.zeros(n)
            c[-1] = 1  # sink node
            thr = np.random.uniform(-1, 1)
            result = diffalgo(W, b, c, thr, lmbda)
            robustnesses_exp.append(result['robustness'])
        
        robustnesses_mean.append(np.mean(robustnesses_exp))
        robustnesses_std.append(np.std(robustnesses_exp))
    
    plt.figure()
    plt.errorbar(lambdas, robustnesses_mean, yerr=robustnesses_std, marker='o', capsize=5)
    plt.xlabel(r'Sparsity parameter $\lambda$')
    plt.ylabel('Robustness $c^T x - thr$')
    plt.xscale('log')
    plt.grid(True)
    plt.title(r'Discovered Cause Robustness vs $\lambda$')
    
    # Create directory if it doesn't exist
    os.makedirs('images/diffrentiable', exist_ok=True)
    plt.savefig('images/diffrentiable/4_robustness_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

# ================== Example usage ======================
if __name__ == "__main__":
    # --- Plot 1: Runtime vs n
    n_list = [5, 10, 15, 20, 25, 30]
    plot_runtime_vs_n(n_list, p=0.3, lmbda=10.0, repeats=10)

    # --- Plot 2: Cause size vs lambda (4 experiments per lambda)
    lambdas = np.logspace(-2, 2, 15)
    plot_cause_size_vs_lambda(n=100, p=0.3, lambdas=lambdas, repeats=10)

    # --- Plot 3: Robustness vs lambda (4 experiments per lambda)
    plot_robustness_vs_lambda(n=10, p=0.3, lambdas=lambdas, repeats=10)
