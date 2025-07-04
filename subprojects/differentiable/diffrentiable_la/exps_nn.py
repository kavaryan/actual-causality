from exps import compute_v_and_o_lower_tri

# def soft_intervention(adj, W, X, u, x_prime, alpha):
#     M = np.eye(len(adj)) - adj
#     v_no_int = np.linalg.inv(np.eye(len(adj)) - adj) @ u
    
#     M_int = adj.copy()
#     for i in W|X:
#         M_int[i,:] = 0
#         M_int[i,i] = 1
    
#     if np.linalg.det(M_int) == 0:
#         return None
    
#     M_int_inv = np.linalg.inv(M_int)
    
#     u_int = []
#     for i in range(n):
#         if i in W:
#             u_int.append(float(v_no_int[i]))
#         elif i in X:
#             u_int.append(x_prime[i])
#         else:
#             u_int.append(float(u[i]))
    
#     u_int = np.array(u_int)
#     v_int = M_int_inv @ u_int
#     v_final = np.array([v_no_int[i] if i not in X|W else alpha * v_int[i] + (1 - alpha) * v_no_int[i] for i in range(n)])
#     o_final = o_func(v_final)
#     return v_final, o_final

# def loss(adj, W, X, u, x_prime, alpha=.1):
#     return (soft_intervention(adj, W, X, u, x_prime, alpha) - o_thr) + np.linalg.norm(x_prime) + len(X)

import math, time, torch, numpy as np

# def DiffRobHPScore(pW, pX, x_prime, v_final, o_func, l1, l2, l3):
#     return (
#         o_func(v_final) + # minimize robustness
#         l1*torch.norm(x_prime) + # minimize x′
#         l2*pW.sum() + l3*pX.sum() # minimize |W| + |X|
#     )

# def DiffRobHPScore_o_cf(pW, pX, x_prime, o_cf, l1, l2, l3):
#     return (
#         o_cf + # minimize robustness
#         l1*torch.norm(x_prime) + # minimize x′
#         l2*pW.sum() + l3*pX.sum() # minimize |W| + |X|
#     )


def DiffRobHPScore(pX, v_final, o_func, o_thr, lambda_, nV, temperature=1.0):
    return (
        lambda_ * torch.sigmoid((o_func(v_final) - o_thr)/temperature) + # minimize robustness
        (1 - lambda_) * (pX.sum() / nV) # minimize |W| + |X|
    )

def DiffRobHPScore_parts(pX, v_final, o_func, o_thr, nV, temperature=1.0):
    robustness = o_func(v_final) - o_thr
    sparsity   = pX.sum() / nV
    return robustness, sparsity


class LossBalancer:
    def __init__(self, alpha, lambda_, eps=1e-8):
        self.ema_loss1 = None
        self.ema_loss2 = None
        self.alpha = alpha
        self.lambda_ = lambda_
        self.eps = eps

    def update(self, loss1_val, loss2_val):
        if self.ema_loss1 is None:
            self.ema_loss1 = loss1_val
            self.ema_loss2 = loss2_val
        else:
            self.ema_loss1 = self.alpha * self.ema_loss1 + (1 - self.alpha) * loss1_val
            self.ema_loss2 = self.alpha * self.ema_loss2 + (1 - self.alpha) * loss2_val

    def balanced_loss(self, loss1, loss2):
        # loss1: robustness, loss2: sparcity
        w1 = 1.0 / (self.ema_loss1 + self.eps)
        w2 = 1.0 / (self.ema_loss2 + self.eps)
        return self.lambda_ * w1 * loss1 + (1 - self.lambda_) * w2 * loss2


# ------------------------------------------------------------------ #
# ---------------------------  MODEL  ------------------------------ #
class End2EndIntervention(torch.nn.Module):
    def __init__(self, n, alpha, lambda_, o_func, o_thr, device, expr_name=None):
        super().__init__()
        self.n = n # number of nodes
        self.alpha   = alpha # blending factor for soft intervention
        self.lambda_ = lambda_ # robustness factor for loss
        # 3‑way logits per node: (none, W, X)
        # for each node (endogenous variable) i we have 3 logits, representing the probability of
        #   i being in the set of non‑intervened nodes, the set of intervened nodes W, or the set of 
        #   intervened nodes X
        self.logits   = torch.zeros(n, 3, device=device, requires_grad=True)
        # continuous replacement vector x′
        self.x_prime  = torch.randn(n, device=device, requires_grad=True)
        self.o_func   = o_func
        self.o_thr    = o_thr
        self.device   = device
        self.expr_name = expr_name
        self.epsilon = 1e-4
        self.loss_balancer = LossBalancer(alpha=.9, lambda_=lambda_)

    def forward(self, adj: torch.Tensor, u: torch.Tensor,
                v_no_int: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, v_final)
        adj is passed heree to enable transfer learning between graphs of the same size
        """
        z_soft = torch.nn.functional.gumbel_softmax(self.logits, tau, hard=False)
        pW, pX = z_soft[:, 1], z_soft[:, 2]          # masks in [0,1]
        m      = pW + pX                             # overall intervention mask

        # intervention matrix (row‑wise blend with identity)
        M_int = adj * (1.0 - m).unsqueeze(1) + torch.diag(m)
        # add small epsilon to diagonal to avoid singularity
        M_int += self.epsilon * torch.eye(M_int.size(0), device=self.device)
        # cond_number = torch.linalg.cond(M_int)
        # if cond_number > 1e6:  # heuristically define "too singular"
        #     # skip this step or log warning
        #     # print(f"Warning: condition number {cond_number} too high, using fallback {self.expr_name=}")
        #     # v_int = torch.zeros_like(u)  # or any neutral fallback
        #     rhs = u*(1-m) + pW*v_no_int + pX*self.x_prime
        #     v_int = torch.linalg.lstsq(M_int, rhs).solution
        # else:
        #     v_int = torch.linalg.solve(M_int, u*(1-m) + pW*v_no_int + pX*self.x_prime)


        rhs = u*(1-m) + pW*v_no_int + pX*self.x_prime
        v_int = torch.linalg.lstsq(M_int, rhs).solution
        
        # final node potentials after soft blending
        # v_final = (1-m)*v_no_int + m*(self.alpha*v_int + (1-self.alpha)*v_no_int)
        v_final = v_int

        # objective & regularisation
        # loss = DiffRobHPScore(pW, pX, self.x_prime, v_final, self.o_func, self.l1, self.l2, self.l3)
        # loss = DiffRobHPScore(pX, v_final, self.o_func, self.o_thr, self.lambda_, self.n, temperature=5.0)
        robustness, sparsity = DiffRobHPScore_parts(
            pX, v_final, self.o_func, self.o_thr, self.n, temperature=5.0
        )

        if self.loss_balancer is not None:
            self.loss_balancer.update(robustness.item(), sparsity.item())
            loss = self.loss_balancer.balanced_loss(robustness, sparsity)
        else:
            # fallback if not using balancing
            loss = self.lambda_ * robustness + (1 - self.lambda_) * sparsity
        
        # this makes loss nan
        # loss += self.l4 * -(z_soft * torch.log(z_soft + 1e-9)).sum() # minimize entropy

        return loss, v_final
   

    # helper to extract hard sets after training
    def hard_sets(self):
        decision = self.logits.argmax(dim=1)
        W = {int(i) for i in torch.where(decision == 1)[0]}
        X = {int(i) for i in torch.where(decision == 2)[0]}
        return W, X, self.x_prime.detach().cpu().numpy()


def hp_cause_nn(adj_np, u_np, o_func, o_thr, num_restarts=5, expr_name=None):
    best_res = None
    best_o_cf = float('inf')
    for i in range(num_restarts):
        res = hp_cause_nn_do(adj_np, u_np, o_func, o_thr, expr_name=expr_name)
        if res and res['o_cf'] < best_o_cf:
            best_o_cf = res['o_cf']
            best_res = res
        
    return best_res


def hp_cause_nn_do(adj_np, u_np, o_func, o_thr, expr_name=None):
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    adj     = torch.as_tensor(adj_np, dtype=torch.float32, device=device)
    u       = torch.as_tensor(u_np,   dtype=torch.float32, device=device)
    I       = torch.eye(adj.shape[0], device=device)
    n = adj.shape[0]

    alpha     = 0.9                       # soft‑intervention blend weight
    # lmbdas      = (1e-2, 1e-3, 1e-3, 1e-2)  # (λ1, λ2, λ3, λ4) regularisation
    lambda_ = 0.8
    epochs    = 1000
    lr        = 3e-2
    tau_init  = 1.0
    tau_decay = 0.9995                    # τ ← τ * decay each step
    
    model   = End2EndIntervention(n, alpha, lambda_, o_func, o_thr, device, expr_name=expr_name).to(device)
    optim   = torch.optim.Adam([model.logits, model.x_prime], lr=lr)


    v_no_int = torch.linalg.solve(I - adj, u)

    tau = tau_init
    best_loss = math.inf
    tic = time.time()
    for epoch in range(epochs):
        optim.zero_grad()
        loss, _ = model(adj, u, v_no_int, tau)
        loss.backward()
        optim.step()

        tau = max(tau * tau_decay, 1e-3)  # keep temperature ≥ 0.001

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        #if epoch % 500 == 0 or epoch == 1:
        #    print(f"Epoch {epoch:5d} | τ={τ:.3f} | loss={loss.item():.6f}")

    toc = time.time()
    # print(f"\nTraining finished in {toc - tic:.1f}s. Best loss = {best_loss:.6f}")

    model.load_state_dict(best_state)
    W, X, x_prime_opt = model.hard_sets()

    # print(f"\nSelected hard sets:")
    # print(f"  W = {W}")
    # print(f"  X = {X}")
    # print(f"Optimised x'  = {x_prime_opt}")

    cf = dict(zip(sorted(X),x_prime_opt[sorted(X)]))
    w = v_no_int[sorted(W)]
    W_dict = dict(zip(sorted(W), w))
    v_int, o_cf = compute_v_and_o_lower_tri(adj_np, u_np, cf | W_dict, o_func)
    return dict(cf_X=cf, W=W_dict, o=o_func(v_no_int), o_cf=o_cf)