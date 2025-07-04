import torch

class LinearSCM:
    def __init__(self, B, device):
        # B = torch.from_numpy(B).to(device)
        B = torch.as_tensor(B, dtype=torch.float32, device=device)
        print(B)
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise ValueError("B must be square")
        if not torch.allclose(B, torch.tril(B)):
            raise ValueError("B must be lower triangular")
        self.n = B.shape[0]
        self.B = B
        self.functions = []
        self._build_functions()

    def __call__(self, u):
        if u.shape != (self.n,):
            raise ValueError("u must have shape (n,)")
        x = torch.zeros_like(u)
        for i in range(self.n):
            x[i] = self.functions[i](x) + u[i]
        return x

    def soft_intervention(self, alpha, beta, x_prime, w):
        if not (alpha.shape == beta.shape == x_prime.shape == w.shape == (self.n,)):
            raise ValueError("all inputs must have shape (n,)")
        for i in range(self.n):
            a = alpha[i]
            b = beta[i]
            if a.item() == 0 and b.item() == 0:
                continue
            xp = x_prime[i]
            wi = w[i]
            old_f = self.functions[i]
            def new_f(x, old_f=old_f, a=a, b=b, xp=xp, wi=wi):
                return (1 - a - b) * old_f(x) + a * xp + b * wi
            self.functions[i] = new_f

    def _build_functions(self):
        self.functions.clear()
        for i in range(self.n):
            coeffs = self.B[i, : i + 1].clone()
            def make_f(i_, coeffs_):
                def f(x):
                    return torch.dot(coeffs_, x[: i_ + 1])
                return f
            self.functions.append(make_f(i, coeffs))

def _test():
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    B = torch.tensor([[0., 0., 0.],
                      [0.5, 0., 0.],
                      [-1., 0.3, 0.]])
    scm = LinearSCM(B, device)
    u = torch.tensor([1., 0., 2.])
    x = scm(u)
    expected = torch.tensor([1., 0.5, -1. + 0.3 * 0.5 + 2.])
    assert torch.allclose(x, expected)
    alpha = torch.tensor([0., 1., 0.])
    beta = torch.tensor([0., 0., 0.])
    x_prime = torch.tensor([0., 10., 0.])
    w = torch.tensor([0., 0., 0.])
    scm.soft_intervention(alpha, beta, x_prime, w)
    x2 = scm(u)
    expected2 = torch.tensor([1., 10., -1. + 0.3 * 10. + 2.])
    assert torch.allclose(x2, expected2)
    print("All tests passed.")

if __name__ == "__main__":
    _test()
