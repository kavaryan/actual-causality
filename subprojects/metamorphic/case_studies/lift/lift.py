from search_formulation import SearchSpace


class LiftSearchSpace(SearchSpace):
    def __init__(self, simulate_lifts_func, awt_thr, num_vars):
        self.simulate_lifts_func = simulate_lifts_func
        self.awt_thr = awt_thr
        super().__init__(list(range(num_vars)))

    def is_goal(self, X, v, awt_thr=None):
        # Use provided awt_thr or fall back to instance variable
        threshold = awt_thr if awt_thr is not None else self.awt_thr
        
        x = [v[i] for i in X]
        rest = [v[i] for i in range(len(v)) if i not in X]
        x_prime = [0 if x_i else 1 for x_i in x]
        num_lifts = sum(rest) + sum(x_prime)
        if num_lifts == 0:
            return False
        awt = self.simulate_lifts_func(num_lifts)
        return awt <= threshold
