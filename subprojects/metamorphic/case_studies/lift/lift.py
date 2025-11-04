from search_formulation import SearchSpace


class LiftSearchSpace(SearchSpace):
    def __init__(self, simulate_lifts_func, awt_thr, num_vars):
        self.simulate_lifts_func = simulate_lifts_func
        self.awt_thr = awt_thr
        super().__init__(list(range(num_vars)))

    def is_goal(self, X, v, awt_thr=None):
        # Use provided awt_thr or fall back to instance variable
        threshold = awt_thr if awt_thr is not None else self.awt_thr
        
        # X represents variables to flip from their original values in v
        # Create the new configuration after interventions
        v_new = v.copy()
        for i in X:
            v_new[i] = 1 - v_new[i]  # flip the bit
        
        num_lifts = sum(v_new)
        if num_lifts == 0:
            return False
        awt = self.simulate_lifts_func(num_lifts)
        
        # # DEBUG: Print details for first few calls
        # if len(X) <= 1:  # Only debug empty set and single-variable sets
        #     print(f"DEBUG is_goal: X={X}, v={list(v)}, v_new={list(v_new)}")
        #     print(f"DEBUG is_goal: num_lifts={num_lifts}, awt={awt}, threshold={threshold}")
        #     print(f"DEBUG is_goal: awt <= threshold = {awt <= threshold}")
        
        return awt <= threshold
