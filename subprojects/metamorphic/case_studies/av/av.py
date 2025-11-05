from search_formulation import SearchSpace


class AVSearchSpace(SearchSpace):
    def __init__(self, simulator, td_thr, num_vars):
        self.simulator = simulator
        self.td_thr = td_thr
        super().__init__(list(range(num_vars)))

    def is_goal(self, X, v, td_thr=None):
        # Use provided td_thr or fall back to instance variable
        threshold = td_thr if td_thr is not None else self.td_thr
        
        # X represents variables to flip from their original values in v
        # Create the new configuration after interventions
        v_new = v.copy()
        for i in X:
            v_new[i] = 1 - v_new[i]  # flip the bit
        
        num_obstacles = sum(v_new)
        if num_obstacles == 0:
            return False
        td = self.simulator.simulate(num_obstacles)
        
        # # DEBUG: Print details for first few calls
        # if len(X) <= 1:  # Only debug empty set and single-variable sets
        #     print(f"DEBUG is_goal: X={X}, v={list(v)}, v_new={list(v_new)}")
        #     print(f"DEBUG is_goal: num_obstacles={num_obstacles}, td={td}, threshold={threshold}")
        #     print(f"DEBUG is_goal: td <= threshold = {td <= threshold}")
        
        return td <= threshold
