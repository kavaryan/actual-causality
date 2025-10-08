import tempfile
import os
from core.scm import read_system
from subprojects.metamorphic.search_formulation import SearchSpace, hp_cause_bfs

class SuzyBillySearchSpace(SearchSpace):
    def __init__(self, scm, context, Y, op, y_thr):
        """
        Initialize search space for Suzy/Billy SCM system.
        
        :param scm: SCM system
        :param context: dict of exogenous variable assignments
        :param Y: target variable name
        :param op: comparison operator ('==', '!=', '<=', '<', '>=', '>')
        :param y_thr: threshold value for Y
        """
        self.scm = scm
        self.context = context
        self.Y = Y
        self.op = op
        self.y_thr = y_thr
        
        # Get actual state to determine what variables can be intervened on
        actual_state = scm.get_state(context)
        self.actual_state = actual_state
        
        # Variables that can be intervened on (endogenous vars except Y)
        candidate_vars = [v for v in scm.endogenous_vars if v != Y]
        super().__init__(candidate_vars)
        print(f"DEBUG: Endogenous vars: {scm.endogenous_vars}")
        print(f"DEBUG: Candidate vars: {candidate_vars}")
    
    def check_op(self, y_actual, op, y_thr):
        """Check if y_actual op y_thr holds."""
        assert op in ['==', '!=', '<=', '<', '>=', '>']
        return eval(f'{y_actual}{op}{y_thr}')
    
    def is_goal(self, X, v=None):
        """
        Check if intervening on variables in X causes the effect Y op y_thr.
        
        This implements the causality check similar to find_all_causes_ac1_and_ac2:
        - X is a set of variable indices to intervene on
        - We need to find interventions that flip Y from satisfying the condition
          to not satisfying it (or vice versa)
        """
        # Convert X (set of indices) to variable names
        X_vars = [self.V[i] for i in X] if isinstance(list(X)[0] if X else None, int) else list(X)
        
        if not X_vars:
            return False
            
        # Get actual values for variables in X
        x_actual = {v: self.actual_state[v] for v in X_vars}
        
        # Try all possible alternative values for variables in X
        # (similar to the logic in find_all_causes_ac1_and_ac2)
        domain_options = []
        for v in X_vars:
            if v not in self.scm.domains:
                return False
            # Get alternative values (different from actual)
            alternatives = [val for val in self.scm.domains[v].all if val != x_actual[v]]
            if not alternatives:
                return False
            domain_options.append(alternatives)
        
        # Try all combinations of alternative values
        from itertools import product
        for combo in product(*domain_options):
            x_prime = dict(zip(X_vars, combo))
            
            # Evaluate new state with intervention
            new_state = self.scm.get_state(self.context, interventions=x_prime)
            
            # Check if Y changed to not satisfy the condition
            if not self.check_op(new_state[self.Y], self.op, self.y_thr):
                return True
                
        return False

def create_suzy_billy_system():
    """Create the Suzy/Billy SCM system."""
    # Create temporary file with system definition
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write("""[equations]

ST = STu
BT = BTu

# Suzy's rock hits if she throws.
SH = ST

# Billy's rock hits only if he throws and Suzy doesn't throw.
BH = BT * (1 - ST)

# The bottle is shattered if either rock hits.
BS = min(1, SH + BH)

[domains]
ST,BT,SH,BH,BS: Int(0,1)
""")
        temp_path = f.name
    
    try:
        system = read_system(temp_path)
        return system
    finally:
        os.unlink(temp_path)

def test_hp_cause_bfs(context):
    """Test hp_cause_bfs with Suzy/Billy example."""
    print("\n\nSuzy/Billy example:")
    
    # Create SCM system
    system = create_suzy_billy_system()
    
    # Test case: STu=0, BTu=1 (only Billy throws)
    # In this case, BS=1 (bottle shatters)
    actual_state = system.get_state(context)
    print(f"Actual state: {actual_state}")
    print(f"BS = {actual_state['BS']}")
    
    # Create search space to find causes of BS == 1
    search_space = SuzyBillySearchSpace(system, context, 'BS', '==', 1)
    
    # Use variable names instead of indices for the search
    V = search_space.V  # This should be ['ST', 'BT', 'SH', 'BH']
    print(f"Variables that can be intervened on: {V}")
    
    # Run BFS to find minimal cause
    results = list(hp_cause_bfs(actual_state, search_space))
    for result in results:
        print(f"Found cause: {result}")

    

if __name__ == '__main__':
    test_hp_cause_bfs({'STu': 0, 'BTu': 1})

    test_hp_cause_bfs({'STu': 1, 'BTu': 1})
