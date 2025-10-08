import tempfile
import os
from core.scm import read_system
from subprojects.metamorphic.search_formulation import SearchSpace, hp_cause_bfs
from core.hp_modified import find_all_causes_ac1_and_ac2

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

        # Get all causes using find_all_causes_ac1_and_ac2
        self.all_causes = find_all_causes_ac1_and_ac2(scm, context, Y, op, y_thr, include_exo=False)
    
    def check_op(self, y_actual, op, y_thr):
        """Check if y_actual op y_thr holds."""
        assert op in ['==', '!=', '<=', '<', '>=', '>']
        return eval(f'{y_actual}{op}{y_thr}')
    
    def is_goal(self, X, v=None):
        """
        Check if intervening on variables in X causes the effect Y op y_thr.
        
        Uses find_all_causes_ac1_and_ac2 to determine if X is a cause.
        """
        # Convert X (set of indices) to variable names
        X_vars = [self.V[i] for i in X] if isinstance(list(X)[0] if X else None, int) else list(X)
        
        if not X_vars:
            return False
        
        # Check if X_vars (as a set) matches any of the found causes
        X_set = frozenset(X_vars)
        
        for cause in self.all_causes:
            cause_vars = frozenset(cause['X_x_prime'].keys())
            if cause_vars == X_set:
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
# BH = BT * (1 - ST) --> this leads to that under {BTu=1, STu=0}, ST=0 becomes a cause of BS=1, because (X,x') = {ST=1}, (W,w) = {SH=0}, Bh=1*(1-1)=0, BS=min(1,0+0)=0
BH = BT * (1 - SH)

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
    
    # Run BFS to find minimal cause
    return list(hp_cause_bfs(actual_state, search_space))

    

if __name__ == '__main__':
    no_throw_causes = test_hp_cause_bfs({'STu': 0, 'BTu': 0})
    assert len(no_throw_causes) == 0  # No cause, bottle not shattered

    bt_only_causes = test_hp_cause_bfs({'STu': 0, 'BTu': 1})
    print(bt_only_causes)
    assert frozenset({'BT'}) in bt_only_causes
    assert frozenset({'BH'}) in bt_only_causes
    # ST is NOT a cause because intervening ST=1 still results in BS=1 (SH=1, so bottle still shatters)
    assert frozenset({'ST'}) not in bt_only_causes
    assert frozenset({'SH'}) not in bt_only_causes

    st_only_causes = test_hp_cause_bfs({'STu': 1, 'BTu': 0})
    assert frozenset({'BT'}) not in st_only_causes
    assert frozenset({'BH'}) not in st_only_causes
    assert frozenset({'ST'}) in st_only_causes
    assert frozenset({'SH'}) in st_only_causes

    both_throw_causes = test_hp_cause_bfs({'STu': 1, 'BTu': 1})
    assert frozenset({'BT'}) not in both_throw_causes
    assert frozenset({'BH'}) not in both_throw_causes
    assert frozenset({'ST'}) in both_throw_causes
    assert frozenset({'SH'}) in both_throw_causes

    print("All tests passed.")
