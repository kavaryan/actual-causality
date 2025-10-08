import tempfile
import os
from core.scm import read_system
from core.hp_modified import find_all_causes_ac1_and_ac2
from search_formulation_test import SuzyBillySearchSpace
from subprojects.metamorphic.search_formulation import hp_cause_bfs

def create_float_multiplication_system():
    """Create a simple SCM system with float domains: z = x * y"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write("""[equations]

# Simple multiplication
z = x * y

[domains]
x, y: Float(0.0, 10.0)
z: Float(0.0, 100.0)
""")
        temp_path = f.name
    
    try:
        system = read_system(temp_path)
        return system
    finally:
        os.unlink(temp_path)

def test_float_hp_cause():
    """Test HP cause finding with float domains"""
    print("\n\nFloat multiplication example:")
    
    # Create SCM system
    system = create_float_multiplication_system()
    
    # Test case: x=4, y=4, so z=16 > 10
    context = {'x': 4.0, 'y': 4.0}
    
    actual_state = system.get_state(context)
    print(f"Actual state: {actual_state}")
    print(f"z = {actual_state['z']}")
    
    # Find all causes of z > 10
    all_causes = find_all_causes_ac1_and_ac2(system, context, 'z', '>', 10.0, include_exo=True)
    
    print(f"\nFound {len(all_causes)} causes of z > 10:")
    for cause in all_causes:
        print(f"  X={cause['X_x_prime']}, W={cause['W']}, w={cause['w']}")
    
    # Verify that both x and y are causes
    x_causes = [c for c in all_causes if 'x' in c['X_x_prime'] and len(c['X_x_prime']) == 1]
    y_causes = [c for c in all_causes if 'y' in c['X_x_prime'] and len(c['X_x_prime']) == 1]
    xy_causes = [c for c in all_causes if 'x' in c['X_x_prime'] and 'y' in c['X_x_prime']]
    
    print(f"\nCauses involving only x: {len(x_causes)}")
    print(f"Causes involving only y: {len(y_causes)}")
    print(f"Causes involving both x and y: {len(xy_causes)}")
    
    # Test with search space approach
    search_space = SuzyBillySearchSpace(system, context, 'z', '>', 10.0)
    bfs_causes = list(hp_cause_bfs(actual_state, search_space))
    print(f"\nBFS found {len(bfs_causes)} minimal causes:")
    for cause in bfs_causes:
        print(f"  {cause}")
    
    return all_causes

if __name__ == '__main__':
    test_float_hp_cause()
    print("\nFloat domain test completed.")
