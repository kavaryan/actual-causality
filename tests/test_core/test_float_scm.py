import tempfile
import os
from core.scm import read_system

# FIXME: use search formulation
from core.hp_modified import find_all_causes_ac1_and_ac2
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

def create_narrow_domain_system():
    """Create SCM with narrower domains: x in [1,2], y in [0,10]"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write("""[equations]

# Simple multiplication
z = x * y

[domains]
x: Float(1.0, 2.0)
y: Float(0.0, 10.0)
z: Float(0.0, 20.0)
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
    
    return all_causes

def test_narrow_domain_case():
    """Test case with narrower domains: x in [1,2], y in [0,10], x=1.5, y=10, z>9"""
    print("\n\nNarrow domain example:")
    
    # Create SCM system with narrower domains
    system = create_narrow_domain_system()
    
    # Test case: x=1.5, y=10, so z=15 > 9
    context = {'x': 1.5, 'y': 10.0}
    
    actual_state = system.get_state(context)
    print(f"Actual state: {actual_state}")
    print(f"z = {actual_state['z']}")
    
    # Find all causes of z > 9
    all_causes = find_all_causes_ac1_and_ac2(system, context, 'z', '>', 9.0, include_exo=True)
    
    print(f"\nFound {len(all_causes)} causes of z > 9:")
    for cause in all_causes:
        print(f"  X={cause['X_x_prime']}, W={cause['W']}, w={cause['w']}")
    
    # Analyze the causes
    x_causes = [c for c in all_causes if 'x' in c['X_x_prime'] and len(c['X_x_prime']) == 1]
    y_causes = [c for c in all_causes if 'y' in c['X_x_prime'] and len(c['X_x_prime']) == 1]
    xy_causes = [c for c in all_causes if 'x' in c['X_x_prime'] and 'y' in c['X_x_prime']]
    
    print(f"\nCauses involving only x: {len(x_causes)}")
    if x_causes:
        print("  Sample x interventions that prevent z > 9:")
        for c in x_causes[:3]:  # Show first 3
            x_val = c['X_x_prime']['x']
            z_val = x_val * 10.0  # y stays at 10
            print(f"    x={x_val:.2f} -> z={z_val:.2f}")
    
    print(f"Causes involving only y: {len(y_causes)}")
    if y_causes:
        print("  Sample y interventions that prevent z > 9:")
        for c in y_causes[:3]:  # Show first 3
            y_val = c['X_x_prime']['y']
            z_val = 1.5 * y_val  # x stays at 1.5
            print(f"    y={y_val:.2f} -> z={z_val:.2f}")
    
    print(f"Causes involving both x and y: {len(xy_causes)}")
    
    return all_causes

if __name__ == '__main__':
    test_float_hp_cause()
    test_narrow_domain_case()
    print("\nFloat domain test completed.")
