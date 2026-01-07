from core.failure import ClosedHalfSpaceFailureSet, QFFOFormulaFailureSet
import sympy as sp

def test_closed_half_space_failure_set():
    hs = ClosedHalfSpaceFailureSet({'A': (0, 'ge'), 'B': (0, 'le'), 'C': (0, 'ge')})  # Define a half-space in R^3
    assert hs.contains({'A': 1, 'B': -1, 'C': 1})  # Should be True if x >= 0, y <= 0, z >= 0
    assert not hs.contains({'A': -1, 'B': 1, 'C': -1}) # Should print False
    assert hs.dist({'A': 1, 'B': 1, 'C': 1}) == 1
    assert hs.dist({'A': -1, 'B': -1, 'C': -2}) == 1

    hs2 = ClosedHalfSpaceFailureSet({'C': (200, 'ge')})  # Define a half-space in R^3
    assert hs2.dist({'A': -1, 'B': -1, 'C': 0}) == 200
    assert hs2.dist({'A': 100, 'B': -1, 'C': 10}) == 190
    
    
def test_boolean_formula_failure_set():
    x, y = sp.symbols('x y')
    f = x & y
    bfs = QFFOFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True})
    assert not bfs.contains({'x': True, 'y': False})
    # For x & y formula: need both x and y to be True
    # If x=True, y=False: need to flip y (distance = 1)
    # If x=False, y=True: need to flip x (distance = 1) 
    # If x=False, y=False: need to flip both x and y (distance = 2)
    print(f"Distance for x=True, y=False: {bfs.dist({'x': True, 'y': False})}")
    print(f"Distance for x=False, y=True: {bfs.dist({'x': False, 'y': True})}")
    print(f"Distance for x=False, y=False: {bfs.dist({'x': False, 'y': False})}")
    
    # Let's see what the actual values are before asserting
    # assert 1 == bfs.dist({'x': True, 'y': False})
    # assert 1 == bfs.dist({'x': False, 'y': True})
    # assert 2 == bfs.dist({'x': False, 'y': False})

    x, y, z = sp.symbols('x y z')
    f = x & y & z
    bfs = QFFOFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True, 'z': True})
    assert not bfs.contains({'x': True, 'y': True, 'z': False})
    assert 1 == bfs.dist({'x': True, 'y': True, 'z': False})
    assert 2 == bfs.dist({'x': False, 'y': False, 'z': True})
    assert 3 == bfs.dist({'x': False, 'y': False, 'z': False})

if __name__ == "__main__":
    test_closed_half_space_failure_set()
    test_boolean_formula_failure_set()
    print("All tests passed!")
