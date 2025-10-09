from core.bf import bf
from core.scm import SCMSystem


def test_bf():
    from core.failure import ClosedHalfSpaceFailureSet
    from core.scm import Component, BoundedIntInterval
    
    # Create specification system components
    a_sp = Component('A = a')
    b_sp = Component('B = b')
    c_sp = Component('C = c + A*B')
    
    # Create implementation system components  
    a_im = Component('A = a + 10')
    b_im = Component('B = b + 10')
    c_im = Component('C = c + A*B + 10')

    # Create domains
    domains = {
        'a': BoundedIntInterval(0, 20),
        'b': BoundedIntInterval(0, 20), 
        'c': BoundedIntInterval(0, 20)
    }

    S = SCMSystem([a_sp, b_sp, c_sp], domains)
    T = SCMSystem([a_im, b_im, c_im], domains)
    F = ClosedHalfSpaceFailureSet({'C': (250, 'ge')})
    
    assert bf(T, S, 'A', {'a': 10, 'b': 10, 'c': 10}, F)
    assert bf(T, S, 'B', {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, 'C', {'a': 10, 'b': 10, 'c': 10}, F)
    assert not bf(T, S, ['A', 'C'], {'a': 10, 'b': 10, 'c': 10}, F)

    print("All tests passed!")

if __name__ == "__main__":
    test_bf()