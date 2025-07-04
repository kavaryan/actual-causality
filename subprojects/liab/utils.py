from itertools import chain, combinations

import numpy as np

def subsets_upto(iterable, k):
    """Generate all subsets up to size k.
    
    Args:
        iterable (iterable): The iterable to generate subsets from.
        k (int): The maximum size of the subsets. Pass -1 to generate all subsets.
    """
    s = list(iterable)
    if k == -1:
        k = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(k+1))


def test_subsets_upto():
    assert list(subsets_upto([1, 2, 3], 2)) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    assert list(subsets_upto([1, 2, 3], -1)) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]


if __name__ == "__main__":
    test_subsets_upto()
