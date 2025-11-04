#!/usr/bin/env python3
"""
Test to isolate the tqdm + matplotlib interaction issue.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

def test_matplotlib_after_tqdm():
    """Test if tqdm causes matplotlib slowdown."""
    print("Testing matplotlib performance after tqdm...")
    
    # First, test matplotlib without tqdm
    print("\n1. Testing matplotlib WITHOUT tqdm:")
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', label='Test')
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_title('Test Plot')
    ax.legend()
    fig.savefig('test_no_tqdm.png', dpi=100)
    plt.close(fig)
    end_time = time.time()
    print(f"✓ Plot saved in {end_time - start_time:.3f} seconds")
    
    # Now run tqdm and then test matplotlib
    print("\n2. Running tqdm operations...")
    for i in tqdm(range(100), desc="Processing"):
        time.sleep(0.01)  # Simulate work
    
    # Nested tqdm like in the original code
    for outer in tqdm(range(4), desc="Problem Sizes"):
        for inner in tqdm(range(10), desc=f"N={outer*10+5}", leave=False):
            time.sleep(0.01)
    
    print("\n3. Testing matplotlib AFTER tqdm:")
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', label='Test')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Plot After TQDM')
    ax.legend()
    fig.savefig('test_after_tqdm.png', dpi=100)
    plt.close(fig)
    end_time = time.time()
    print(f"✓ Plot saved in {end_time - start_time:.3f} seconds")
    
    # Test if clearing tqdm state helps
    print("\n4. Testing matplotlib after clearing tqdm state:")
    tqdm._instances.clear()  # Clear tqdm instances
    
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', label='Test')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Plot After Clearing TQDM')
    ax.legend()
    fig.savefig('test_cleared_tqdm.png', dpi=100)
    plt.close(fig)
    end_time = time.time()
    print(f"✓ Plot saved in {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    test_matplotlib_after_tqdm()
