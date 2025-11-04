#!/usr/bin/env python3
"""
Simple test script to check if matplotlib plotting and saving works.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def test_basic_plot():
    """Test basic matplotlib functionality."""
    print("Testing basic matplotlib plot creation and saving...")
    
    try:
        # Create simple data
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        print("Creating figure...")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        print("Adding data to plot...")
        ax.plot(x, y, 'o-', color='blue', label='Test Data')
        
        print("Setting labels...")
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        ax.set_title('Test Plot')
        ax.legend()
        ax.grid(True)
        
        print("Attempting to save plot...")
        fig.savefig('test_plot_basic.png', dpi=100)
        print("✓ Basic plot saved successfully as: test_plot_basic.png")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Basic plot failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_plot():
    """Test minimal plot with no text."""
    print("\nTesting minimal plot with no text...")
    
    try:
        # Create simple data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        print("Creating minimal figure...")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        print("Adding data only...")
        ax.plot(x, y, 'o-', color='red', linewidth=2, markersize=6)
        
        print("Removing all text elements...")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        print("Attempting to save minimal plot...")
        fig.savefig('test_plot_minimal.png', dpi=100)
        print("✓ Minimal plot saved successfully as: test_plot_minimal.png")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Minimal plot failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_backends():
    """Test different matplotlib backends."""
    print("\nTesting different matplotlib backends...")
    
    backends_to_test = ['Agg', 'svg', 'pdf']
    
    for backend in backends_to_test:
        try:
            print(f"Testing {backend} backend...")
            matplotlib.use(backend)
            
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot([1, 2, 3], [1, 4, 2], 'o-')
            
            if backend == 'Agg':
                filename = 'test_plot_agg.png'
            elif backend == 'svg':
                filename = 'test_plot_svg.svg'
            elif backend == 'pdf':
                filename = 'test_plot_pdf.pdf'
            
            fig.savefig(filename)
            print(f"✓ {backend} backend works: {filename}")
            plt.close(fig)
            
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")

def test_font_issues():
    """Test if font rendering is the issue."""
    print("\nTesting font-related issues...")
    
    try:
        # Set very basic font settings
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['font.size'] = 8
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 2, 3], 'o-')
        ax.text(2, 2, 'Test Text', fontsize=8)
        
        print("Attempting to save plot with text...")
        fig.savefig('test_plot_font.png', dpi=72)  # Very low DPI
        print("✓ Font test passed: test_plot_font.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"✗ Font test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("=" * 50)
    print("MATPLOTLIB DIAGNOSTIC TESTS")
    print("=" * 50)
    
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Current backend: {matplotlib.get_backend()}")
    
    # Run tests
    basic_ok = test_basic_plot()
    minimal_ok = test_minimal_plot()
    test_different_backends()
    test_font_issues()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Basic plot: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Minimal plot: {'✓ PASS' if minimal_ok else '✗ FAIL'}")
    
    if minimal_ok:
        print("\n✓ Minimal plotting works - the issue is likely with text rendering")
        print("  Recommendation: Use text-free plots or fix font configuration")
    elif basic_ok:
        print("\n✓ Basic plotting works - the issue might be with complex layouts")
        print("  Recommendation: Simplify plot elements")
    else:
        print("\n✗ All plotting failed - there's a fundamental matplotlib issue")
        print("  Recommendation: Check matplotlib installation or system dependencies")

if __name__ == "__main__":
    main()
