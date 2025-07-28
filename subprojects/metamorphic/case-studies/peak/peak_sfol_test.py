import unittest
import numpy as np
import matplotlib.pyplot as plt
from peak_sfol import check_peak, plot_signal_with_peak

class TestCheckPeakWithPlotComprehansive(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 10, 100)

    def test_01_peak_pass_all(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 10), ('>', 10), ('>', 0.1), ('<', -0.1), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 1: Peak Pass All")
        self.assertIsNone(result)

    def test_02_peak_fail_a1(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 100), ('>', 10), ('>', 0.1), ('<', -0.1), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 2: Fail a1")
        self.assertIsNone(result)

    def test_03_peak_fail_a2(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 10), ('<', 1), ('>', 0.1), ('<', -0.1), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 3: Fail a2")
        self.assertIsNone(result)

    def test_04_peak_fail_sp1(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 10), ('>', 10), ('>', 1000), ('<', -0.1), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 4: Fail sp1")
        self.assertIsNone(result)

    def test_05_peak_fail_sp2(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 10), ('>', 10), ('>', 0.1), ('>', 0), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 5: Fail sp2")
        self.assertIsNone(result)

    def test_06_peak_fail_width(self):
        s = -1 * (self.t - 5)**2 + 25
        result = check_peak(self.t, s, ('>', 10), ('>', 10), ('>', 0.1), ('<', -0.1), ('<', 0.01))
        plot_signal_with_peak(self.t, s, result, "Test 6: Fail width")
        self.assertIsNone(result)

    def test_07_no_peak_flat(self):
        s = np.ones_like(self.t)
        result = check_peak(self.t, s, ('>', 0), ('>', 0), ('>', 0), ('<', 0), ('>', 0))
        plot_signal_with_peak(self.t, s, result, "Test 7: Flat Signal")
        self.assertIsNone(result)

    def test_08_no_peak_noise(self):
        np.random.seed(0)
        s = np.random.normal(0, 1, len(self.t))
        result = check_peak(self.t, s, ('>', 10), ('>', 10), ('>', 1), ('<', -1), ('>', 2))
        plot_signal_with_peak(self.t, s, result, "Test 8: Random Noise")
        self.assertIsNone(result)

    def test_09_multiple_peaks_one_matches(self):
        s = -1 * (self.t - 3)**2 + 20 + -1 * (self.t - 7)**2 + 20
        result = check_peak(self.t, s, ('>', 5), ('>', 5), ('>', 0.1), ('<', -0.1), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 9: Multiple Peaks, One Match")
        self.assertIsNotNone(result)

    def test_10_small_peak_threshold_fail(self):
        s = -0.1 * (self.t - 5)**2 + 1
        result = check_peak(self.t, s, ('>', 1), ('>', 1), ('>', 0.05), ('<', -0.05), ('>', 1))
        plot_signal_with_peak(self.t, s, result, "Test 10: Small Peak Threshold Fail")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
# unittest.main(argv=[''], verbosity=2, exit=False)
