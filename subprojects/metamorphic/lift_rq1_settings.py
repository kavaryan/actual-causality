#!/usr/bin/env python3
"""
Shared settings for RQ1 lift case study experiments.
"""

# Experiment parameters
DEFAULT_NUM_LIFTS_LIST = [1, 2, 3, 4, 5]
DEFAULT_NUM_TRIALS = 5
DEFAULT_AWT_COEFF = 0.8
DEFAULT_TIMEOUT = 5
AVERAGE_MAX_TIME = 2.0
PROB_ACTIVE = 0.7
BUNDLE_SIZE = 5

# Subject classes - 6 speed × 6 call density = 36 combinations
SPEED_CLASSES = {'S1': 0.5, 'S2': 1.0, 'S3': 1.5}
DENSITY_CLASSES = {'C1': 0.5, 'C2': 1.0, 'C3': 1.5}