#!/usr/bin/env python3
"""
Shared configuration settings for AV RQ1 experiments.
Defines subject classes and experimental parameters.
"""

# Default experimental parameters
DEFAULT_NUM_VARS_LIST = [5, 10, 15, 20]
DEFAULT_NUM_TRIALS = 5
DEFAULT_TD_COEFF = 0.8
DEFAULT_TIMEOUT = 2

# Subject classes - 2 speed × 2 distance = 4 classes
SPEED_CLASSES = {
    'slow': 2.5,    # Slower vehicle speed
    'fast': 7.5     # Faster vehicle speed
}

DISTANCE_CLASSES = {
    'short': 5.0,   # Short distance
    'long': 15.0    # Long distance
}

# Simulation parameters
AVERAGE_MAX_TIME = 1.0
PROB_ACTIVE = 0.5  # Probability that each obstacle is active
SIMULATOR_STARTUP_COST = 0.1
