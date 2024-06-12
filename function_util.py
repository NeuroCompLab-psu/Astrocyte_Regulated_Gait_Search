import numpy as np

def alive_time_dependent_learning_rate(lr, progress, critical_progress, transition_width):
    return lr[1] + 1 / (1 + np.exp((progress - critical_progress) / transition_width)) * (lr[0] - lr[1])