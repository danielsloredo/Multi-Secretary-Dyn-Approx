import numpy as np 
from scipy.stats import uniform

n_types = 8
probabilities = uniform.rvs(size = n_types)
probabilities /= probabilities.sum()
rewards = uniform.rvs(scale = 10, size = n_types)
capacity = 5
time_periods = 1

