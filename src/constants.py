"""
Andrew McDonald
constants.py
Constants used throughout src.
"""

# number of contour levels to show
levels = 10

# grid spacing
dx = 0.05

# numerical padding to avoid div by zero errors
div_by_zero_epsilon = 0.01

# sampling noise of observations
sampling_noise = 1

# seed for all random number generators for reproducibility
seed = 1234

# maximum f value for any function in plotting and computations
f_max = 10

# maximum var value for any function in plotting and computations
var_max = 5

# tsp genetic algorithm mutation probability
tsp_mutation = 0.2

# tsp genetic algorithm number of attempts
tsp_max_attempts = 10
