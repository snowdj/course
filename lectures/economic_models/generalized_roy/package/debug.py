# Time access and conversions
import time

# Unix Pattern Extensions
import glob

# System-specific parameters and functions
import sys
import numpy as np
# Operating System Interfaces
import os

# Plotting
import matplotlib.pyplot as plt

# Notebook Displays
from IPython.display import display, HTML, Image

# Importing the grmpy package by editing the PYTHONPATH
sys.path.insert(0, 'grmpy')

# Package
import grmpy as gp

# Hidden function
from tests._auxiliary import random_init

def rmse(rslt):
    """ Calculate the root-mean squared error.
    """
    # Antibugging
    assert (isinstance(rslt, dict))

    # Distribute information
    x_internal = rslt['AUX']['x_internal']
    init_values = rslt['AUX']['init_values']

    # Calculate statistic
    rslt = ((x_internal - init_values) ** 2).mean()

    # Antibugging
    assert (np.isfinite(rslt))
    assert (rslt > 0.0)

    # Finishing
    return rslt

# Ensure recomputability
np.random.seed(456)

# Set grid for varying degree of noise
noise_grid = range(20)

init_dict = random_init()

init_dict['BASICS']['agents'] = 1000
init_dict['ESTIMATION']['version'] = 'fast'
init_dict['ESTIMATION']['maxiter'] = 10000
init_dict['ESTIMATION']['start'] = 'random'
        
# Simulate synthetic sample
rslt = dict()

for algorithm in ['bfgs', 'nm']:

    # Initialize containers
    rslt[algorithm] = []

    # Ensure same simulated setup
    np.random.seed(123)

    for i in noise_grid:

        # Increase noise in observed sample
        for key_ in ['COST', 'TREATED', 'UNTREATED']:
            init_dict[key_]['sd'] = 0.05 + i*0.50
        
        print init_dict
        # Simulate dataset
        gp.simulate(init_dict)

        # Select estimation setup
        init_dict['ESTIMATION']['algorithm'] = algorithm

        # Calculate performance statistic
        stat = rmse(gp.estimate(init_dict)) 

        # Collect results
        rslt[algorithm] += [stat]

# Initialize canvas
ax = plt.figure(figsize=(12,8)).add_subplot(111)

# Plot execution times by implementations
ax.plot(noise_grid, rslt['bfgs'], label='BFGS')
ax.plot(noise_grid, rslt['nm'], label='Nelder-Mead')

# Set axis labels
ax.set_xlabel('Unobserved Variability', fontsize=20)
ax.set_ylabel('Root-Mean-Squared Error', fontsize=20)

# Set up legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
    fancybox=False, frameon=False, shadow=False, ncol=2,
    fontsize=20)

# Remove first element on y-axis
ax.yaxis.get_major_ticks()[0].set_visible(False)

plt.show()