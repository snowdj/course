""" This module contains all functions related to the inspection of the
    results from an estimation run.
"""

# standard library
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# project library
from tools.user.processing import process

def inspect(parameters, init_dict):

    print(init_dict)
    print('\n\n')
    print(parameters)

    print('inspection')

    # Update parameters
