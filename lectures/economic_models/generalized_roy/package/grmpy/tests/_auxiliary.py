""" This module contains a set of auxiliary functions for the testing
    of the grmpy package
"""

# standard library
import random
import string
import numpy as np

# project library
from tools.user.processing import _add_auxiliary
# Note that I need to explicitly import _add_auxiliary. As it is a function
# private to the processing module, a standard import of the module is not
# sufficient.

# Module-wide variables
MAX_AGENTS = 1000
MAX_MAXITER = 100
MAX_NUM_COVARS_OUT = 5
MAX_NUM_COVARS_COST = 5

''' Main function '''


def random_init(seed=None):
    """ This function simulated a dictionary version of a random
        initialization file. This function already imposes that we have at
        least one covariate in X and Z, and also an intercept is defined.
    """

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Initialize container
    init_dict = dict()

    # Basics
    init_dict['BASICS'] = dict()

    init_dict['BASICS']['agents'] = np.random.random_integers(1, MAX_AGENTS)

    init_dict['BASICS']['source'] = id_generator()

    # Correlation of unobservables
    init_dict['DIST'] = dict()

    for group in ['rho0', 'rho1']:
        init_dict['DIST'][group] = np.random.uniform(-0.5, 0.5)

    # Estimation details
    init_dict['ESTIMATION'] = dict()
    init_dict['ESTIMATION']['algorithm'] = np.random.choice(['bfgs', 'nm'])
    init_dict['ESTIMATION']['start'] = np.random.choice(['random', 'init',
                                                         'auto'])
    init_dict['ESTIMATION']['version'] = np.random.choice(['slow', 'fast',
                                                           'object'])
    init_dict['ESTIMATION']['maxiter'] = \
        np.random.random_integers(0, MAX_MAXITER)

    # Model
    num_coeffs_out = np.random.random_integers(1, MAX_NUM_COVARS_OUT)
    num_coeffs_cost = np.random.random_integers(1, MAX_NUM_COVARS_COST)

    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        init_dict[key_] = dict()

        # Draw standard deviations and intercepts
        init_dict[key_]['sd'] = np.random.uniform(0.1, 0.5)
        init_dict[key_]['int'] = np.random.uniform(-0.1, 0.1)

        # Draw coefficients
        num_coeffs = num_coeffs_out
        if key_ == 'COST':
            num_coeffs = num_coeffs_cost

        init_dict[key_]['coeff'] = []
        for i in range(num_coeffs):
            init_dict[key_]['coeff'] += [np.random.uniform(-0.5, 0.5)]

    # Add auxiliary information
    init_dict = _add_auxiliary(init_dict)

    # Finishing
    return init_dict

''' Auxiliary functions '''
# Note that the name of all auxiliary functions starts with an underscore.
# This ensures that the function is private to the module. A standard import
# of this module will not make this function available.

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

    return ''.join(random.choice(chars) for _ in range(size)) + '.grm.txt'


