""" This module is used in the development process of the grmpy package.
"""

# standard library
import os
import sys
import numpy as np

# edit PYTHONPATH
sys.path.insert(0, 'grmpy')

# project library
import grmpy as gp
# project library
from tests._auxiliary import random_init
from tools.economics.clsAgent import AgentCls
from tools.optimization.estimation import _load_data, _object_negative_log_likelihood
# Generate random request

if False:
    init_dict = gp.process('init.ini')

    # Simulate synthetic sample
    gp.simulate(init_dict)

    # Load dataset
    Y, D, X, Z, agent_objs = _load_data(init_dict)

    _object_negative_log_likelihood(init_dict, agent_objs)

    # Process initialization file
    init_dict = gp.process('init.ini')

    # Simulate synthetic sample
    gp.simulate(init_dict)

    # Estimate model
    rslt = gp.estimate(init_dict)

if False:
    # project library
    from tests._auxiliary import random_init


    # Set the number of tests to run
    NUM_TESTS = 10

    # Run repeated tests
    while True:

        # Generate random request
        init_dict = random_init()

        # Ensure same starting value. If we choose the random
        # starting values instead, the points of evaluation
        # differ for the slow and fast implementations.
        init_dict['ESTIMATION']['start'] = 'init'

        init_dict['ESTIMATION']['maxiter'] = 1

        # Simulate sample
        gp.simulate(init_dict)

        # Initialize result container
        rslt = dict()

        # Estimate generalized Roy model
        for version in ['slow', 'fast', 'object']:

            init_dict['ESTIMATION']['version'] = version

            rslt[version] = gp.estimate(init_dict)['fval']

        # Assert equality of results
        np.testing.assert_allclose(rslt['slow'], rslt['fast'])

        np.testing.assert_allclose(rslt['slow'], rslt['object'])

        # Cleanup
        os.remove(init_dict['BASICS']['file'])

        print 'next'

        raise NotImplementedError, 'test'



print 'Special case where maxiter = 0 and start = init'

def rmse(rslt):
    """ Calculate the root-mean squared error.
    """
    # Antibugging
    assert (isinstance(rslt, dict))

    # Distribute information
    x_internal = rslt['AUX']['x_internal']
    start_internal = rslt['AUX']['init_values']

    # Calculate statistic
    rslt = ((x_internal - start_internal) ** 2).mean()

    # Antibugging
    assert (np.isfinite(rslt))
    assert (rslt > 0.0)

    # Finishing
    return rslt

init_dict = gp.process('init.ini')

# Simulate synthetic sample
rslt = dict()

for optimizer in ['bfgs', 'nm']:

    # Initialize containers
    rslt[optimizer] = []

    # Ensure same simulated setup
    np.random.seed(123)

    for i in range(10):

        # Increase noise in observed sample
        init_dict['COST']['var'] = 0.01 + i*0.25
        init_dict['TREATED']['var'] = 0.01 + i*0.25
        init_dict['UNTREATED']['var'] = 0.01 + i*0.25

        # Simulate dataset
        gp.simulate(init_dict)

        # Select estimation setup
        init_dict['ESTIMATION']['version'] = 'fast'
        init_dict['ESTIMATION']['maxiter'] = 10000
        init_dict['ESTIMATION']['optimizer'] = optimizer
        init_dict['ESTIMATION']['start'] = 'random'

        # Calculate performance statistic
        stat = rmse(gp.estimate(init_dict))

        # Collect results
        rslt[optimizer] += [stat]
