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

    # Write results
    gp.inspect(rslt, init_dict)

    # Inspect the results
    print rslt

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