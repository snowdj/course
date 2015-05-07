#!/usr/bin/env python
""" Driver to illustrate the grmpy package, which allows for the simulation
    and estimation of the generalized Roy model.

    For some basic explanations of the underlying model, please visit our
    organization's presence of GitHub at

        https://github.com/softEcon/course

    and visit the notebook for the Generalized Roy lecture.

"""

# standard library
import os
import sys
import numpy as np

# edit PYTHONPATH
sys.path.insert(0, 'grmpy')

# project library
from tests._auxiliary import random_init

import grmpy as gp

''' Calling the function of the library '''

# Process initialization file
init_dict = gp.process('init.ini')

# Simulate synthetic sample
gp.simulate(init_dict)

# Estimate model
rslt = gp.estimate(init_dict)

# Inspect results
gp.inspect(rslt, init_dict)


''' Testing the alternative implementations of the likelihood function '''

NUM_TESTS = 1

for _ in range(NUM_TESTS):

    # Generate random request
    init_dict = random_init()

    # Ensure same starting value
    init_dict['ESTIMATION']['start'] = 'init'

    # Simulate sample
    gp.simulate(init_dict)

    # Estimate generalize Roy model
    rslt = dict()

    for version in ['slow', 'fast']:

        init_dict['ESTIMATION']['version'] = version

        rslt[version] = gp.estimate(init_dict)['fval']

    # Assert equality of results
    np.testing.assert_allclose(rslt['slow'], rslt['fast'])

    # Cleanup
    os.remove(init_dict['BASICS']['file'])
