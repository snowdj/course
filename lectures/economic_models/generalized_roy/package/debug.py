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

# Generate random request
init_dict = random_init()

# Let us make sure to have a setup that is very favourable to
# the performance of our estimator for now. In my experience,
# small unobserved variability in agent choices and outcomes
# and a large agent count does the trick. Of course, later
# you would want to investigate the performance of your
# estimator for more challenging setups.
for key_ in ['COST', 'TREATED', 'UNTREATED']:
    init_dict[key_]['var'] = 0.02

init_dict['BASICS']['agents'] = 10000

# We need to ensure that the random request actually entails
# a serious estimation run.
init_dict['ESTIMATION']['maxiter'] = 0
init_dict['ESTIMATION']['start'] = 'random'
init_dict['ESTIMATION']['version'] = 'fast'
init_dict['ESTIMATION']['optimizer'] = 'bfgs'

# Simulate synthetic sample
gp.simulate(init_dict)

# Estimate model
rslt = gp.estimate(init_dict)
# Write results
gp.inspect(rslt, init_dict)
print rslt['COST']['all']
print init_dict['COST']['all']

