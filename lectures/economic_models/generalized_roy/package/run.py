#!/usr/bin/env python
""" Driver to illustrate the grmpy package, which allows for the simulation
    and estimation of the generalized Roy model.

    For some basic explanations of the underlying model, please visit our
    organization's presence of GitHub at

        https://github.com/softEcon/course

    and visit the notebook for the Generalized Roy lecture.

"""

# standard library
import sys

# edit PYTHONPATH
sys.path.insert(0, 'grmpy')

# project library
import grmpy as gp

''' Calling the function of the library
'''

# Process initialization file
init_dict = gp.process()

# Simulate synthetic sample
gp.simulate(init_dict)

# Estimate generalize Roy model
rslt = gp.estimate(init_dict)

# Inspect result
gp.inspect(init_dict, rslt)



