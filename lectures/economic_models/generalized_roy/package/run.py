#!/usr/bin/env python
""" Driver to illustrate the grmpy library, which allows for the simulation
    and estimation of the generalized Roy model.

    For some basic explanations of the underlying model, please visit our
    organization's presence of GitHub at

        https://github.com/softEcon/course

    and visit the notebook for the Generalized Roy lecture.

"""

# edit PYTHONPATH
import sys
sys.path.insert(0, 'grmpy')

# project library
import grmpy as gp

''' Calling the function of the library
'''

# Process initialization file
gp.process()

# Simulate synthetic sample
gp.simulate()

# Estimate generalize Roy model
gp.estimate()





