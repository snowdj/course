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
raise AssertionError, 'Code Missing'

''' Main function '''


def random_init(seed=None):
    """ This function simulated a dictionary version of a random
        initialization file. This function already imposes that we have at
        least one covariate in X and Z, and also an intercept is defined.
    """

    raise AssertionError, 'Code Missing'



''' Auxiliary functions '''
# Note that the name of all auxiliary functions starts with an underscore.
# This ensures that the function is private to the module. A standard import
# of this module will not make this function available.

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

    return ''.join(random.choice(chars) for _ in range(size)) + '.grm.txt'



