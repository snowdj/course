""" Module that contains the class for the illustration
    of the Toolkit for Advanced Optimization.
"""

# standard library
import numpy as np
import matplotlib.pyplot as plt

from tao4py import TAO
from petsc4py import PETSc


class OptCls(object):
    """ Class to illustrate the use of the Toolkit for
        Advanced Optimization.
    """
    def __init__(self, exog, endog, START):
        """ Initialize class.
        """
        # Attach attributes
        self.exog = exog
        self.endog = endog
        self.start = START

        # Derived attributes
        self.num_agents = len(self.exog)
        self.num_paras = len(self.start)

    def create_vectors(self):
        """ Create instances of PETSc objects.
        """
        # Distribute class attributes
        num_agents = self.num_agents
        num_paras = self.num_paras

        # Create container for parameter values
        paras = PETSc.Vec().create(PETSc.COMM_SELF)
        paras.setSizes(num_paras)

        # Create container for criterion function
        crit = PETSc.Vec().create(PETSc.COMM_SELF)
        crit.setSizes(num_agents)

        # Management
        paras.setFromOptions()
        crit.setFromOptions()

        # Finishing
        return paras, crit

    def set_initial_guess(self, paras):
        """ Initialize the initial parameter values
        """
        # Set starting value
        paras[:] = self.start

    def form_separable_objective(self, tao, paras, f):
        """ Form objective function for the POUNDerS algorithm.
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Attach to PETSc object
        f.array = dev

    def form_objective(self, tao, paras):
        """ Form objective function for Nelder-Mead algorithm. The
            FOR loop results in a costly evaluation of the
            criterion function..
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Aggregate deviations
        ff = 0
        for i in range(self.num_agents):
            ff += dev[i]**2

        # Finishing
        return ff

    ''' Private methods
    '''
    def _get_deviations(self, paras):
        """ Get whole vector of deviations.
        """
        # Distribute class attributes
        exog = self.exog
        endog = self.endog

        # Calculate deviations
        dev = endog - np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog)

        # Finishing
        return dev