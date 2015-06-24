""" Module that contains the class for the illustration
    of the Toolkit for Advanced Optimization.
"""

# standard library
import numpy as np
from tao4py import TAO
from petsc4py import PETSc

class optCls(object):
    """ Class to illustrate the use of the Toolkit for
        Advanced Optimization.
    """
    def __init__(self, num_agents, paras):
        """ Initialize class.
        """
        # Attach attributes
        self.num_agents = num_agents

        self.paras = paras

        # Exogeneous parameter values
        self.num_paras = len(self.paras)

        # Simulate data
        self._simulate()

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

    def set_initial_guess(self, paras, START):
        """ Initialize the initial parameter values
        """
        # Set starting value
        paras[:] = START

    def form_separable_objective(self, tao, paras, f):
        """ Form objective function for the POUNDerS algorithm.
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Attach to PETSc object
        f.array = dev

    def form_objective(self, tao, paras):
        """ Form objective function for Nelder-Mead algorithm.
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Aggregate deviations
        ff = 0
        for i in range(self.num_agents):
            ff += dev[i]**2

        # Finishing
        return ff

    def plot_solution(self, X):
        """ Plot the solution of the estimation
            run.
        """
        # Distribute class attributes
        exog = self.endog
        endog = self.exog

        # Create grid
        u = np.linspace(exog.min(), exog.max(), 100)
        v = np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog)

        # Set up graph
        pylab.plot(exog, endog, 'ro')
        pylab.plot(u, v, 'b-')
        pylab.show()

    ''' Private methods
    '''
    def _simulate(self):
        """ Simulate data.
        """
        # Distribute class attributes
        num_agents = self.num_agents
        paras = self.paras

        # Simulate exogenous data
        exog = np.random.rand(num_agents)
        eps = np.random.normal(size=num_agents)

        # Determine endogenous data
        endog = np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog) + eps

        # Attach data to class instance
        self.exog = exog
        self.endog = endog

    def _get_deviations(self, paras):
        """ Get whole vector of deviations.
        """
        # Distribute class attributes
        exog = self.endog
        endog = self.exog

        # Calculate deviations
        dev = endog - np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog)

        # Finishing
        return dev