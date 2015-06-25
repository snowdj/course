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

    @staticmethod
    def set_initial_guess(paras, START):
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

    def plot_solution(self, paras):
        """ Plot the solution of the estimation
            run.
        """
        # Distribute class attributes
        exog = self.exog
        endog = self.endog

        # Initialize grid
        u = np.linspace(exog.min(), exog.max(), 100)
        v = np.exp(-paras[0]*u)/(paras[1] + paras[2]*u)

        # Initialize canvas
        ax = plt.figure(figsize=(12,8)).add_subplot(111)

        # Plot execution times by implementations
        ax.plot(exog, endog, 'ro', label='Observed')
        ax.plot(u, v, 'b-', label='Predicted')

        # Set axis labels
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)

        # Change background color
        ax.set_axis_bgcolor('white')

        # Set up legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False, ncol=2,
            fontsize=20)

        # Remove first element on y-axis
        ax.yaxis.get_major_ticks()[0].set_visible(False)

        # Add title
        plt.suptitle('Inspecting Model Fit', fontsize=20)

        # Show plot
        plt.show()

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
        exog = self.exog
        endog = self.endog

        # Calculate deviations
        dev = endog - np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog)

        # Finishing
        return dev