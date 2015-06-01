""" This module contains the class representation of the agents in the
    economy.
"""

__all__ = ['RationalAgent', 'RandomAgent']
# This restricts the imported names to the two agent classes when
# -- from package import * -- is encountered.

# standard library
import numpy as np
from scipy.optimize import minimize

# We collect all integrity checks to an
# external function in an external file.
from _checks import integrity_checks


# Superclass
class _AgentCls(object):

    def __init__(self):
        """ Initialize instance of agent class.
        """

        # Define class attributes
        self.y, self.x = None, None     # Endowment and consumption bundle

        self.alpha = None               # Set preference parameter

    ''' Public Methods'''

    def set_endowment(self, y):
        """ Set the endowment.
        """
        # Antibugging
        integrity_checks('set_endowment', y)

        # Attach endowment as class attribute
        self.y = y

    def set_preference_parameter(self, alpha):
        """ Set the preference parameter.
        """
        # Antibugging
        integrity_checks('set_preference_parameter', alpha)

        # Attach preference parameter as class attribute
        self.alpha = alpha

    def get_individual_demand(self):
        """ Get the agents demand for the goods.
        """
        # Extract demand from class attributes
        rslt = self.x[:]

        # Quality Checks
        integrity_checks('get_individual_demand', rslt)

        # Finishing
        return rslt

    # Static methods do not receive an implicit first argument.
    @ staticmethod
    def spending(x, p1, p2):
        """ Calculate spending level.
        """
        # Antibugging
        integrity_checks('spending', x, p1, p2)

        # Distribute demands
        x1, x2 = x

        # Calculate expenses
        e = x1 * p1 + x2 * p2

        # Finishing
        return e

    def choose(self, p1, p2):
        """ Choose the desired bundle of goods for different agent
            decision rules.
        """
        # Antibugging
        integrity_checks('choose', p1, p2)

        # Distribute class attributes
        y = self.y

        x = self._choose(y, p1, p2)

        # Update class attributes
        self.x = x

# Polymorphism
# ------------
#
# The design of the agent classes also provides an example of a Polymorphism.
# The agents always _choose(), but they _choose() differently. The
# _choose() behavior is polymorphic in the sense that it is realized
# differently depending on the agent's type.
#
    def _choose(self, y, p1, p2):
        """ Actual implementation depends on the type of agent. This method
            is overridden by the relevant method in the subclass.
        """

        raise NotImplementedError('Subclass must implement abstract method')

# Inheritance
# -----------
#
# The two different types of agents are based on the _AgentCls(). It is a
# mechanism to reuse the code and allow for easy extensions.
#

# Subclass of _AgentCls() with random decision rule
class RandomAgent(_AgentCls):

    def _choose(self, y, p1, p2):
        """ Choose a random bundle on the budget line.
        """
        # Antibugging
        integrity_checks('_choose_random_in', y, p1, p2)

        # Determine maximal consumption of good two
        max_two = y / p2

        # Initialize result container
        x = [None, None]

        # Select random bundle
        x[1] = float(np.random.uniform(0, max_two))

        x[0] = (y - x[1] * p2) / p1

        # Quality Checks
        integrity_checks('_choose_random_out', x)

        # Finishing
        return x

# Subclass of _AgentCls() wuth rational decision rule
class RationalAgent(_AgentCls):

    def get_utility(self, x):
        """ Evaluate utility of agent.
        """
        # Distribute input arguments
        x1, x2 = x

        alpha = self.alpha

        # Utility calculation
        u = (x1 ** alpha) * (x2 ** (1.0 - alpha))

        # Finishing
        return u

    def _choose(self, y, p1, p2):
        """ Choose utility-maximizing bundle.
        """
        # Antibugging
        integrity_checks('_choose_rational_in', y, p1, p2)

        # Determine starting values
        x0 = np.array([(0.5 * y) / p1, (0.5 * y) / p2])

        # Construct budget constraint
        constraint_divergence = dict()

        constraint_divergence['type'] = 'eq'

        constraint_divergence['args'] = (p1, p2)

        constraint_divergence['fun'] = self._constraint

        constraints = [constraint_divergence, ]

        # Call constraint-optimizer. Of course, we could determine the
        # optimal bundle directly, but I wanted to illustrate the use of
        # a constraint optimization algorithm to you.
        rslt = minimize(self._criterion, x0, method='SLSQP',
                        constraints=constraints)

        # Check for convergence
        assert (rslt['success'] == True)

        # Transformation of result.
        x = rslt['x'] ** 2

        # Type conversion
        x = x.tolist()

        # Quality Checks
        integrity_checks('_choose_rational_out', x)

        # Finishing
        return x

    def _criterion(self, x):
        """ Evaluate criterion function.
        """
        # Antibugging
        integrity_checks('_criterion', x)

        # Ensure non-negativity of demand
        x = x ** 2

        # Utility calculation
        u = self.get_utility(x)

        # Finishing
        return -u

    def _constraint(self, x, p1, p2):
        """ Non-negativity constraint for the SLSQP algorithm.
        """
        # Antibugging
        integrity_checks('_constraint', x, p1, p2)

        # Distribute endowment
        y = self.y

        # Ensure non-negativity
        x = x ** 2

        # Calculate savings
        cons = y - self.spending(x, p1, p2)

        # Finishing
        return cons