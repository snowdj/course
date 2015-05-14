# Introduce Meta Clas later ...

# The rational agent.
import numpy as np

from scipy.optimize import minimize

class AgentCls(object):

    def __init__(self):


        # Goods demand
        self.x1 = None

        self.x2 = None

        # Endowment
        self.y = None

        self.type = None

        # Prices
        self.p1 = None

        self.p2 = None

        self.made_choice = False

        self.u = None

        self.alpha = None

    def set_type(self, type_):

        self.type = type_

    def set_endowment(self, y):
        """ Set the endowment.
        """
        # Antibugging
        assert (isinstance(y, float))
        assert (y >= 0)

        # Set agent endowment
        self.y = y

    def set_preference_parameter(self, alpha):

        self.alpha = alpha

    def choose(self, p1, p2):
        """ Choose goods given agent preferences and budget constraints.
        """


        type_ = self.type

        self.p1 = p1

        self.p2 = p2



        # Determine starting values
        y = self.y


        if type_ == 'rational':

            x0 = np.array([(0.5*y)/p1, (0.5*y)/p2])

            # Construct budget constraint
            constraint_divergence = dict()

            constraint_divergence['type'] = 'eq'

            constraint_divergence['fun'] = self._constraint

            constraints = [constraint_divergence, ]

            # Call constraint-optimizer
            rslt = minimize(self._criterion, x0, method='SLSQP',
                            constraints=constraints)

            assert (rslt['success'] == True)

            # Transformation of result.
            x = rslt['x']**2

            # Type conversion
            x = x.tolist()

        elif type_ == 'random':
            #
            # Random choice on on the budget line.
            #

            # Determine maximum demand of good two.
            max_two = y/p2

            x = [None, None]

            x[1] = float(np.random.uniform(0, max_two))

            x[0] = (y - x[1]*p2)/p1

        else:

            raise AssertionError

        # Update class attributes
        self.x, self.made_choice = x, True

    def get_individual_demand(self):
        """ Get good demands
        """
        # Antibugging
        assert (self.made_choice is True)

        # Copy demands
        rslt = self.x

        # Finishing
        return rslt

    def utility(self, x):
        """ Evaluate utility.
        """
        # Distribute input arguments
        x1, x2 = x

        alpha = self.alpha

        # Utility calculation
        u = (x1 ** alpha)*(x2 ** (1.0 - alpha))

        # Finishing
        return u

    def spending(self, x):
        """ Calculate spending level.
        """
        # Distribute prices
        p1, p2 = self.p1, self.p2

        # Distribute demands
        x1, x2 = x

        # Calculate expenses
        e = x1 * p1 + x2 * p2

        # Finishing
        return e

    ''' Private methods '''

    def _criterion(self, x):
        """ Evaluate utility.
        """
        # Ensure non-negativity of demand
        x = x ** 2

        # Utility calculation
        u = self.utility(x)

        # Finishing
        return -u

    def _constraint(self, x):
        """ Non-negativity constraint for the SLSQP algorithm.
        """
        # Distribute endowment
        y = self.y

        # Ensure non-negativity
        x = x ** 2

        # Calculate savings
        cons = y - self.spending(x)

        # Constraint
        return cons

class EconomyCls(object):

    def __init__(self, agent_objs):

        self.population = agent_objs

        # Check all choosen?

    def get_aggregate_demand(self, p1, p2):

        agent_objs = self.population

        rslt = np.zeros(2)

        for agent_obj in agent_objs:

            agent_obj.choose(p1, p2)

            demand = agent_obj.get_individual_demand()

            rslt += demand

        return rslt

if False:

    NUM_AGENTS = 1

    ENDOWMENT = 10.0

    ALPHA = 0.5

    P1 = 0.1

    # Auxiliary
    price_grid = np.linspace(P1, 1.0, num=25)


    # Simulate agent populations of different types
    agent_objs = dict()

    agent_obj = AgentCls()

           agent_obj.set_type('random')

            agent_obj.set_preference_parameter(ALPHA)

            agent_obj.set_endowment(ENDOWMENT)

            agent_objs[type_] += [agent_obj]