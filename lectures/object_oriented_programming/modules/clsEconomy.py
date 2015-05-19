""" This module contains the class representation of economy.
"""

__all__ = ['EconomyCls']
# This restricts the imported names to the EconomyCls when
# -- from package import * -- is encountered.

# standard library
import numpy as np

# external function in an external file.
from _checks import integrity_checks


class EconomyCls(object):

    def __init__(self, agent_objs):
        """ Initialize instance of economy class and attach the agent
            population as a class attribute.
        """
        # Antibugging
        integrity_checks('__init__', agent_objs)

        # Attach initialization attributes
        self.population = agent_objs

    ''' Public Methods'''

    def get_aggregate_demand(self, p1, p2):
        """ Aggregate demand of the individual agents to the overall demand
            for each good in the economy.
        """
        # Antibugging
        integrity_checks('get_aggregate_demand_in', p1, p2)

        # Distribute class attributes
        agent_objs = self.population

        # Loop over all agents in the population.
        demand_x1 = []
        demand_x2 = []
        for agent_obj in agent_objs:

            agent_obj.choose(p1, p2)

            demand_x1.append(agent_obj.get_individual_demand()[0])
            demand_x2.append(agent_obj.get_individual_demand()[1])

        # Record output
        rslt = {'demand': [np.sum(demand_x1), np.sum(demand_x2)], 'sd': [np.std(demand_x1), np.std(demand_x2)]}

        # Quality Assurance
        integrity_checks('get_aggregate_demand_out', rslt)

        # Finishing
        return rslt