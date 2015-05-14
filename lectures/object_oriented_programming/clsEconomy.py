""" This module contains the class representation of economy.
"""

__all__ = ['EconomyCls']
# This restricts the imported names to the EconomyCls when
# -- from package import * -- is encountered.

# standard library
import numpy as np

# We collect all integrity checks to an
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

        # Initialize result container
        rslt = [0.0, 0.0]

        # Loop over all agents in the population.
        for agent_obj in agent_objs:

            agent_obj.choose(p1, p2)

            demand = agent_obj.get_individual_demand()

            rslt += demand

        # Quality Assurance
        integrity_checks('get_aggregate_demand_out', rslt)

        # Finishing
        return rslt