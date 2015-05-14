""" This module contains the class representation of economy.
"""

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