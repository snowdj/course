# Fundamental Numerical Methods
import numpy as np

# System-specific parameters and functions
import sys

# Adding the modules subdirectory
#sys.path.insert(0, 'modules')

# Project library
from clsAgent import *
from clsEconomy import *

NUM_AGENTS = 1000  # Number of agents in the population

ENDOWMENT = 10.0  # Endowments of agents

ALPHA = 0.75      # Utility weights

P1 = 1.0          # Price of first good (Numeraire)

NUM_POINTS = 1   # Number of points for grid of price changes

# Construct grid for price changes.
PRICE_GRID = np.linspace(P1, 10, num=NUM_POINTS)

# Simulate agent populations of different types
agent_objs = dict()

for type_ in ['random', 'rational']:

    agent_objs[type_] = []

    for _ in range(NUM_AGENTS):

        # Specify agent
        if type_ == 'rational':
            agent_obj = RationalAgent()
        elif type_ == 'random':
            agent_obj = RandomAgent()
        else:
            raise AssertionError

        agent_obj.set_preference_parameter(ALPHA)

        agent_obj.set_endowment(ENDOWMENT)

        # Collect a list fo agents, i.e. the population
        agent_objs[type_] += [agent_obj]

# Get market demands for varying price schedules
market_demands = dict()

for type_ in ['random', 'rational']:

    market_demands[type_] = {'demand': [], 'sd': []}

    # Initialize economy with agent of particular types
    economy_obj = EconomyCls(agent_objs[type_])

    # Vary price schedule for the second good.
    for p2 in PRICE_GRID:

        # Get market demand information
        rslt = economy_obj.get_aggregate_demand(P1, p2)

        # Construct average demand for second good
        demand = rslt['demand'][1]/float(NUM_AGENTS)

        # Construct standard deviation for second good
        demand_sd = rslt['sd'][1]

        # Collect demands and standard deviations
        market_demands[type_]['demand'] += [demand]
        market_demands[type_]['sd'] += [demand_sd]