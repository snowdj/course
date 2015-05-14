# Introduce Meta Clas later ...

# The rational agent.
import numpy as np

from clsEconomy import *
from clsAgent import *

NUM_AGENTS = 10   # Number of agents in the population

ENDOWMENT = 10.0  # Endowments of agents

ALPHA = 0.75      # Utility weights

P1 = 1.0         # Price of first good (Numeraire)

NUM_POINTS = 25   # Number of points for grid of proce changes

# Construct grid for price changes.
PRICE_GRID = np.linspace(P1, 10, num=NUM_POINTS)

# Simulate agent populations of different types
agent_objs = dict()

for type_ in ['random', 'rational']:

    agent_objs[type_] = []

    for _ in range(NUM_AGENTS):

        agent_obj = AgentCls()

        agent_obj.set_type(type_)

        agent_obj.set_preference_parameter(ALPHA)

        agent_obj.set_endowment(ENDOWMENT)

        agent_obj.choose(P1, P1)

        agent_objs[type_] += [agent_obj]


# Get market demands for varying price schedules
market_demands = dict()

for type_ in ['random', 'rational']:

    market_demands[type_] = []

    # Initialze economy with agent of particular types
    economy_obj = EconomyCls(agent_objs[type_])

    # Vary price schedule
    for p2 in PRICE_GRID:

        # Construct market demand for second good
        demand = economy_obj.get_aggregate_demand(P1, P1)[1]

        print demand


        # Scaling to average demand
        demand = demand/float(NUM_AGENTS)

        # Collect demands
        market_demands[type_] += [demand]
