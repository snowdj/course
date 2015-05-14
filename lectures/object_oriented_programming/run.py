# Introduce Meta Clas later ...

# The rational agent.
import numpy as np

from clsEconomy import *
from clsAgent import *


if True:

    NUM_AGENTS = 1

    ENDOWMENT = 10.0

    ALPHA = 0.5

    P1 = 0.1

    # Auxiliary
    price_grid = np.linspace(P1, 1.0, num=25)


    # Simulate agent populations of different types
    agent_objs = dict()

    agent_obj = AgentCls()

    agent_obj.set_type('rational')

    agent_obj.set_preference_parameter(ALPHA)

    agent_obj.set_endowment(ENDOWMENT)

    agent_obj.choose(P1, P1)

    agent_obj.get_individual_demand()

    agent_objs = [agent_obj]

    EconomyCls(agent_objs)

