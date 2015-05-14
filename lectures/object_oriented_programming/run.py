# Introduce Meta Clas later ...

# The rational agent.
import numpy as np

from scipy.optimize import minimize



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