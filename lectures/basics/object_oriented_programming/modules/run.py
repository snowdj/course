# Fundamental Numerical Methods
import numpy as np

# System-specific parameters and functions
import sys

# Adding the modules subdirectory
#sys.path.insert(0, 'modules')

# Project library
from clsAgent import *
from clsEconomy import *

class Agent():
    def __init__(self, endowment):
        """ Initialize agents with endowment as class attributes.
        """
        # Endowment
        self.endowment = endowment

        # Demands
        self.butter = None
        self.milk = None

    def choose(self, price_butter, price_milk):
        """ Allocate half of endowment to each
            of the two goods.
        """
        self.butter = self.endowment/price_butter
        self.milk = self.endowment/price_milk

    def get_demands(self):
        """ Return demands.
        """
        return self.butter, self.milk

    def __str__(self):
        """ String representation of class instance
        """
        # Distribute class attributes
        endowment = self.endowment

        return 'I am an agent with an endwoment of ' + str(endowment) + '.'

    def __eq__(self, other):
        """ Check for equality.
        """
        # Antibugging
        assert isinstance(other, Agent)

        # Distribute class attributes
        self_end = self.endowment
        other_end = other.endowment

        # Check equality
        return self_end == other_end




# Initialize and agent with an endowment
ENDOWMENT, PRICE_BUTTER, PRICE_MILK = 10, 2, 3

agent_obj = Agent(11.234)

agent_obj.choose(PRICE_BUTTER, PRICE_MILK)

print ' Let us have a look at the demand for the two goods: '
print '... using the demand() method      ', agent_obj.get_demands()
print '... accessing the class attributes ', (agent_obj.butter, agent_obj.milk)

agent_obj2 = Agent(11.2224)

print agent_obj == agent_obj2