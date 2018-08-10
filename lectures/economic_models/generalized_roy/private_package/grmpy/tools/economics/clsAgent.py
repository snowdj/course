""" This module contains the class managing agents.
"""

# standard library
import numpy as np
from scipy.stats import norm

# project library
from tools.clsMeta import MetaCls

class AgentCls(MetaCls):
    """ Class instance that represents the agent.
    """
    def __init__(self):

        self.attr = {}

        # Exogenous attributes
        self.attr['exog'] = {}
        self.attr['exog']['outcome'] = None
        self.attr['exog']['cost'] = None
        self.attr['exog']['choice'] = None

        self.attr['endo'] = {}
        self.attr['endo']['choice'] = None
        self.attr['endo']['outcome'] = None

        # Economic Environment.
        self.attr['coeffs'] = {}
        self.attr['coeffs']['treated'] = None
        self.attr['coeffs']['untreated'] = None
        self.attr['coeffs']['cost'] = None
        self.attr['coeffs']['choice'] = None

        # Standard deviations
        self.attr['sds'] = {}
        self.attr['sds']['treated'] = None
        self.attr['sds']['untreated'] = None
        self.attr['sds']['cost'] = None

        # Correlations
        self.attr['rhos'] = {}
        self.attr['rhos']['treated'] = None
        self.attr['rhos']['untreated'] = None

        # Status indicator.
        self.is_locked = False

    def set_exogeneous_characteristics(self, which, attr):
        """ Method that allows to set the exogeneous agent characteristics.
        """
        # Antibugging
        assert (which in ['cost', 'outcome'])
        assert (self.get_status() is False)
        assert (isinstance(attr, list))

        # Set attribute.
        self.attr['exog'][which] = attr

    def set_endogenous_characteristics(self, which, attr):
        """ Method that allows to set the endogenous agent characteristics.
        """
        # Antibugging
        assert (which in ['choice', 'outcome'])
        assert (self.get_status() is False)
        assert (isinstance(attr, float) or isinstance(attr, int))

        # Set attribute.
        self.attr['endo'][which] = attr

    def set_economic_environment(self, init_dict):
        """ Method that allows to set the parametrization of the agent's
            economic environment.
        """
        # Antibugging
        assert (isinstance(init_dict, dict))
        assert (self.get_status() is False)

        # Outcome
        self.attr['coeffs']['treated'] = init_dict['TREATED']['all']
        self.attr['coeffs']['untreated'] = init_dict['UNTREATED']['all']

        # Cost
        self.attr['coeffs']['cost'] = init_dict['COST']['all']

        # Variances
        self.attr['sds']['treated'] = init_dict['TREATED']['sd']
        self.attr['sds']['untreated'] = init_dict['UNTREATED']['sd']
        self.attr['sds']['cost'] = init_dict['COST']['sd']

        # Correlations
        self.attr['rhos']['treated'] = init_dict['DIST']['rho1']
        self.attr['rhos']['untreated'] = init_dict['DIST']['rho0']

    ''' Private methods '''

    def _calculate_individual_likelihood(self):
        """ Method that calculates the individual likelihood.
        """
        # Distribute endogenous characteristics
        y = self.attr['endo']['outcome']
        d = self.attr['endo']['choice']

        # Distribute exogeneous characteristics
        x = self.attr['exog']['outcome']
        z = self.attr['exog']['cost']
        g = self.attr['exog']['choice']

        # Select relevant economic environment
        coeffs_choice = self.attr['coeffs']['choice']
        sd_v = self.attr['sds']['cost']

        if d == 1:
            coeffs_outcome = self.attr['coeffs']['treated']
            rho = self.attr['rhos']['treated']
            sd_u = self.attr['sds']['treated']
        else:
            coeffs_outcome = self.attr['coeffs']['untreated']
            rho = self.attr['rhos']['untreated']
            sd_u = self.attr['sds']['untreated']

        arg_one = (y - np.dot(coeffs_outcome, x)) / sd_u
        arg_two = (np.dot(coeffs_choice, g) - rho * sd_v * arg_one) / \
                  np.sqrt((1.0 - rho ** 2) * sd_v**2)

        pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

        if d == 1.0:
            contrib = (1.0 / float(sd_u)) * pdf_evals * cdf_evals
        else:
            contrib = (1.0 / float(sd_u)) * pdf_evals * (1.0 - cdf_evals)

        # Finishing
        return contrib

    def _derived_attributes(self):
        """ Calculate derived attributes.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Choice characteristics
        self.attr['exog']['choice'] = np.concatenate((self.attr['exog'][
                        'outcome'], self.attr['exog']['cost']))

        # Choice
        coeffs_choice = self.attr['coeffs']['treated'] - \
                        self.attr['coeffs']['untreated']

        coeffs_choice = np.concatenate((coeffs_choice, - self.attr['coeffs']['cost']))

        self.attr['coeffs']['choice'] = coeffs_choice

