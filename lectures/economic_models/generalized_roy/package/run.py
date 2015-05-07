#!/usr/bin/env python
""" Driver to illustrate the grmpy package, which allows for the simulation
    and estimation of the generalized Roy model.

    For some basic explanations of the underlying model, please visit our
    organization's presence of GitHub at

        https://github.com/softEcon/course

    and visit the notebook for the Generalized Roy lecture.

"""

# standard library
import sys

# edit PYTHONPATH
sys.path.insert(0, 'grmpy')

# project library
import grmpy as gp

''' Calling the function of the library
'''

# Process initialization file
init_dict = gp.process()

# Simulate synthetic sample
gp.simulate(init_dict)

# Estimate generalize Roy model
#rslt = gp.estimate(init_dict)

# Inspect result
#gp.inspect(rslt, init_dict)

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def _fast_negative_log_likelihood(args, Y, D, X, Z):
    """ Negative Log-likelihood function of the Generalized Roy Model.
    """
    # Distribute parametrization
    Y1_coeffs = np.array(args['TREATED']['all'])
    Y0_coeffs = np.array(args['UNTREATED']['all'])

    C_coeffs = np.array(args['COST']['all'])

    U1_var = args['TREATED']['var']
    U0_var = args['UNTREATED']['var']

    U1V_rho = args['RHO']['treated']
    U0V_rho = args['RHO']['untreated']

    U1_sd = np.sqrt(U1_var)
    U0_sd = np.sqrt(U0_var)

    var_V = args['COST']['var']
    sdV  = np.sqrt(args['COST']['var'])

    # Auxiliary objects.
    num_agents = Y.shape[0]
    choice_coeffs = np.concatenate((Y1_coeffs - Y0_coeffs, - C_coeffs))

    # Likelihood construction.
    if True:

        G =  np.concatenate((X,Z), axis = 1)

        choiceIndices = np.dot(choice_coeffs, G.T)

        argOne = D*(Y - np.dot(Y1_coeffs, X.T))/U1_sd + \
                (1 - D)*(Y - np.dot(Y0_coeffs, X.T))/U0_sd

        argTwo = D*(choiceIndices - sdV*U1V_rho*argOne)/np.sqrt((1.0 - U1V_rho**2)*var_V) + \
                (1 - D)*(choiceIndices - sdV*U0V_rho*argOne)/np.sqrt((1.0 -
                                                                      U0V_rho**2)*var_V)

        cdfEvals = norm.cdf(argTwo)
        pdfEvals = norm.pdf(argOne)

        likl = D*(1.0/U1_sd)*pdfEvals*cdfEvals + \
                    (1 - D)*(1.0/U0_sd)*pdfEvals*(1.0  - cdfEvals)

        # Transformations.
        likl = np.clip(likl, 1e-20, np.inf)

        likl = -np.log(likl)

        likl = likl.sum()

        likl = (1.0/float(num_agents))*likl

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl

from tools.optimization.estimation import _load_data, _negative_log_likelihood


Y, D, X, Z = _load_data()

rslt_slow = _negative_log_likelihood(init_dict, Y, D, X, Z)

rslt_fast = _fast_negative_log_likelihood(init_dict, Y, D, X, Z)

print rslt_slow - rslt_fast