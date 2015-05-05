""" This module contains all functions related to the simulation of
    a synthetic dataset from the generalized Roy model.
"""

# standard libraries
import numpy as np
import os

# project library
from tools.user.processing import process

''' Main function '''


def simulate(init_dict):
    """ Simulate a model based on the initialization file.
    """

    # Antibugging
    assert (isinstance(init_dict, dict))

    # Ensure recomputability
    np.random.seed(123)

    # Distribute information
    num_agents = init_dict['BASICS']['agents']
    file_name = init_dict['BASICS']['file']

    Y1_coeffs = init_dict['TREATED']['all']
    Y0_coeffs = init_dict['UNTREATED']['all']

    C_coeffs = np.array(init_dict['COST']['coeff'])

    U1_var = init_dict['TREATED']['var']
    U0_var = init_dict['UNTREATED']['var']

    V_var = init_dict['COST']['var']

    U1V_rho = init_dict['RHO']['treated']
    U0V_rho = init_dict['RHO']['untreated']

    # Auxiliary objects
    U1V_cov = U1V_rho * np.sqrt(U1_var) * np.sqrt(V_var)
    U0V_cov = U0V_rho * np.sqrt(U0_var) * np.sqrt(V_var)

    num_covars_out = Y1_coeffs.shape[0]
    num_covars_cost = C_coeffs.shape[0]

    # Simulate observables
    means = np.tile(0.0, num_covars_out)
    covs = np.identity(num_covars_out)

    X = np.random.multivariate_normal(means, covs, num_agents)
    X[:, 0] = 1.0

    means = np.tile(0.0, num_covars_cost)
    covs = np.identity(num_covars_cost)

    Z = np.random.multivariate_normal(means, covs, num_agents)

    # Construct index of observable characteristics
    Y1_level = np.dot(Y1_coeffs, X.T)
    Y0_level = np.dot(Y0_coeffs, X.T)
    C_level = np.dot(C_coeffs, Z.T)

    # Simulate unobservables
    means = np.tile(0.0, 3)
    vars_ = [U1_var, U0_var, V_var]
    covs = np.diag(vars_)

    covs[0, 2] = U1V_cov
    covs[2, 0] = covs[0, 2]

    covs[1, 2] = U0V_cov
    covs[2, 1] = covs[1, 2]

    U = np.random.multivariate_normal(means, covs, num_agents)

    # Simulate endogenous variables
    Y1 = np.tile(np.nan, num_agents)
    Y0 = np.tile(np.nan, num_agents)
    Y = np.tile(np.nan, num_agents)

    D = np.tile(np.nan, num_agents)

    for i in range(num_agents):
        # Distribute unobservables.
        U1, U0, V = U[i, 0], U[i, 1], U[i, 2]

        # Decision Rule.
        expectedBenefits = Y1_level[i] - Y0_level[i]
        cost = C_level[i] + V

        D[i] = np.float((expectedBenefits - cost > 0))

        # Potential outcomes.
        Y1[i] = Y1_level[i] + U1
        Y0[i] = Y0_level[i] + U0

        # Observed outcomes.
        Y[i] = D[i] * Y1[i] + (1.0 - D[i]) * Y0[i]

    # Check integrity
    _check_integrity_simulate(Y1, Y0, Y, D)

    # Save to disk
    _write_out(Y, D, X, Z, file_name)

    # Return selected features of data
    return Y1, Y0, D


''' Auxiliary functions '''


def _check_integrity_simulate(Y1, Y0, Y, D):
    """ Check quality of simulated sample.
    """
    assert (np.all(np.isfinite(Y1)))
    assert (np.all(np.isfinite(Y0)))

    assert (np.all(np.isfinite(Y)))
    assert (np.all(np.isfinite(D)))

    assert (Y1.dtype == 'float')
    assert (Y0.dtype == 'float')

    assert (Y.dtype == 'float')
    assert (D.dtype == 'float')

    assert (D.all() in [1.0, 0.0])


def _write_out(Y, D, X, Z, file_name):
    """ Write out simulated data to file.
    """

    np.savetxt(file_name, np.column_stack((Y, D, X, Z)), fmt='%8.3f')
