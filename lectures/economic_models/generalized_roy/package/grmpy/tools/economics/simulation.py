""" This module contains all functions related to the simulation of
    a synthetic dataset from the generalized Roy model.
"""

# standard library
import numpy as np
import os


''' Main function '''


def simulate(init_dict, unobserved=False):
    """ Simulate a model based on the initialization file.
    """

    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (unobserved in [True, False])

    # Ensure recomputability
    np.random.seed(123)

    # Distribute information
    num_agents = init_dict['BASICS']['agents']
    source = init_dict['BASICS']['source']

    Y1_coeffs = init_dict['TREATED']['all']
    Y0_coeffs = init_dict['UNTREATED']['all']

    C_coeffs = np.array(init_dict['COST']['all'])

    U1_sd = init_dict['TREATED']['sd']
    U0_sd = init_dict['UNTREATED']['sd']

    V_sd = init_dict['COST']['sd']

    U1V_rho = init_dict['DIST']['rho1']
    U0V_rho = init_dict['DIST']['rho0']

    # Auxiliary objects
    U1V_cov = U1V_rho * U1_sd * V_sd
    U0V_cov = U0V_rho * U0_sd * V_sd

    num_covars_out = Y1_coeffs.shape[0]
    num_covars_cost = C_coeffs.shape[0]

    # Simulate observables
    means = np.tile(0.0, num_covars_out)
    covs = np.identity(num_covars_out)

    X = np.random.multivariate_normal(means, covs, num_agents)

    means = np.tile(0.0, num_covars_cost)
    covs = np.identity(num_covars_cost)

    Z = np.random.multivariate_normal(means, covs, num_agents)

    # Add intercepts. The first column of the X and Z matrix always contains
    # the intercept term. This is exploited throughout the code.
    Z[:,0], X[:, 0] = 1.0, 1.0

    # Construct index of observable characteristics
    Y1_level = np.dot(Y1_coeffs, X.T)
    Y0_level = np.dot(Y0_coeffs, X.T)
    C_level = np.dot(C_coeffs, Z.T)

    # Simulate unobservables
    means = np.tile(0.0, 3)
    vars_ = [U1_sd**2, U0_sd**2, V_sd**2]
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

        # Select individual unobservables and observables
        u1, u0, v = U[i, 0], U[i, 1], U[i, 2]

        y1_idx, y0_idx, c_idx = Y1_level[i], Y0_level[i], C_level[i]

        # Decision Rule
        expected_benefits = y1_idx - y0_idx
        cost = c_idx + v

        d = np.float((expected_benefits - cost > 0))

        # Potential outcomes
        y1, y0 = y1_idx + u1, y0_idx + u0

        # Observed outcomes
        y = d * y1 + (1.0 - d) * y0

        # Collect data matrices
        Y[i], Y0[i], Y1[i], D[i] = y, y1, y0, d

    # Check integrity of simulated data
    _check_integrity_simulate(Y1, Y0, Y, D)

    # Save to disk
    _write_out(Y, D, X, Z, source, unobserved, Y1, Y0)

    # Return selected features of data
    return Y1, Y0, D


''' Auxiliary functions '''
# Note that the name of all auxiliary functions starts with an underscore.
# This ensures that the function is private to the module. A standard import
# of this module will not make this function available.

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


def _write_out(Y, D, X, Z, source, unobserved=False, Y1=None, Y0=None):
    """ Write out simulated data to file.
    """

    if not unobserved:

        np.savetxt(source, np.column_stack((Y, D, X, Z)), fmt='%8.3f')

    else:

        assert (isinstance(Y1, np.ndarray))
        assert (isinstance(Y0, np.ndarray))

        np.savetxt(source, np.column_stack((Y, D, X, Z, Y1, Y0)),
                   fmt='%8.3f')