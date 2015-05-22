""" This module contains all functions related to the estimation of the
    generalized Roy model.
"""

# standard library
import sys
import numpy as np
import statsmodels.api  as sm
from scipy.stats import norm
from scipy.optimize import minimize

# project library
from tools.economics.clsAgent import AgentCls

''' Main function '''


def estimate(init_dict):
    """ Estimate our version of the generalized Roy model.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))

    # Load dataset
    Y, D, X, Z, agent_objs = _load_data(init_dict)

    # Create auxiliary objects
    start = init_dict['ESTIMATION']['start']
    maxiter = init_dict['ESTIMATION']['maxiter']

    optimizer = init_dict['ESTIMATION']['algorithm']
    version = init_dict['ESTIMATION']['version']
    num_covars_out = init_dict['AUX']['num_covars_out']

    # Initialize different starting values
    x0 = _get_start(start, init_dict, Y, D, X, Z)

    # Select optimizer
    if optimizer == 'nm':

        optimizer = 'Nelder-Mead'

    elif optimizer == 'bfgs':

        optimizer = 'BFGS'

    # Provide additional arguments to the optimizer
    opts = dict()

    opts['maxiter'] = maxiter

    # Run optimization or just evaluate function at starting values
    if maxiter == 0:

        # Collect maximization arguments.
        rslt = _distribute_parameters(np.array(x0), init_dict, num_covars_out)

        # Calculate likelihood according to user's request
        likl = _negative_log_likelihood(rslt, Y, D, X, Z, agent_objs, version)

        # Compile results
        x_rslt, fun, success = x0, likl, False

    else:

        # Check out the SciPy documentation for details about the interface
        # to the `minimize' function that provides a convenient interface to
        # a variety of alternative maximization algorithms. You will also
        # find information about the return information.
        opt_rslt = minimize(_max_interface, x0,
                            args=(Y, D, X, Z, agent_objs, version, init_dict),
                            method=optimizer, options=opts)

        # Compile results
        x_rslt, fun = opt_rslt['x'], opt_rslt['fun']
        success = opt_rslt['success']

    # Tranformation to internal parameters
    rslt = _distribute_parameters(x_rslt, init_dict, num_covars_out)

    rslt['fval'], rslt['success'] = fun, success

    # Finishing
    return rslt


''' Auxiliary functions '''
# Note that the name of all auxiliary functions starts with an underscore.
# This ensures that the function is private to the module. A standard import
# of this module will not make this function available.


def _distribute_parameters(x, init_dict, num_covars_out):
    """ Distribute the parameters.
    """
    # Antibugging
    assert (isinstance(x, np.ndarray))
    assert (isinstance(num_covars_out, int))
    assert (num_covars_out > 0)

    # Initialize containers
    rslt = dict()

    rslt['TREATED'] = dict()
    rslt['UNTREATED'] = dict()
    rslt['COST'] = dict()
    rslt['DIST'] = dict()

    # Distribute parameters
    rslt['TREATED']['all'] = x[:num_covars_out]
    rslt['UNTREATED']['all'] = x[num_covars_out:(2 * num_covars_out)]

    rslt['COST']['all'] = x[(2 * num_covars_out):(-4)]
    rslt['COST']['sd'] = init_dict['COST']['sd']

    rslt['TREATED']['sd'] = np.exp(x[(-4)])
    rslt['UNTREATED']['sd'] = np.exp(x[(-3)])

    rslt['DIST']['rho1'] = -1.0 + 2.0 / (1.0 + float(np.exp(-x[-2])))
    rslt['DIST']['rho0'] = -1.0 + 2.0 / (1.0 + float(np.exp(-x[-1])))

    # Update auxiliary versions
    rslt['AUX'] = dict()

    rslt['AUX']['x_internal'] = x.copy()
    rslt['AUX']['x_internal'][-4] = np.exp(x[(-4)])
    rslt['AUX']['x_internal'][-3] = np.exp(x[(-3)])
    rslt['AUX']['x_internal'][-2] = -1.0 + 2.0 / (1.0 + float(np.exp(-x[-2])))
    rslt['AUX']['x_internal'][-1] = -1.0 + 2.0 / (1.0 + float(np.exp(-x[-1])))

    rslt['AUX']['init_values'] = init_dict['AUX']['init_values']

    # Finishing.
    return rslt


def _max_interface(x, Y, D, X, Z, agent_objs, version, init_dict):
    """ Interface to the SciPy maximization routines.
    """
    # Auxiliary objects
    try:
        num_covars_out = X.shape[1]
    except AttributeError:
        num_covars_out = len(agent_objs[0].attr['exog']['outcome'])

    # Collect maximization arguments
    rslt = _distribute_parameters(x, init_dict, num_covars_out)

    # Calculate likelihood
    likl = _negative_log_likelihood(rslt, Y, D, X, Z, agent_objs, version)

    # Finishing.
    return likl


def _negative_log_likelihood(args, Y, D, X, Z, agent_objs, version):
    """ Negative log-likelihood evaluation.
    """

    # Select version
    if version == 'slow':
        likl = _slow_negative_log_likelihood(args, Y, D, X, Z)
    elif version == 'fast':
        likl = _fast_negative_log_likelihood(args, Y, D, X, Z)
    elif version == 'object':
        likl = _object_negative_log_likelihood(args, agent_objs)
    else:
        raise AssertionError

    # Finishing
    return likl


def _object_negative_log_likelihood(args, agent_objs):
    """ Negative Log-likelihood function of the generalized Roy model.
    """

    num_agents = len(agent_objs)

    likl = np.tile(np.nan, num_agents)

    for i, agent_obj in enumerate(agent_objs):

        agent_obj.unlock()

        agent_obj.set_economic_environment(args)

        agent_obj.lock()

        likl[i] = agent_obj._calculate_individual_likelihood()

    # Transformations.
    likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl


def _slow_negative_log_likelihood(args, Y, D, X, Z):
    """ Negative Log-likelihood function of the generalized Roy model.
    """
    # Distribute arguments
    Y1_coeffs, Y0_coeffs, C_coeffs, choice_coeffs, U1_sd, U0_sd, U1V_rho, \
    U0V_rho, V_sd = _distribute_arguments(args)

    # Auxiliary objects
    num_agents = Y.shape[0]

    # Initialize containers
    likl = np.tile(np.nan, num_agents)
    choice_idx = np.tile(np.nan, num_agents)

    # Likelihood construction.
    for i in range(num_agents):

        g = np.concatenate((X[i, :], Z[i,:]))
        choice_idx[i] = np.dot(choice_coeffs, g)

        # Select outcome information
        if D[i] == 1.00:
            coeffs, rho, sd = Y1_coeffs, U1V_rho, U1_sd
        else:
            coeffs, rho, sd = Y0_coeffs, U0V_rho, U0_sd

        arg_one = (Y[i] - np.dot(coeffs, X[i, :])) / sd
        arg_two = (choice_idx[i] - rho * V_sd * arg_one) / \
                  np.sqrt((1.0 - rho ** 2) * V_sd**2)

        pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

        if D[i] == 1.0:
            contrib = (1.0 / float(sd)) * pdf_evals * cdf_evals
        else:
            contrib = (1.0 / float(sd)) * pdf_evals * (1.0 - cdf_evals)

        likl[i] = contrib

    # Transformations.
    likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl


def _fast_negative_log_likelihood(args, Y, D, X, Z):
    """ Negative Log-likelihood function of the Generalized Roy Model.
    """
    # Distribute arguments
    Y1_coeffs, Y0_coeffs, C_coeffs, choice_coeffs, U1_sd, U0_sd, U1V_rho, \
    U0V_rho, V_sd = _distribute_arguments(args)

    # Likelihood construction.
    G = np.concatenate((X, Z), axis=1)
    choice_idx = np.dot(choice_coeffs, G.T)

    arg_one = D * (Y - np.dot(Y1_coeffs, X.T)) / U1_sd + \
             (1 - D) * (Y - np.dot(Y0_coeffs, X.T)) / U0_sd

    arg_two = D * (choice_idx - V_sd * U1V_rho * arg_one) / np.sqrt(
        (1.0 - U1V_rho ** 2) * V_sd**2) + \
        (1 - D) * (choice_idx - V_sd * U0V_rho * arg_one) / np.sqrt(
                 (1.0 - U0V_rho ** 2) * V_sd**2)

    pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

    likl = D * (1.0 / U1_sd) * pdf_evals * cdf_evals + \
           (1 - D) * (1.0 / U0_sd) * pdf_evals * (1.0 - cdf_evals)

    # Transformations.
    likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl


def _load_data(init_dict):
    """ Load dataset.
    """
    # Auxiliary objects
    num_covars_out = init_dict['AUX']['num_covars_out']
    num_covars_cost = init_dict['AUX']['num_covars_cost']
    num_agents = init_dict['BASICS']['agents']

    is_object = (init_dict['ESTIMATION']['version'] == 'object')

    # Read dataset
    data = np.genfromtxt(init_dict['BASICS']['source'])

    # Reshaping, this ensure that the program also runs with just one agent
    # as otherwise only an vector is created. This creates problems for the
    # subsetting of the overall data into the components.
    data = np.array(data, ndmin=2)

    # Distribute data
    Y, D = data[:, 0], data[:, 1]

    X, Z = data[:, 2:(num_covars_out + 2)], data[:, -num_covars_cost:]

    # Create object-oriented version of sample
    agent_objs = None

    if is_object:

        agent_objs = []

        for i in range(num_agents):

            agent_obj = AgentCls()

            agent_obj.set_exogeneous_characteristics('cost', list(Z[i,:]))

            agent_obj.set_exogeneous_characteristics('outcome', list(X[i,:]))

            agent_obj.set_endogenous_characteristics('choice', int(D[i]))

            agent_obj.set_endogenous_characteristics('outcome', Y[i])

            agent_obj.set_economic_environment(init_dict)

            agent_obj.lock()

            agent_objs += [agent_obj]

    # Finishing
    return Y, D, X, Z, agent_objs


def _get_start(which, init_dict, Y, D, X, Z):
    """ Get different kind of starting values.
    """
    # Antibugging.
    assert (which in ['random', 'init', 'auto'])

    # Distribute auxiliary objects
    num_paras = init_dict['AUX']['num_paras']
    num_covars_cost = init_dict['AUX']['num_covars_cost']

    # Construct auxiliary objects
    G = np.concatenate((X, Z[:, 1:]), axis=1)

    # Select relevant values.
    if which == 'random':
        x0 = np.random.uniform(size=num_paras)

        # Variances
        x0[(-4)] = max(x0[(-4)], 0.01)
        x0[(-3)] = max(x0[(-3)], 0.01)

        # Correlations
        x0[(-2)] -= 0.5
        x0[(-1)] -= 0.5

    elif which == 'init':
        x0 = np.array(init_dict['AUX']['init_values'][:])

    elif which == 'auto':

        # Subsetting
        Y1, X1 = Y[D == 1], X[(D == 1), :]
        olsRslt = sm.OLS(Y1, X1).fit()

        # Extract results
        coeffs_treated = olsRslt.params
        sd_treated = np.array(np.sqrt(olsRslt.scale))

        # Subsetting
        Y0, X0 = Y[D == 0], X[(D == 0), :]
        olsRslt = sm.OLS(Y0, X0).fit()

        # Extract results
        coeffs_untreated = olsRslt.params
        sd_untreated = np.array(np.sqrt(olsRslt.scale))

        # Estimate choice model
        probitRslt = sm.Probit(D, G).fit()
        sd = init_dict['COST']['sd']
        coeffs = probitRslt.params*sd

        # Special treatment of cost intercept
        cost_int = coeffs_treated[0] - coeffs_untreated[0] - coeffs[0]

        # Collect results
        x0 = np.concatenate((coeffs_treated, coeffs_untreated))
        x0 = np.concatenate((x0, [cost_int], -coeffs[-(num_covars_cost - 1):]))
        x0 = np.concatenate((x0, [sd_treated, sd_untreated]))
        x0 = np.concatenate((x0, [0.00, 0.00]))

    else:
        raise AssertionError

    # Document starting values
    init_dict['AUX']['start_values'] = x0.copy()

    # Transform to real line
    x0 = _transform_start(x0)

    # Type conversion
    x0 = np.array(x0)

    # Quality assurance.
    assert (np.all(np.isfinite(x0)))

    # Finishing.
    return x0


def _transform_start(x):
    """ Transform starting values to cover the whole real line.
    """

    # Coefficients
    x[:(-4)] = x[:(-4)]

    # Variances
    x[(-4)] = np.log(x[(-4)])
    x[(-3)] = np.log(x[(-3)])

    # Correlations
    transform = (x[(-2)] + 1) / 2
    x[(-2)] = np.log(transform / (1.0 - transform))

    transform = (x[(-1)] + 1) / 2
    x[(-1)] = np.log(transform / (1.0 - transform))

    # Finishing
    return x


def _distribute_arguments(args):
    """ Distribute arguments for evaluation of criterion function and some
        auxiliary parameters.
    """
    Y1_coeffs = np.array(args['TREATED']['all'])
    Y0_coeffs = np.array(args['UNTREATED']['all'])

    C_coeffs = np.array(args['COST']['all'])

    U1_sd = args['TREATED']['sd']
    U0_sd = args['UNTREATED']['sd']

    U1V_rho = args['DIST']['rho1']
    U0V_rho = args['DIST']['rho0']
    V_sd = args['COST']['sd']

    choice_coeffs = np.concatenate((Y1_coeffs - Y0_coeffs, - C_coeffs))

    # Finishing
    return Y1_coeffs, Y0_coeffs, C_coeffs, choice_coeffs, U1_sd, U0_sd, \
           U1V_rho, U0V_rho, V_sd