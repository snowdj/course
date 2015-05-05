""" This module contains all functions related to the processing of the
    initialization file.
"""

# standard library
import shlex

import numpy as np

''' Main function '''


def process():
    """ This function reads the init.ini file.
    """
    # Initialization
    dict_ = {}

    for line in open('init.ini').readlines():

        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_keyword = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        if is_keyword:
            keyword = list_[0]
            dict_[keyword] = {}
            continue

        # Distribute information
        name, val = list_[0], list_[1]

        # Prepare container.
        if name not in dict_[keyword].keys():
            dict_[keyword][name] = None

            if name in ['coeff']:
                dict_[keyword][name] = []

        # Type conversion
        if name in ['agents', 'maxiter']:
            val = int(val)
        elif name in ['file', 'optimizer', 'start', 'version']:
            val = str(val)
        else:
            val = float(val)

        # Collect information
        if name in ['coeff']:
            dict_[keyword][name] += [val]
        else:
            dict_[keyword][name] = val

    # Add auxiliary objects
    dict_ = _add_auxiliary(dict_)

    # Check quality.
    _check_integrity(dict_)

    # Finishing.
    return dict_


''' Auxiliary functions '''


def _check_integrity(dict_):
    """ Check integrity of initFile dict.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Check number of agents
    assert (dict_['BASICS']['agents'] > 0)
    assert (isinstance(dict_['BASICS']['agents'], int))

    # Check optimizer
    assert (dict_['ESTIMATION']['optimizer'] in ['bfgs', 'nm'])

    # Check starting values
    assert (dict_['ESTIMATION']['start'] in ['random', 'init', 'zero'])

    # Maximum iterations
    assert (dict_['ESTIMATION']['maxiter'] >= 0)

    # Implementations
    assert (dict_['ESTIMATION']['version'] in ['functional', 'object', 'optimized'])
    assert (dict_['ESTIMATION']['version'] in ['functional'])

    # Finishing
    return True


def _add_auxiliary(dict_):
    """ Add some auxiliary objects.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Initialize container
    dict_['AUX'] = {}

    # Full set of coefficients.
    dict_['TREATED']['all'] = [dict_['TREATED']['int']]
    dict_['TREATED']['all'] += dict_['TREATED']['coeff']
    dict_['TREATED']['all'] = np.array(dict_['TREATED']['all'])

    dict_['UNTREATED']['all'] = [dict_['UNTREATED']['int']]
    dict_['UNTREATED']['all'] += dict_['UNTREATED']['coeff']
    dict_['UNTREATED']['all'] = np.array(dict_['UNTREATED']['all'])

    dict_['COST']['all'] = np.array(dict_['COST']['coeff'])

    # Number of covariates
    num_covars_out = len(dict_['TREATED']['coeff']) + 1
    num_covars_cost = len(dict_['COST']['coeff'])

    dict_['AUX']['num_covars_out'] = num_covars_out
    dict_['AUX']['num_covars_cost'] = num_covars_cost

    # Number of parameters
    dict_['AUX']['num_paras'] = 2 * num_covars_out + num_covars_cost + 2 + 2

    # Starting values
    dict_['AUX']['start_values'] = []
    dict_['AUX']['start_values'] += [dict_['TREATED']['int']]
    dict_['AUX']['start_values'] += dict_['TREATED']['coeff']
    dict_['AUX']['start_values'] += [dict_['UNTREATED']['int']]
    dict_['AUX']['start_values'] += dict_['UNTREATED']['coeff']
    dict_['AUX']['start_values'] += dict_['COST']['coeff']
    dict_['AUX']['start_values'] += [dict_['TREATED']['var']]
    dict_['AUX']['start_values'] += [dict_['UNTREATED']['var']]
    dict_['AUX']['start_values'] += [dict_['RHO']['treated']]
    dict_['AUX']['start_values'] += [dict_['RHO']['untreated']]

    # Finishing
    return dict_


def _process_cases(list_):
    """ Process cases and determine whether keyword or empty
        line.
    """
    # Antibugging
    assert (isinstance(list_, list))

    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_keyword = list_[0].isupper()
    else:
        is_keyword = False

    # Antibugging
    assert (is_keyword in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_keyword