""" This module contains all functions related to the processing of the
    initialization file.
"""

# standard library
import shlex
import codecs

import numpy as np

''' Main function '''


def process(file_):
    """ Process initialization file.
    """
    # Initialization
    dict_ = {}

    for line in open(file_).readlines():

        # Remove UTF-3 marker
        if line.startswith(codecs.BOM_UTF8):
            line = line[3:]

        # Split line
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

        if keyword not in ['BENE']:
            dict_ = _process_not_bene(list_, dict_, keyword)

        else:
            dict_ = _process_bene(list_, dict_, keyword)

    # Remove BENE
    del dict_['BENE']

    # Add auxiliary objects
    dict_ = _add_auxiliary(dict_)

    # Check quality.
    _check_integrity_process(dict_)

    # Finishing.
    return dict_



''' Auxiliary functions '''
# Note that the name of all auxiliary functions starts with an underscore.
# This ensures that the function is private to the module. A standard import
# of this module will not make this function available.


def _process_bene(list_, dict_, keyword):
    """ This function processes the BENE part of the initialization file.
    """
    # Distribute information
    name, val_treated, val_untreated = list_[0], list_[1], list_[2]

    # Initialize dictionary
    if 'TREATED' not in dict_.keys():
        for subgroup in ['TREATED', 'UNTREATED']:
            dict_[subgroup] = {}
            dict_[subgroup]['coeff'] = []
            dict_[subgroup]['int'] = None
            dict_[subgroup]['sd'] = None

    # Type conversion
    val_treated = float(val_treated)
    val_untreated = float(val_untreated)

    # Collect information
    if name in ['coeff']:
        dict_['TREATED'][name] += [val_treated]
        dict_['UNTREATED'][name] += [val_untreated]
    else:
        dict_['TREATED'][name] = val_treated
        dict_['UNTREATED'][name] = val_untreated

    # Finishing
    return dict_


def _process_not_bene(list_, dict_, keyword):
    """ This function processes all of the initialization file, but the
        BENE section.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # Prepare container.
    if name not in dict_[keyword].keys():
        if name in ['coeff']:
            dict_[keyword][name] = []

    # Type conversion
    if name in ['agents', 'maxiter']:
        val = int(val)
    elif name in ['source', 'algorithm', 'start', 'version']:
        val = str(val)
    else:
        val = float(val)

    # Collect information
    if name in ['coeff']:
        dict_[keyword][name] += [val]
    else:
        dict_[keyword][name] = val

    # Finishing.
    return dict_


def _check_integrity_process(dict_):
    """ Check integrity of initFile dict.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Check number of agents
    assert (dict_['BASICS']['agents'] > 0)
    assert (isinstance(dict_['BASICS']['agents'], int))

    # Check optimizer
    assert (dict_['ESTIMATION']['algorithm'] in ['bfgs', 'nm'])

    # Check starting values
    assert (dict_['ESTIMATION']['start'] in ['random', 'init'])

    # Maximum iterations
    assert (dict_['ESTIMATION']['maxiter'] >= 0)

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
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        dict_[key_]['all'] = [dict_[key_]['int']]
        dict_[key_]['all'] += dict_[key_]['coeff']
        dict_[key_]['all'] = np.array(dict_[key_]['all'])

    # Number of covariates
    num_covars_out = len(dict_['TREATED']['all'])
    num_covars_cost = len(dict_['COST']['all'])

    dict_['AUX']['num_covars_out'] = num_covars_out
    dict_['AUX']['num_covars_cost'] = num_covars_cost

    # Number of parameters
    dict_['AUX']['num_paras'] = 2 * num_covars_out + num_covars_cost + 2 + 2

    # Starting values
    dict_['AUX']['init_values'] = []

    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        dict_['AUX']['init_values'] += dict_[key_]['all'].tolist()

    dict_['AUX']['init_values'] += [dict_['TREATED']['sd']]
    dict_['AUX']['init_values'] += [dict_['UNTREATED']['sd']]
    dict_['AUX']['init_values'] += [dict_['DIST']['rho1']]
    dict_['AUX']['init_values'] += [dict_['DIST']['rho0']]

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
