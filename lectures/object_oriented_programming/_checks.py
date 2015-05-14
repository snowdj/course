""" This module contains all the checks related to economy and agent class.
"""

# standard library
import numpy as np


def integrity_checks(str_, *args):
    """ Check integrity of interface and computed results for the methods of
        the agent and economy class.
    """

    ''' AgentCls'''
    if str_ == 'set_type':

        type_, = args

        assert (type_ in ['random', 'rational'])

    elif str_ == 'set_endowment':

        y, = args

        assert (isinstance(y, float))
        assert (y >= 0)

    elif str_ == 'set_preference_parameter':

        alpha, = args

        assert (isinstance(alpha, float))
        assert (0.0 < alpha < 1.0)

    elif str_ == 'choose':

        p1, p2 = args

        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == 'get_individual_demand':

        rslt, = args

        assert (isinstance(rslt, list))
        assert (np.all(rslt > 0))

    elif str_ == 'spending':

        x, p1, p2 = args

        assert (np.all(x > 0))
        assert (isinstance(p1, float) and isinstance(p2, float))
        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == '_choose_random_in':

        y, p1, p2 = args

        assert (isinstance(y, float))
        assert (y > 0)
        assert (isinstance(p1, float) and isinstance(p2, float))
        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == '_choose_random_out':

        x, = args

        assert isinstance(x, list)
        assert (np.all(x > 0))

    elif str_ == '_choose_rational_in':

        y, p1, p2 = args

        assert (isinstance(y, float))
        assert (y > 0)
        assert (isinstance(p1, float) and isinstance(p2, float))
        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == '_choose_rational_out':

        x, = args

        assert isinstance(x, list)
        assert (np.all(x > 0))

    elif str_ == '_criterion':

        x, = args

        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.ndim == 1)

    elif str_ == '_constraint':

        x, p1, p2 = args

        assert (np.all(np.isfinite(x)))
        assert (isinstance(p1, float) and isinstance(p2, float))
        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == '__init__':

        agent_objs, = args

        assert (isinstance(agent_objs, list))

    elif str_ == 'get_aggregate_demand_in':

        p1, p2 = args

        assert (np.all([p1, p2] > 0))
        assert (np.all(np.isfinite([p1, p2])))
        assert (isinstance(p1, float) and isinstance(p2, float))

    elif str_ == 'get_aggregate_demand_out':

        rslt, = args

        assert (isinstance(rslt, list))
        assert (np.all(rslt > 0))

    else:

        raise AssertionError
