#!/usr/bin/env python
""" This module conducts the Monte Carlo exercise discussed in the
    lecture.
"""

# standard library
import os
import argparse
import numpy as np


def conduct_monte_carlo(scale):
    """ This function conducts a Monte Carlo exercise to test the
        reliabilty of the grmToolbox.
    """

    # Clean directory
    os.system('grmToolbox-clean')

    # Estimate generalized Roy model on DATA/source
    os.system('grmToolbox-estimate')

    # Simulate dataset with perturbed parameter values
    # and store it as SIMULATION/target
    os.system('grmToolbox-simulate --init init.ini --update')

    # Perturbing parameter values
    cmd = 'grmToolbox-perturb --init init.ini --scale 0.1 --seed 1234 --update'
    os.system(cmd)

    # Re-estimate generalize Roy model using SIMULATION/target as
    # DATA/source
    os.system('grmToolbox-estimate --init init.ini --simulation')

def process(args):
    """ Process arguments.
    """
    # Distribute arguments.
    scale = args.scale

    # Quality checks.
    assert (isinstance(scale, float))

    # Finishing.
    return scale

def print_results():
    # Load true and estimated parameters
    true_values = np.loadtxt('simulation.paras.grm.out')
    est_values = np.loadtxt('stepParas.grm.out')
    start_values = np.loadtxt('startParas.grm.out')

    # Auxiliary objects
    num_paras = len(true_values)

    # Formatting
    fmt = '{0:10.2f}{1:10.2f}{2:10.2f}{3:10.2f}'

    # Print both parameters
    print '     Start  Estimate     Truth    Difference \n'
    for i in range(num_paras):

        start, est, true = start_values[i], est_values[i], true_values[i]

        diff = est - true

        print fmt.format(start, est, true, diff)

''' Execution of module as script.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'Conduct Monte Carlo exercise with grmToolbox.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scale', type=float, default=0.1,
                        dest='scale', help='magnitude of perturbation')

    args = parser.parse_args()

    scale = process(args)

    conduct_monte_carlo(scale)

    print_results()