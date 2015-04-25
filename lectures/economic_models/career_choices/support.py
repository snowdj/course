""" This file contains some supporting functions drawn from the QuantEcon
    Library.
"""

# libraries
from scipy.special import binom, beta
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

''' Random Variables
'''
def BetaBinomial_pdf(n, a, b):
    ''' BetaBinomial.
    '''
    k = np.arange(n + 1)

    probs = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)

    return probs

''' Graphs
'''
def plot_optimal_policy(optimal_policy, attr):

    fig, ax = plt.subplots(figsize=(6,6))
    tg, eg = np.meshgrid(attr['theta'], attr['epsilon'])
    lvls=(0.5, 1.5, 2.5, 3.5)
    ax.contourf(tg, eg, optimal_policy.T, levels=lvls, cmap=cm.winter, alpha=0.5)
    ax.contour(tg, eg, optimal_policy.T, colors='k', levels=lvls, linewidths=2)
    ax.set_xlabel(r"${\theta}$", fontsize=25)
    ax.set_ylabel('$\epsilon$', fontsize=25)
    ax.text(1.8, 2.5, 'new life', fontsize=14)
    ax.text(4.5, 2.5, 'new job', fontsize=14, rotation='vertical')
    ax.text(4.0, 4.5, 'stay put', fontsize=14)

