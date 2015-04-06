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
class DiscreteRV(object):

    def __init__(self, q):
        self._q = q
        self.Q = np.cumsum(q)

    @property
    def q(self):

        return self._q

    @q.setter
    def q(self, val):

        self._q = val
        self.Q = np.cumsum(val)

    def draw(self, k=1):

        return self.Q.searchsorted(np.random.uniform(0, 1, size=k))

class BetaBinomial(object):

    def __init__(self, n, a, b):

        self.n, self.a, self.b = n, a, b

    @property
    def mean(self):

        n, a, b = self.n, self.a, self.b
        return n * a / (a + b)

    @property
    def std(self):

        return sqrt(self.var)

    @property
    def var(self):

        n, a, b = self.n, self.a, self.b
        top = n*a*b * (a + b + n)
        btm = (a+b)**2.0 * (a+b+1.0)
        return top / btm

    @property
    def skew(self):

        n, a, b = self.n, self.a, self.b
        t1 = (a+b+2*n) * (b - a) / (a+b+2)
        t2 = sqrt((1+a+b) / (n*a*b * (n+a+b)))
        return t1 * t2

    def pdf(self):

        n, a, b = self.n, self.a, self.b
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

