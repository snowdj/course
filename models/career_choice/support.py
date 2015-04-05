''' This file contains some supporting functions drawn from the QuantEcon
    Library.
'''

"""
Filename: discrete_rv.py

Authors: Thomas Sargent, John Stachurski

Generates an array of draws from a discrete random variable with a
specified vector of probabilities.

"""

from numpy import cumsum
from numpy.random import uniform


class DiscreteRV(object):
    """
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.

    Parameters
    ----------
    q : array_like(float)
        Nonnegative numbers that sum to 1

    Attributes
    ----------
    q : see Parameters
    Q : array_like(float)
        The cumulative sum of q

    """

    def __init__(self, q):
        self._q = q
        self.Q = cumsum(q)

    def __repr__(self):
        return "DiscreteRV with {n} elements".format(n=self._q.size)

    def __str__(self):
        return self.__repr__()

    @property
    def q(self):
        """
        Getter method for q.

        """
        return self._q

    @q.setter
    def q(self, val):
        """
        Setter method for q.

        """
        self._q = val
        self.Q = cumsum(val)

    def draw(self, k=1):
        """
        Returns k draws from q.

        For each such draw, the value i is returned with probability
        q[i].

        Parameters
        -----------
        k : scalar(int), optional
            Number of draws to be returned

        Returns
        -------
        array_like(int)
            An array of k independent draws from q

        """
        return self.Q.searchsorted(uniform(0, 1, size=k))

"""
Filename: compute_fp.py
Authors: Thomas Sargent, John Stachurski

Compute the fixed point of a given operator T, starting from
specified initial condition v.

"""
import time
import numpy as np


def _print_after_skip(skip, it=None, dist=None, etime=None):
    if it is None:
        # print initial header
        msg = "{i:<13}{d:<15}{t:<17}".format(i="Iteration",
                                             d="Distance",
                                             t="Elapsed (seconds)")
        print(msg)
        print("-" * len(msg))

        return

    if it % skip == 0:
        if etime is None:
            print("After {it} iterations dist is {d}".format(it=it, d=dist))

        else:
            # leave 4 spaces between columns if we have %3.3e in d, t
            msg = "{i:<13}{d:<15.3e}{t:<18.3e}"
            print(msg.format(i=it, d=dist, t=etime))

    return


def compute_fixed_point(T, v, para, attr, error_tol=1e-3, max_iter=50,
                        print_skip=5):
    """
    Computes and returns :math:`T^k v`, an approximate fixed point.

    Here T is an operator, v is an initial condition and k is the number
    of iterates. Provided that T is a contraction mapping or similar,
    :math:`T^k v` will be an approximation to the fixed point.

    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : bool, optional(default=True)
        If True then print current error at each iterate.
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called

    Returns
    -------
    v : object
        The approximate fixed point

    """

    iterate = 0
    error = error_tol + 1

    while iterate < max_iter and error > error_tol:
        new_v = T(v, para, attr)
        iterate += 1
        error = np.max(np.abs(new_v - v))

        v = new_v
    return v

"""
Filename: distributions.py

Probability distributions useful in economics.

References
----------

http://en.wikipedia.org/wiki/Beta-binomial_distribution

"""
from math import sqrt
import numpy as np
from scipy.special import binom, beta


class BetaBinomial(object):
    """
    The Beta-Binomial distribution

    Parameters
    ----------
    n : scalar(int)
        First parameter to the Beta-binomial distribution
    a : scalar(float)
        Second parameter to the Beta-binomial distribution
    b : scalar(float)
        Third parameter to the Beta-binomial distribution

    Attributes
    ----------
    n, a, b : see Parameters

    """

    def __init__(self, n, a, b):
        self.n, self.a, self.b = n, a, b

    @property
    def mean(self):
        "mean"
        n, a, b = self.n, self.a, self.b
        return n * a / (a + b)

    @property
    def std(self):
        "standard deviation"
        return sqrt(self.var)

    @property
    def var(self):
        "Variance"
        n, a, b = self.n, self.a, self.b
        top = n*a*b * (a + b + n)
        btm = (a+b)**2.0 * (a+b+1.0)
        return top / btm

    @property
    def skew(self):
        "skewness"
        n, a, b = self.n, self.a, self.b
        t1 = (a+b+2*n) * (b - a) / (a+b+2)
        t2 = sqrt((1+a+b) / (n*a*b * (n+a+b)))
        return t1 * t2

    def pdf(self):
        r"""
        Generate the vector of probabilities for the Beta-binomial
        (n, a, b) distribution.

        The Beta-binomial distribution takes the form

        .. math::
            p(k \,|\, n, a, b) =
            {n \choose k} \frac{B(k + a, n - k + b)}{B(a, b)},
            \qquad k = 0, \ldots, n,

        where :math:`B` is the beta function.

        Parameters
        ----------
        n : scalar(int)
            First parameter to the Beta-binomial distribution
        a : scalar(float)
            Second parameter to the Beta-binomial distribution
        b : scalar(float)
            Third parameter to the Beta-binomial distribution

        Returns
        -------
        probs: array_like(float)
            Vector of probabilities over k

        """
        n, a, b = self.n, self.a, self.b
        k = np.arange(n + 1)
        probs = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
        return probs

    # def cdf(self):
    #     r"""
    #     Generate the vector of cumulative probabilities for the
    #     Beta-binomial(n, a, b) distribution.

    #     The cdf of the Beta-binomial distribution takes the form

    #     .. math::
    #         P(k \,|\, n, a, b) = 1 -
    #         \frac{B(b+n-k-1, a+k+1) {}_3F_2(a,b;k)}{B(a,b) B(n-k, k+2)},
    #         \qquad k = 0, \ldots, n

    #     where :math:`B` is the beta function.

    #     Parameters
    #     ----------
    #     n : scalar(int)
    #         First parameter to the Beta-binomial distribution
    #     a : scalar(float)
    #         Second parameter to the Beta-binomial distribution
    #     b : scalar(float)
    #         Third parameter to the Beta-binomial distribution

    #     Returns
    #     -------
    #     probs: array_like(float)
    #         Vector of probabilities over k

    #     """


import matplotlib.pyplot as plt
from matplotlib import cm
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

