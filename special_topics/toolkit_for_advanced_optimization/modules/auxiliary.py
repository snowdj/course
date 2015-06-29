""" Module that contains some auxiliary functions for the illustration
    of the Toolkit for Advanced Optimization.
"""

# standard library
import numpy as np
import matplotlib.pyplot as plt

def simulate_sample(num_agents, paras):
    """ Simulate data.
    """
    # Simulate exogenous data
    exog = np.random.rand(num_agents)
    eps = np.random.normal(size=num_agents)

    # Determine endogenous data
    endog = np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog) + eps

    # Finishing
    return exog, endog

def plot_solution(paras, endog, exog):
    """ Plot the solution of the estimation run.
    """
    # Initialize grid
    u = np.linspace(exog.min(), exog.max(), 100)
    v = np.exp(-paras[0]*u)/(paras[1] + paras[2]*u)

    # Initialize canvas
    ax = plt.figure(figsize=(12,8)).add_subplot(111)

    # Plot execution times by implementations
    ax.plot(exog, endog, 'ro', label='Observed')
    ax.plot(u, v, 'b-', label='Predicted')

    # Set axis labels
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)

    # Change background color
    ax.set_axis_bgcolor('white')

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=2,
        fontsize=20)

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Add title
    plt.suptitle('Inspecting Model Fit', fontsize=20)

    # Show plot
    plt.show()