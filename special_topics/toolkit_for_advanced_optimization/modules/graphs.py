""" Module that contains the functions to create the graphs for the lecture on the Toolkit for Advanced Optimization.
"""

# standard library
import matplotlib.pyplot as plt

def plot_performance_time(rslt, agent_grid):
    """ Plot the execution time for the two algorithms.
    """
    # Antibugging
    assert (isinstance(rslt, dict))

    # Initialize canvas
    ax = plt.figure(figsize=(12,8)).add_subplot(111)

    # Plot execution times by implementations
    ax.plot(agent_grid, rslt['tao_pounders']['time'], label='POUNDerS')
    ax.plot(agent_grid, rslt['tao_nm']['time'], label='Nelder-Mead')

    # Set axis labels
    ax.set_xlabel('Simulated Agents', fontsize=20)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=20)

    # Change background color
    ax.set_axis_bgcolor('white')

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=2,
        fontsize=20)

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Add title
    plt.suptitle('Comparing Performance I', fontsize=20)

    # Show plot
    plt.show()

def plot_performance_rmse(rslt, agent_grid):
    """ Plot the RMSE for the two algorithms.
    """
    # Antibugging
    assert (isinstance(rslt, dict))

    # Initialize canvas
    ax = plt.figure(figsize=(12,8)).add_subplot(111)

    # Plot execution times by implementations
    ax.plot(agent_grid, rslt['tao_pounders']['rmse'], label='POUNDerS')
    ax.plot(agent_grid, rslt['tao_nm']['rmse'], label='Nelder-Mead')

    # Set axis labels
    ax.set_xlabel('Simulated Agents', fontsize=20)
    ax.set_ylabel('Root-Mean Squared Error', fontsize=20)

    # Change background color
    ax.set_axis_bgcolor('white')

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
        fancybox=False, frameon=False, shadow=False, ncol=2,
        fontsize=20)

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Add title
    plt.suptitle('Comparing Performance II', fontsize=20)
    plt.ylim([0.00,0.20])

    # Show plot
    plt.show()