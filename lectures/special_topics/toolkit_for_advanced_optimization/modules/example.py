import tao4py
import time
args = ['-tao_pounders_gqt']

tao4py.init(args)


from clsOpt import optCls

# Initialize solver
import numpy as np
from tao4py import TAO
from petsc4py import PETSc

# Ensure recomputability
np.random.seed(456)

# module wide variables
PARAS = [0.20, 0.12, 0.08]
START = [0.10, 0.08, 0.05]


# Initialize container for result.
rslt = dict()

for algorithm in ['tao_pounders', 'tao_nm']:
    rslt[algorithm] = dict()
    for key_ in ['time', 'rmse']:
        rslt[algorithm][key_] = []

# Initialize agent grid
agent_grid = range(1000, 10000, 5000)




for algorithm in ['tao_pounders', 'tao_nm']:

    for num_agents in agent_grid:

        # Initialize class container
        opt_obj = optCls(num_agents, PARAS)

        # Manage PETSc objects.
        paras, crit = opt_obj.create_vectors()

        opt_obj.set_initial_guess(paras, START)

        # Initialize solver instance
        tao = TAO.Solver().create(PETSc.COMM_SELF)

        tao.setType(algorithm)

        tao.setFromOptions()

        # Attach objective function
        if algorithm == 'tao_pounders':
            tao.setSeparableObjective(opt_obj.form_separable_objective, crit)
        else:
            tao.setObjective(opt_obj.form_objective)

        # Measure execution time of for estimation
        start_time = time.time()

        tao.solve(paras)

        # Collect performance measures
        rslt[algorithm]['time'] += [time.time() - start_time]

        rslt[algorithm]['rmse'] += [np.mean((PARAS - paras[:])**2)]

        print tao.getgit comm
        paras.destroy()

        crit.destroy()

        tao.destroy()

print rslt
# Plotting
import matplotlib.pyplot as plt

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

#plt.show()


# Initialize canvas
ax = plt.figure(figsize=(12,8)).add_subplot(111)

# Plot execution times by implementations
ax.plot(agent_grid, rslt['tao_pounders']['rmse'], label='POUNDerS')
ax.plot(agent_grid, rslt['tao_nm']['rmse'], label='Nelder-Mead')
# Set axis labels
ax.set_xlabel('Simulated Agents', fontsize=20)
ax.set_ylabel('Root-Mean-Squared Error', fontsize=20)

# Change background color
ax.set_axis_bgcolor('white')

# Set up legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
    fancybox=False, frameon=False, shadow=False, ncol=2,
    fontsize=20)

# Remove first element on y-axis
ax.yaxis.get_major_ticks()[0].set_visible(False)

#plt.show()