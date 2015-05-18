""" This module illustrates the use of the cProfile package to
    profile Python code.
"""

# standard library
import sys
import cProfile

# PYTHONPATH
sys.path.insert(0, 'grmpy')

# Package
import grmpy as gp


''' Generate profiling output.
'''

# Process initialization file
init_dict = gp.process('init.ini')

# Modify estimation request
init_dict['ESTIMATION']['maxiter'] = 0

init_dict['ESTIMATION']['version'] = 'slow'

init_dict['BASICS']['agents'] = 10000

# Simulate synthetic sample
gp.simulate(init_dict)

# Run profiler and print to screen
cProfile.run('gp.estimate(init_dict)')

# Run profiler and print to file
cProfile.run('gp.estimate(init_dict)', 'grmpy.prof')
