# Import grm.py file
import grm

# Process initializtion file
init_dict = grm.process('init.ini')

# Simulate dataset
grm.simulate(init_dict)

# Estimate model
rslt = grm.estimate(init_dict)

# Inspect results
grm.inspect(rslt, init_dict)

