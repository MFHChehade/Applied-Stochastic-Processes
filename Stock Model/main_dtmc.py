import numpy as np
from chains.dtmc import DTMC

# Provided transition matrix and initial distribution
transition_matrix = np.array([
    [0.1, 0.0, 0.2, 0.3, 0.4],
    [0.0, 0.6, 0.0, 0.4, 0.0],
    [0.2, 0.0, 0.0, 0.4, 0.4],
    [0.0, 0.4, 0.0, 0.5, 0.1],
    [0.6, 0.0, 0.3, 0.1, 0.0]
])

initial_distribution = np.array([0.5, 0, 0, 0, 0.5])

# DTMC parameters
horizon = 10
n_episodes = 10000

# Create DTMC instance
dtmc_instance = DTMC(transition_matrix, initial_distribution, horizon, n_episodes)

# Access the simulation results
simulation_results = dtmc_instance.history

# Use the methods in DTMC class for analysis
# Example: Get the probability mass function for state 2
pmf_state_2 = dtmc_instance.get_pmf(2)
print("PMF for state 2:", pmf_state_2)

joint_prob_X2_X4 = dtmc_instance.get_joint_probability({2: 2, 4: 5})
print(f"Joint Probability P(X2 = 2, X4 = 5): {joint_prob_X2_X4}")

conditional_prob_X7_given_X3 = dtmc_instance.get_conditional_probability({7: 3, 3: 4})
print(f"Conditional Probability P(X7 = 3 | X3 = 4): {conditional_prob_X7_given_X3}")

joint_prob_X1_X2 = dtmc_instance.get_joint_probability({1: [1,2,3], 2: [4,5]})
print(f"Joint Probability P(X1 in [1,2,3], X2 in [4,5]): {joint_prob_X1_X2}")
