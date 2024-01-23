import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random_walk_1d import RandomWalk1D  # Importing the RandomWalk class from random_walk.py

# Set the seed for reproducibility
np.random.seed(42)  # You can use any integer value as the seed

# Set the number of iterations
num_iterations = 10000000

# Create a RandomWalk object
random_walk = RandomWalk1D(0.5, 0.5, 0)

# Create a folder to save figures if it doesn't exist
figures_folder = 'figures'
os.makedirs(figures_folder, exist_ok=True)

# Use tqdm to display a progress bar
for i in tqdm(range(num_iterations), desc="Simulating 1D Random Walk", position=0):
    random_walk.step()
    random_walk.check_return()

# Plot the cumulative average on a log scale for the y-axis
plt.plot(random_walk.cum_average)
plt.title('Cumulative Average of Random Walk')
plt.xlabel('Number of Steps')
plt.ylabel('Cumulative Average')

# Save the figure in the "figures" folder
figure_path = os.path.join(figures_folder, 'cumulative_average_plot.png')
plt.savefig(figure_path)

# Close the plot to prevent opening a new figure in the next run
plt.close()