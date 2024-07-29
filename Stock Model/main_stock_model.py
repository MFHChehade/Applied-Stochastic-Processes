import matplotlib.pyplot as plt
import numpy as np
from chains.stock_model import StockModel
from matplotlib.ticker import ScalarFormatter
import os

# Stock model parameters
horizon = 25
n_episodes = 100000
initial_price = 1
u = 0.1
d = 0.05

# Values of p for which you want to calculate the expected value
p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Values of n for which you want to calculate the expected value
n_values = list(range(horizon))

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Use ScalarFormatter for the y-axis
ax.yaxis.set_major_formatter(ScalarFormatter())

# Create a folder for figures if it doesn't exist
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

for p in p_values:
    # Create StockModel with the current p value
    stock_model = StockModel(horizon=horizon, n_episodes=n_episodes, initial_price=initial_price, p=p, u=u, d=d)

    # Calculate expected values and variances for different n with the current p value
    expected_values = [stock_model.expected_value(n=n) for n in n_values]
    variances = [stock_model.variance(n=n) for n in n_values]

    # Plotting for Expected Value
    ax.plot(n_values, expected_values, marker='o', label=f'p = {p}')
    
    # Deduce standard deviation from variance
    std_dev_values = np.sqrt(variances)

    # Shade the area between mean + std_dev and mean - std_dev
    ax.fill_between(n_values, np.array(expected_values) + np.array(std_dev_values), np.array(expected_values) - np.array(std_dev_values), alpha=0.3)

# Adding legend
ax.legend()

# Set labels and title
ax.set_title('Mean with One Standard Deviation Shaded')
ax.set_xlabel('n')
ax.set_ylabel('Mean (Log Scale)')
# ax.set_yscale('log')

# Save the figure
output_filename = os.path.join(output_folder, 'stock_model.png')
plt.savefig(output_filename)

# Show the plot
plt.show()
