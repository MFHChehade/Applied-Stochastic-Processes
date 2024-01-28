# Markov Chains and Stock Model

This repository contains Python implementations for simulating and analyzing Discrete-Time Markov Chains (DTMC) and a simple Stock Model. The code is organized into the following files and directories:

## Files

### `dtmc.py`

The `DTMC` class in this file allows for the simulation and analysis of Discrete-Time Markov Chains. It includes methods for computing the Probability Mass Function (PMF), expected value, variance, joint probability, and conditional probability.

### `stock_model.py`

The `StockModel` class, which extends the `DTMC` class, represents a simple stock model. It includes methods for simulating stock price trajectories and inherits the functionality of the `DTMC` class for analysis.

### `main_dtmc.py`

This file demonstrates the usage of the `DTMC` class. It initializes a DTMC instance with a provided transition matrix, initial distribution, and other parameters. It then uses various methods to analyze the simulated trajectories.

### `main_stock_model.py`

Similar to `main_dtmc.py`, this file demonstrates the usage of the `StockModel` class. It initializes a `StockModel` instance with specific parameters and showcases how to simulate stock price trajectories and perform analysis using inherited methods from the `DTMC` class.

## Directory

### `chains`

Contains submodules related to Markov Chains.

- `dtmc.py`: Implements the `DTMC` class.
- `stock_model.py`: Implements the `StockModel` class.

### `figures`

A directory to store any relevant figures or visualizations generated during analysis.

## Requirements

Make sure you have the following dependencies installed:

- `numpy`

Install the dependencies using:

```bash
pip install -r requirements.txt
