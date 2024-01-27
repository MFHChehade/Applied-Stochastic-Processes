from typing import List
import numpy as np
from chains.dtmc import DTMC

class StockModel(DTMC):
    def __init__(self, horizon: int, n_episodes: int, initial_price: float, p: float, u: float, d: float):
        """
        Initialize the StockModel with parameters.

        Parameters:
        - horizon (int): Time horizon for each episode.
        - n_episodes (int): Number of episodes for simulation.
        - initial_price (float): Initial price of the stock at time 0.
        - p (float): Probability of the stock going up.
        - u (float): Up factor for stock price.
        - d (float): Down factor for stock price.
        """
        self.X0 = initial_price
        self.p = p
        self.u = u
        self.d = d
        super().__init__(horizon=horizon, n_episodes=n_episodes)

    def rollout(self) -> List[float]:
        """
        Simulate a single episode and return the trajectory of stock prices.

        Returns:
        - List[float]: List representing the trajectory of stock prices for one episode.
        """
        x = self.X0
        trajectory = [x]
        
        for _ in range(self.horizon):
            # Randomly determine whether the stock goes up or down
            if np.random.random() < self.p:
                x *= (1 + self.u)
            else:
                x *= (1 - self.d)
            
            trajectory.append(x)
        
        return trajectory



