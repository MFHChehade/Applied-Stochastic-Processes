import numpy as np

class RandomWalk1D:
    def __init__(self, p, q, x0):
        """
        Initialize a 1D Random Walk.

        Parameters:
        - p: Probability of moving to the right (+1)
        - q: Probability of moving to the left (-1)
        - x0: Initial position
        """
        self.p = p
        self.q = q
        self.x0 = x0
        self.x = x0
        self.t = 0
        self.history = [x0]
        self.steps = []
        self.cum_average = []

    def step(self):
        """
        Take a single step in the random walk.
        """
        self.x = self.x + np.random.choice([-1, 1], p=[self.q, self.p])
        self.t += 1
        self.history.append(self.x)

    def reset(self):
        """
        Reset the random walk to its initial state.
        """
        self.x = self.x0
        self.t = 0
        self.history = [self.x0]
        self.steps = []

    def check_return(self):
        """
        Check if the random walk has returned to the initial position.
        If so, update the steps and cumulative average, then reset.
        """
        if self.x == self.x0:
            self.steps.append(self.t)
            self.cum_average.append(np.mean(self.steps))
            self.reset()
