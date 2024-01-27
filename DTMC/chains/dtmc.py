from typing import List
import numpy as np

class DTMC:
    def __init__(self, horizon: int = 10, n_episodes: int = 1000, transition_matrix: np.ndarray =  None, initial_distribution: np.ndarray = None, ):
        """
        Initializes a Discrete-Time Markov Chain (DTMC) instance.

        Parameters:
        - horizon (int): The length of each episode or time horizon. Default is 10.
        - n_episodes (int): The number of episodes to simulate. Default is 1000.
        - transition_matrix (np.ndarray): The transition matrix representing the probabilities
          of transitioning from one state to another. Default is None
        - initial_distribution (np.ndarray): The initial distribution of states. Default is None.
        """
        self.horizon = horizon
        self.n_episodes = n_episodes
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution
        self.history = self.simulate()

    def reset(self) -> None:
        """
        Resets the DTMC by generating a new set of simulated episodes.
        """
        self.history = self.simulate()

    def rollout(self) -> List[int]:
        """
        Simulates a single episode or trajectory of the DTMC.

        Returns:
        - List[int]: The sequence of states in the simulated trajectory.
        """
        x = np.random.choice(range(len(self.initial_distribution)), p=self.initial_distribution) + 1
        trajectory = [x]
        for _ in range(self.horizon):
            x = np.random.choice(range(len(self.transition_matrix)), p=self.transition_matrix[x - 1]) + 1
            trajectory.append(x)
        return trajectory

    def simulate(self) -> List[List[int]]:
        """
        Simulates multiple episodes and returns a list of trajectories.

        Returns:
        - List[List[int]]: A list containing simulated trajectories.
        """
        return [self.rollout() for _ in range(self.n_episodes)]

    def get_pmf(self, n: int) -> dict:
        """
        Computes the Probability Mass Function (PMF) for a given state.

        Parameters:
        - n (int): The index of the state for which PMF is calculated.

        Returns:
        - dict: A dictionary representing the PMF of the specified state, sorted by keys.
        """
        pmf = {}
        for trajectory in self.history:
            x = trajectory[n]
            pmf[x] = pmf.get(x, 0) + 1

        # Sort the dictionary by keys
        sorted_pmf = dict(sorted(pmf.items()))

        return {key: value / self.n_episodes for key, value in sorted_pmf.items()}
    
    def expected_value(self, n: int) -> float:
        """
        Computes the expected value of a given state.

        Parameters:
        - n (int): The index of the state for which expected value is calculated.

        Returns:
        - float: The expected value of the specified state.
        """
        pmf = self.get_pmf(n)
        return sum([key * value for key, value in pmf.items()])
    
    def variance(self, n: int) -> float:
        """
        Computes the variance of a given state.

        Parameters:
        - n (int): The index of the state for which variance is calculated.

        Returns:
        - float: The variance of the specified state.
        """
        pmf = self.get_pmf(n)
        expected_value = self.expected_value(n)
        return sum([value * (key - expected_value)**2 for key, value in pmf.items()])

    def get_joint_probability(self, indices_dict: dict) -> float:
        """
        Computes the joint probability of two specified states.

        Parameters:
        - indices_dict (dict): A dictionary containing the indices of the two states (values can be int or List[int]).

        Returns:
        - float: The joint probability of the specified states.
        """
        n = list(indices_dict.keys())[0]  # Index of the first variable
        m = list(indices_dict.keys())[1]  # Index of the second variable
        count = 0  # Count number of samples in which the joint event occurs

        if isinstance(indices_dict[n], int):
            indices_dict[n] = [indices_dict[n]]
        if isinstance(indices_dict[m], int):
            indices_dict[m] = [indices_dict[m]]

        for trajectory in self.history:
            if trajectory[n] in indices_dict[n] and trajectory[m] in indices_dict[m]:
                count += 1
        return count / self.n_episodes

    def get_conditional_probability(self, indices_dict: dict) -> float:
        """
        Computes the conditional probability of two specified states.

        Parameters:
        - indices_dict (dict): A dictionary containing the indices of the two states and their values (values can be int or List[int]).

        Returns:
        - float: The conditional probability of the specified states.
        """
        n = list(indices_dict.keys())[0]
        m = list(indices_dict.keys())[1]

        if isinstance(indices_dict[n], int):
            indices_dict[n] = [indices_dict[n]]
        if isinstance(indices_dict[m], int):
            indices_dict[m] = [indices_dict[m]]

        numerator = 0
        denominator = 0
        for trajectory in self.history:
            if trajectory[m] in indices_dict[m]:
                denominator += 1
                if trajectory[n] in indices_dict[n]:
                    numerator += 1

        if denominator == 0:
            return 0
        else:
            return numerator / denominator
