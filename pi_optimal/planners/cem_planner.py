from abc import ABC, abstractmethod
from .base_planner import BasePlanner
import numpy as np

class CEMPlanner(BasePlanner):
    def __init__(self, action_dim, horizon, population_size, topk):
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if not isinstance(topk, int) or topk <= 0 or topk > population_size:
            raise ValueError("topk must be a positive integer and less than or equal to population_size.")

        self.action_dim = action_dim
        self.horizon = horizon
        self.population_size = population_size
        self.topk = topk

        # Initialize mean and standard deviation
        self.mu = None  # To be defined in subclasses
        self.sigma = None  # To be defined in subclasses
        self.current_iter = 0

    @abstractmethod
    def generate_actions(self):
        pass

    @abstractmethod
    def simulate_trajectories(self, models, states, actions, action_history):
        pass

    @abstractmethod
    def evaluate_trajectories(self, trajectories, objective_function):
        pass

    @abstractmethod
    def update_distribution(self, actions_samples, costs):
        pass

    def plan(self, models, starting_state, action_history, objective_function, n_iter=10, allow_sigma=False):
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")

        population_size = self.population_size
        states = np.tile(starting_state, (population_size, 1, 1))  # (population_size, history_length, state_dim)
        action_history = np.tile(action_history, (population_size, 1, 1))  # (population_size, history_length, action_dim)

        for i in range(n_iter):
            actions, actions_samples = self.generate_actions()
            trajectories = self.simulate_trajectories(models, states.copy(), actions, action_history.copy())
            costs, cost_contribution, uncertainty_contribution = self.evaluate_trajectories(trajectories, objective_function)
            self.update_distribution(actions_samples, costs)
            topk_cost = costs[np.argsort(costs)[:self.topk]].mean()
            topk_cost_contribution = cost_contribution[np.argsort(costs)[:self.topk]].mean()
            topk_uncertainty_contribution = uncertainty_contribution[np.argsort(costs)[:self.topk]].mean()
            print(f"Iteration: {i+1}, Top-{self.topk} Cost: {round(topk_cost, 4)} (Cost: {round(topk_cost_contribution, 4)}, Uncertainty: {round(topk_uncertainty_contribution, 4)})")

            if not allow_sigma:
                self.sigma = np.ones_like(self.sigma)

            self.current_iter += 1
            
        return self.get_action_sequence()
    
    @abstractmethod
    def get_action_sequence(self):
        pass


    def visualize(self, models, starting_state, action_history, actions):
        trajectories = self.simulate_trajectories(models, states.copy(), actions, action_history.copy())
