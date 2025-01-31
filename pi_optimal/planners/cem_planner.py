from abc import ABC, abstractmethod
from .base_planner import BasePlanner
import numpy as np
from pi_optimal.utils.logger import Logger

class CEMPlanner(BasePlanner):
    def __init__(self, action_dim, logger=None):

        self.action_dim = action_dim

        # Initialize mean and standard deviation
        self.mu = None  # To be defined in subclasses
        self.sigma = None  # To be defined in subclasses
        self.current_iter = 0

        if logger is not None:
            self.logger = logger
        else:
            self.hash_id = np.random.randint(0, 100000)
            self.logger = Logger(f"CEM-Planner-{self.hash_id}")

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

    @abstractmethod
    def get_action_sequence(self):
        pass
    
    def plan(self, 
             models, 
             starting_state, 
             action_history, 
             objective_function, 
             n_iter: int = 10, 
             allow_sigma: bool = False, 
             horizon: int = 4,
             population_size: int = 1000, 
             topk: int = 100,
             uncertainty_weight: float = 0.5,
             reset_planer: bool = True):
        
        self.validate_planing_params(n_iter, horizon, population_size, topk, uncertainty_weight)

        self.reset_planer = reset_planer
        self.horizon = horizon
        self.population_size = population_size
        self.topk = topk
        self.allow_sigma = allow_sigma
        self.uncertainty_weight = uncertainty_weight

        states = np.tile(starting_state, (population_size, 1, 1))  # (population_size, history_length, state_dim)
        action_history = np.tile(action_history, (population_size, 1, 1))  # (population_size, history_length, action_dim)

        for i in range(n_iter):
            actions, actions_samples = self.generate_actions()
            trajectories = self.simulate_trajectories(models, states.copy(), actions, action_history.copy())
            costs, cost_contribution, uncertainty_contribution = self.evaluate_trajectories(trajectories, objective_function)
            self.update_distribution(actions_samples, costs)
            topk_cost = costs[np.argsort(costs)[:topk]].mean()
            topk_cost_contribution = cost_contribution[np.argsort(costs)[:topk]].mean()
            topk_uncertainty_contribution = uncertainty_contribution[np.argsort(costs)[:topk]].mean()
            self.logger.info(f"Iteration: {i+1}, Top-{topk} Cost: {round(topk_cost, 4)} (Cost: {round(topk_cost_contribution, 4)}, Uncertainty: {round(topk_uncertainty_contribution, 4)})", indent_level=1)

            if not allow_sigma:
                self.sigma = np.ones_like(self.sigma)

            self.current_iter += 1
            
        return self.get_action_sequence()
    
    def evaluate_trajectories(self, ensemble_trajectories, objective_function):
        '''
        Evaluate the actions using the model predictions and the objective function.
        '''
        num_models = len(ensemble_trajectories)
        population_size = self.population_size

        # ensemble_trajectories: list of arrays, each with shape (population_size, horizon, state_dim)
        # Stack trajectories to shape (num_models, population_size, horizon, state_dim)
        ensemble_trajectories = np.array(ensemble_trajectories)

        # Initialize costs array
        costs_per_model = np.zeros((num_models, population_size))

        # Compute costs for each model
        for idx in range(num_models):
            costs_per_model[idx] = np.array([objective_function(traj) for traj in ensemble_trajectories[idx]])

        # Compute mean cost across models for each trajectory
        mean_costs = np.mean(costs_per_model, axis=0)  # Shape: (population_size,)

        # Compute variance of costs across models (uncertainty)
        state_uncertainty = np.var(ensemble_trajectories, axis=(0, 2, 3))  # Adjust axes as needed

        # Min-max normalization directly
        epsilon = 1e-8  # Small constant to prevent division by zero
        mean_costs_normalized = (mean_costs - mean_costs.min()) / (mean_costs.max() - mean_costs.min() + epsilon)
        state_uncertainty_normalized = (state_uncertainty - state_uncertainty.min()) / (state_uncertainty.max() - state_uncertainty.min() + epsilon)

        # Combine mean cost and uncertainty
        total_costs = (1 - self.uncertainty_weight) * mean_costs_normalized + self.uncertainty_weight * state_uncertainty_normalized

        # Calculate cost and uncertainty contribution
        cost_contribution = ((1 - self.uncertainty_weight) * mean_costs_normalized) / (total_costs + epsilon)
        uncertainty_contribution = (self.uncertainty_weight * state_uncertainty_normalized) / (total_costs + epsilon)

        return total_costs, cost_contribution, uncertainty_contribution
    
    def validate_planing_params(self, n_iter, horizon, population_size, topk, uncertainty_weight):
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if not isinstance(topk, int) or topk <= 0 or topk > population_size:
            raise ValueError("topk must be a positive integer and less than or equal to population_size.")
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")
        if not isinstance(uncertainty_weight, float) or uncertainty_weight < 0 or uncertainty_weight > 1:
            raise ValueError("uncertainty_weight must be a float between 0 and 1.")


    def visualize(self, models, starting_state, action_history, actions):
        trajectories = self.simulate_trajectories(models, states.copy(), actions, action_history.copy())
