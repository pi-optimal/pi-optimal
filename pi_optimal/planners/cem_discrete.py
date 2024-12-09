import numpy as np
from tqdm import tqdm
from .cem_planner import CEMPlanner

class CEMDiscretePlanner(CEMPlanner):
    def __init__(self, action_dim, horizon, population_size, topk, uncertainty_weight=5.0):
        super().__init__(action_dim, horizon, population_size, topk)

        self.action_dim = action_dim
        self.mu = np.zeros((self.horizon, self.action_dim))
        self.sigma = np.ones((self.horizon, self.action_dim))
        self.uncertainty_weight = uncertainty_weight
        self.mean_costs = None
        self.std_costs = None

        self.mean_uncertainty = None
        self.std_uncertainty = None

    def generate_actions(self):
        actions_logits = np.random.normal(
            loc=self.mu[None, :, :],
            scale=self.sigma[None, :, :],
            size=(self.population_size, self.horizon, self.action_dim)
        )
        actions = np.argmax(actions_logits, axis=2)
        return actions, actions_logits

    def simulate_trajectories(self, models, states, actions, action_history):
        population_size, history_dim, state_dim = states.shape
        trajectories = []
        num_models = len(models)
        model_predictions = [[] for _ in range(num_models)]

        for t in tqdm(range(self.horizon)):
            current_actions = actions[:, t]
            current_actions_one_hot = np.zeros((self.population_size, self.action_dim))
            current_actions_one_hot[np.arange(self.population_size), current_actions] = 1

            action_history = np.roll(action_history, shift=-1, axis=1)
            action_history[:, -1, :] = current_actions_one_hot

            for idx, model in enumerate(models):
                next_states = model.forward(states, action_history)
                model_predictions[idx].append(next_states)

            # Update states for next timestep (using the first model as reference)
            states = np.roll(states, shift=-1, axis=1)
            states[:, -1, :] = model_predictions[0][-1]

        # Convert lists to numpy arrays
        for idx in range(num_models):
            model_predictions[idx] = np.stack(model_predictions[idx], axis=1)  # (population_size, horizon, state_dim)

        return model_predictions
    
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

    def update_distribution(self, actions_logits, costs):
        elite_idx = np.argsort(costs)[:self.topk]
        elite_actions = actions_logits[elite_idx, :, :]
        self.mu = np.mean(elite_actions, axis=0)
        self.sigma = np.std(elite_actions, axis=0) + 1e-6  # Avoid zero std deviation

    def get_action_sequence(self):
        return np.argmax(self.mu, axis=1)
    
