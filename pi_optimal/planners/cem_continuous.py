import numpy as np
from tqdm.auto import tqdm
from .cem_planner import CEMPlanner

class CEMContinuousPlanner(CEMPlanner):
    def __init__(self, action_dim, constraints=None):
        super().__init__(action_dim)
        self.constraints = constraints

    def generate_actions(self):
        actions = np.random.normal(
            loc=self.mu[None, :, :],
            scale=self.sigma[None, :, :],
            size=(self.population_size, self.horizon, self.action_dim)
        )

        # Clip actions to the action space bounds
        if self.constraints is not None:
            actions = np.clip(actions, self.constraints["min"], self.constraints["max"])
        return actions, actions

    def simulate_trajectories(self, models, states, actions, action_history):
        population_size, history_dim, state_dim = states.shape
        trajectories = []

        # Extend the state with a new dimension and copy the state, the dimension should be the number of models
        states = np.repeat(states[None, :, :, :], len(models), axis=0)
        num_models = len(models)
        model_predictions = [[] for _ in range(num_models)]

        for t in tqdm(range(self.horizon), desc="Simulating trajectories"):
            current_actions = actions[:, t, :]  # Shape: (population_size, action_dim)

            action_history = np.roll(action_history, shift=-1, axis=1)
            action_history[:, -1, :] = current_actions

            for idx, model in enumerate(models):
                next_states = model.forward(states[idx], action_history)
                model_predictions[idx].append(next_states)

                # Update states for next timestep
                states[idx] = np.roll(states[idx], shift=-1, axis=1)
                states[idx, :, -1, :] = model_predictions[idx][-1] 

        # Convert lists to numpy arrays
        for idx in range(num_models):
            model_predictions[idx] = np.stack(model_predictions[idx], axis=1)  # (population_size, horizon, state_dim)

        return model_predictions
    
    def update_distribution(self, actions_samples, costs):
        elite_idx = np.argsort(costs)[:self.topk]
        elite_actions = actions_samples[elite_idx, :, :]
        self.mu = np.mean(elite_actions, axis=0)
        self.sigma = np.std(elite_actions, axis=0) + 1e-6  # Avoid zero std deviation

    def get_action_sequence(self):
        return self.mu  # For continuous actions, the mean represents the optimal actions
    
    def plan(self, models, starting_state, action_history, objective_function, n_iter = 10, allow_sigma = False, horizon = 4, population_size = 1000, topk = 100, uncertainty_weight = 0.5, reset_planer = True):

        if self.mu is None or reset_planer:
            self.mu = np.zeros((horizon, self.action_dim))
            self.sigma = np.ones((horizon, self.action_dim))

        return super().plan(models, starting_state, action_history, objective_function, n_iter, allow_sigma, horizon, population_size, topk, uncertainty_weight, reset_planer)
