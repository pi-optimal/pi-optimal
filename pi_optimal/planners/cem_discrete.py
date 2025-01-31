import numpy as np
from tqdm.auto import tqdm
from .cem_planner import CEMPlanner

class CEMDiscretePlanner(CEMPlanner):
    def __init__(self, action_dim):
        super().__init__(action_dim)

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

        for t in tqdm(range(self.horizon), desc="Simulating trajectories"):
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
    
    def update_distribution(self, actions_logits, costs):
        elite_idx = np.argsort(costs)[:self.topk]
        elite_actions = actions_logits[elite_idx, :, :]
        self.mu = np.mean(elite_actions, axis=0)
        self.sigma = np.std(elite_actions, axis=0) + 1e-6  # Avoid zero std deviation

    def get_action_sequence(self):
        return np.argmax(self.mu, axis=1)
    
    def plan(self, models, starting_state, action_history, objective_function, n_iter = 10, allow_sigma = False, horizon = 4, population_size = 1000, topk = 100, uncertainty_weight = 0.5, reset_planer = True):
        
        if self.mu is None or reset_planer:
            self.mu = np.zeros((horizon, self.action_dim))
            self.sigma = np.ones((horizon, self.action_dim))
            
        return super().plan(models, starting_state, action_history, objective_function, n_iter, allow_sigma, horizon, population_size, topk, uncertainty_weight, reset_planer)
    
