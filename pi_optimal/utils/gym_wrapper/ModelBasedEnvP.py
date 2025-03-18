import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ModelBasedEnvP(gym.Env):
    """
    A Gym environment wrapper around a learned dynamics model for model-based RL.
    The model is assumed to predict a full next state vector which includes:
      - Physical state features (e.g., "state_0" to "state_7")
      - A binary "done" flag (e.g., at index corresponding to 'done')
      - A "reward" value (e.g., at index corresponding to 'reward')
    
    The observation returned to the agent excludes the reward and done flag.
    
    Parameters:
      - model: a trained dynamics model (supporting a forward(state, action) method)
      - dataset_config: configuration dictionary (as in your example)
      - initial_state: full state vector (matching the length of dataset_config["states"])
      - max_episode_steps: maximum steps per episode (after which done is True)
      - default_action_n: default number of discrete actions
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, models, dataset, initial_state, max_episode_steps=200, default_action_n=2):
        super(ModelBasedEnvP, self).__init__()
        self.models = models
        self.dataset = dataset
        self.dataset_config = dataset.dataset_config
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Determine indices for reward and done from the config.
        self.reward_index = self.dataset_config["reward_feature_idx"]
        self.done_index = None
        for idx, state_conf in self.dataset_config["states"].items():
            if state_conf["name"] == "done":
                self.done_index = idx
                break
        if self.done_index is None:
            raise ValueError("No 'done' feature found in dataset config states.")

        # Full state dimension is the total number of state entries.
        self.full_state_dim = len(self.dataset_config["states"])
        # For the observation returned to the agent, remove the reward and done indices.
        self.obs_indices = [i for i in range(self.full_state_dim) if i not in [self.done_index, self.reward_index]]
        self.observation_dim = len(self.obs_indices)
        
        # Define the observation space as a continuous Box.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.observation_dim,), dtype=np.float32)
        
        # Define the action space.
        if "actions" in self.dataset_config and self.dataset_config["actions"]:
            self.action_space = spaces.Discrete(default_action_n)
        else:
            self.action_space = spaces.Discrete(default_action_n)
        
        # Store the initial state.
      
        self.initial_states = initial_state
        
    def step(self, action):
        """
        Executes one simulation step.
        
        Parameters:
          - action (int): A discrete action.
          
        Returns:
          - observation (np.array): The physical state features.
          - reward (float): Predicted reward.
          - done (bool): Episode termination flag.
          - info (dict): Additional info (empty in this example).
        """
        # Convert the discrete action to one-hot encoding.
        action_onehot = np.zeros((1, self.action_space.n), dtype=np.float32)
        action_onehot[0, action] = 1.0
        
 
        if self.action.shape[1] < self.dataset.forecast_timesteps:
            self.action = np.concatenate([self.action, action_onehot.reshape(1, 1, -1)], axis=1)
        else:
            self.action = np.concatenate([self.action[:, 1:], action_onehot.reshape(1, 1, -1)], axis=1)

       
        # Prepare current state input.
        state_input = self.state.reshape(self.state.shape[0], self.state.shape[1],  -1)
        state_input_transfrom = self.dataset._transform_features("states", state_input[0])
        state_input_transfrom = state_input_transfrom.reshape(1, self.state.shape[1], -1)
        
        if state_input_transfrom.shape[1] < self.dataset.lookback_timesteps:
            pad = np.zeros((1, self.dataset.lookback_timesteps - state_input_transfrom.shape[1], state_input_transfrom.shape[2]))
            state_input_transfrom = np.concatenate([pad, state_input_transfrom], axis=1)

        if self.action.shape[1] < self.dataset.lookback_timesteps:
            pad = np.zeros((1, self.dataset.lookback_timesteps - self.action.shape[1], self.action.shape[2]))
            self.action = np.concatenate([pad, self.action], axis=1)

        # Predict the next full state using the model's forward method.
        # Randomly select a model from the ensemble.
        model_idx = np.random.randint(0, len(self.models))
        next_state_pred = self.models[model_idx].forward(state_input_transfrom, self.action)
        next_state_pred = self.dataset.inverse_transform_features("states", next_state_pred)
        next_state = next_state_pred[0]


        
        # Extract reward and done flag.
        reward = float(next_state[self.reward_index])
        done = bool(next_state[self.done_index])
        
        # Construct observation by removing reward and done.
        observation = next_state[self.obs_indices]
        
        # Update state and step count.
        # Add the next state to the self.state history for the next step.
        if self.state.shape[1] < self.dataset.lookback_timesteps:
            self.state = np.concatenate([self.state, next_state.reshape(1, 1, -1)], axis=1)
        else:
            self.state = np.concatenate([self.state[:, 1:], next_state.reshape(1, 1, -1)], axis=1)

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
        
        return observation, reward, done, done, {}
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        
        Parameters:
          - seed (int, optional): Seed to add a small random noise to the initial state.
          - options (dict, optional): Additional options (unused here).
          
        Returns:
          - observation (np.array): The initial observation.
          - info (dict): Additional information (empty dict in this case).
        """
        self.action = np.zeros((1, 0, self.action_space.n), dtype=np.float32)
        self.current_step = 0
        index = np.random.randint(0, len(self.initial_states))
        initial_state = np.array(self.initial_states[index], dtype=np.float32)

        # If a seed is provided, use it to generate a small noise and perturb the initial state.
        if seed is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0, scale=1e-4, size=initial_state.shape)
            self.state = initial_state.copy() + noise
        else:
            noise = np.random.normal(loc=0.0, scale=1e-4, size=initial_state.shape)
            self.state = initial_state.copy() + noise
            
        self.state = self.state.reshape(1, 1, -1)
        observation = self.state[0, 0, self.obs_indices]
        return observation, {}
    
    def render(self, mode="human"):
        """
        Simple rendering: print the current step and full state.
        """
        print(f"Step: {self.current_step}, Full state: {self.state}")
    
    def close(self):
        pass



