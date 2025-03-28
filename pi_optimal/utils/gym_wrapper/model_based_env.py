import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union
from pi_optimal.datasets.timeseries_dataset import TimeseriesDataset

class ModelBasedEnv(gym.Env):
    """
    A generalized Gymnasium environment wrapper around a learned dynamics model for model-based RL.

    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self, 
        models: List[Any], 
        dataset: Any,
        max_episode_steps: int = 200,
        use_start_states: bool = False
    ) -> None:
        super().__init__()
        self.use_start_states = use_start_states
        self.models = models
        self.dataset = dataset
        self.dataset_config: Dict = dataset.dataset_config
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # --- Create observation and action spaces ---
        self._create_observation_space()
        self._create_action_space()
        


    def _create_observation_space(self) -> None:
        lows = []
        highs = []
        self.state_columns = []
        for state_config in self.dataset_config["states"].values():
            name = state_config["name"]
            type = state_config["type"]
            if name != self.dataset_config["reward_column"] and name != "done":
                if type == "numerical":
                    low = self.dataset.df[name].min()
                    high = self.dataset.df[name].max()
                elif type in ["categorial", "binary"]:
                    self.dataset.df[name] = self.dataset.df[name].astype("category").cat.codes
                    low = 0
                    high = len(np.unique(self.dataset.df[name]))
                else:
                    raise ValueError(f"Unknown feature type: {type}")
                lows.append(low)
                highs.append(high)
                self.state_columns.append(name)
      
        self.observation_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def _create_action_space(self) -> None:
        types = []
        for action_config in self.dataset_config["actions"].values():
            type = action_config["type"]
            types.append(type)

        if all([t == "numerical" for t in types]):
            lows = []
            highs = []
            for i in range(len(self.dataset_config["actions"])):
                low = self.dataset.actions[:, i].min()
                high = self.dataset.actions[:, i].max()
                lows.append(low)
                highs.append(high)
            self.action_space = spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                dtype=np.float32,
            )
        elif all([t == "categorial" for t in types]):
            if len(types) == 1:
                if self.dataset.actions.shape[1] == 1:
                    n = len(np.unique(self.actions))
                else:
                    n = self.dataset.actions.shape[1]
                self.action_space = spaces.Discrete(n)
            else:
                raise ValueError("MultiDiscrete action space not supported.")
        else:
            raise ValueError("Mixed action types are not supported.")


    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        
        # Add observation and action to df
        obs_and_actions = np.concatenate([np.array([self.current_episode, self.current_step], dtype=int), self.observation, action.reshape(-1)], axis=0)
        obs_action_reward_done = self._insert_at_final_positions(obs_and_actions, [self.reward_idx_df, self.done_idx_df], [self.last_reward, self.last_done])
        # Add last_reward and last_done to df

        self.df = pd.concat([self.df, pd.DataFrame(obs_action_reward_done.reshape(1, -1), columns=self.dataset.df.columns)], ignore_index=True)
        self.df[self.dataset.dataset_config["episode_column"]] = self.df[self.dataset.dataset_config["episode_column"]].astype(int)
        self.df[self.dataset.dataset_config["timestep_column"]] = self.df[self.dataset.dataset_config["timestep_column"]].astype(int)
        # Create new dataset
        inf_dataset = TimeseriesDataset(df=self.df, dataset_config=self.dataset_config, train_processors=False, is_inference=True, verbose=False)

        state_input, action_input, _, _ = inf_dataset[len(self.df)-1]
        state_input = np.expand_dims(state_input, axis=0)
        action_input = np.expand_dims(action_input, axis=0)

        # Use a randomly selected model to predict the next full state.
        model_idx = np.random.randint(0, len(self.models))
        next_state_pred = self.models[model_idx].forward(state_input, action_input)
        next_state_pred = self.dataset.inverse_transform_features("states", next_state_pred)[0]
        reward = next_state_pred[self.reward_idx]
        done = next_state_pred[self.done_idx]
        self.last_reward, self.last_done = reward, done
        
        observation = next_state_pred[self.state_idx]
        self.observation = observation
        self.current_step += 1
        self.total_steps += 1

        if self.total_steps >= self.max_episode_steps:
            done = True

        return observation, reward, done, done, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict]:
        
        self.total_steps = 0
        if self.use_start_states:
            episode_start_index = self.dataset.episode_start_index
            start_index = np.random.choice(episode_start_index)
            start_index += np.random.randint(0, 20)
            start_index = min(start_index, len(self.dataset.df)-1)
        else:
            start_index = np.random.randint(0, len(self.dataset.df))

        selected_row = self.dataset.df.iloc[start_index]
        self.observation = selected_row[self.state_columns].values
        self.last_reward = selected_row[self.dataset.dataset_config["reward_column"]]
        self.last_done = selected_row["done"]
        self.current_step = selected_row[self.dataset.dataset_config["timestep_column"]]
        self.current_episode = selected_row[self.dataset.dataset_config["episode_column"]]

        self.df = self.dataset.df.iloc[:start_index]
        self.df = self.df[self.df[self.dataset.dataset_config["episode_column"]] == self.current_episode]

        self.reward_idx_df = self.df.columns.get_loc(self.dataset.dataset_config["reward_column"])
        self.done_idx_df = self.df.columns.get_loc("done")
        
        self.state_idx = [key for key in self.dataset_config["states"] if self.dataset_config["states"][key]["name"] in self.state_columns]
        self.reward_idx =[key for key in self.dataset_config["states"] if self.dataset_config["states"][key]["name"] == self.dataset_config["reward_column"]][0]
        self.done_idx = [key for key in self.dataset_config["states"] if self.dataset_config["states"][key]["name"] == "done"][0]
        return self.observation, {}

    def render(self, mode: str = "training") -> None:
        """
        Simple rendering: prints the current step and state history.
        """
        if mode == "human":
            print(f"Step: {self.current_step}, State history: {self.observation}")
    
    def close(self) -> None:
        pass


    def _insert_at_final_positions(self, original, indices, values):
        # Create the final array
        result = np.zeros(len(original) + len(indices), dtype=original.dtype)
        
        # Create boolean mask
        mask = np.ones(len(result), dtype=bool)
        mask[indices] = False
        
        # Set values
        result[indices] = values
        result[mask] = original
        
        return result
