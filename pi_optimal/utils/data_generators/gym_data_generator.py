# gym_data_generator.py
import random
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from .base_data_generator import BaseDataGenerator


class GymDataGenerator(BaseDataGenerator):
    """
    A data generator for collecting datasets from Gymnasium environments.

    This class extends BaseDataGenerator to provide functionality for collecting
    episode data from Gymnasium environments, which can be used for training
    and evaluating reinforcement learning algorithms.

    Attributes:
        env: A Gymnasium environment instance.

    Example:
        generator = GymDataGenerator("LunarLander-v2")
        data = generator.collect(n_episodes=1000, max_steps=300)
    """

    def __init__(self, env_name: str = "LunarLander-v2"):
        """
        Initializes the GymDataGenerator with a specified Gymnasium environment.

        Args:
            env_name: The name of the Gymnasium environment to use.
        """
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def collect(
        self,
        n_steps: int = 1000,
        max_steps_per_episode: int = 300,
        env_seed: Optional[int] = None,
        action_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Collects a dataset from the Gymnasium environment.

        Args:
            n_steps: Total number of steps to collect.
            max_steps_per_episode: Maximum number of steps allowed in each episode.
            env_seed: Seed for the environment's random number generator.
            action_seed: Seed for the action space's random number generator.

        Returns:
            A pandas DataFrame containing the collected dataset.
        """
        if n_steps <= 0 or max_steps_per_episode <= 0:
            raise ValueError("n_episodes and max_steps must be positive integers.")

        self.env = gym.make(
            self.env_name
        )  # Some environments are not resettable (e.g. BipedalWalker-v3)

        self.env = TimeLimit(self.env, max_episode_steps=max_steps_per_episode - 1)
        data = self._collect_data(n_steps, max_steps_per_episode, env_seed, action_seed)
        return self._create_dataframe(data)

    def _collect_data(
        self,
        n_steps: int,
        max_steps_per_episode: int,
        env_seed: Optional[int] = None,
        action_seed: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Collects raw data from the environment for the specified number of episodes.

        In this implementation, the reward stored for each step is the reward received
        for reaching the current state, not for taking the action from the current state.
        This aligns with viewing the reward as part of the state observation.

        Args:
            n_steps: Total number of steps to collect.
            max_steps_per_episode: Maximum number of steps allowed in each episode.
            env_seed: Seed for the environment's random number generator.
            action_seed: Seed for the action space's random number generator.

        Returns:
            A list of tuples containing the collected data for each step.
            Each tuple contains (episode, step, state, action, reward, done).
            The reward is associated with reaching the current state.
        """
        data = []
        steps_collected = 0
        episode = 0

        with tqdm(total=n_steps, desc="Collecting steps") as pbar:
            while steps_collected < n_steps:
                random.seed(action_seed)
                np.random.seed(action_seed)
                self.env.action_space.seed(action_seed)
                state, _ = self.env.reset(seed=env_seed)

                reward = float(0)
                terminated = False
                truncated = False
                step = 0

                while not (terminated or truncated) and steps_collected < n_steps:
                    action = self._sample_action()
                    next_state, next_reward, next_terminated, next_truncated, _ = (
                        self.env.step(action)
                    )
                    if len(action.shape) == 0:
                        action = np.array([action], dtype=np.float32)
                    data.append(
                        (episode, step, state, action, reward, terminated or truncated)
                    )

                    state = next_state
                    reward = float(next_reward)
                    terminated = next_terminated
                    truncated = next_truncated

                    step += 1
                    steps_collected += 1
                    pbar.update(1)

                    if step >= max_steps_per_episode:
                        break

                if (terminated or truncated) and steps_collected < n_steps:
                    next_action = self._sample_action()
                    if len(next_action.shape) == 0:
                        next_action = np.array([next_action], dtype=np.float32)
                    data.append(
                        (
                            episode,
                            step,
                            next_state,
                            next_action,
                            next_reward,
                            next_terminated or next_truncated,
                        )
                    )
                    steps_collected += 1
                    pbar.update(1)

                episode += 1

        return data

    def _sample_action(self) -> np.ndarray:
        """
        Samples a random action from the environment's action space.

        Returns:
            A numpy array representing the sampled action.
        """
        action = self.env.action_space.sample()
        return action

    def _create_dataframe(self, data: List[Tuple]) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from the collected raw data.

        Args:
            data: A list of tuples containing the collected data.

        Returns:
            A pandas DataFrame with columns for episode, step, state, action, reward, and done.
        """
        unpacked_data: Dict[str, List[Any]] = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
        }
        for trajectory in data:
            for i, key in enumerate(unpacked_data.keys()):
                unpacked_data[key].append(trajectory[i])

        df = pd.DataFrame(
            {
                "episode": unpacked_data["episode"],
                "step": unpacked_data["step"],
                "reward": unpacked_data["reward"],
                "done": unpacked_data["done"],
            }
        )
        state_cols = [f"state_{i}" for i in range(len(unpacked_data["state"][0]))]
        df[state_cols] = pd.DataFrame(unpacked_data["state"], columns=state_cols)
        action_cols = [f"action_{i}" for i in range(len(unpacked_data["action"][0]))]
        df[action_cols] = pd.DataFrame(unpacked_data["action"], columns=action_cols)
        return df
