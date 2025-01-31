# timeseries_dataset.py
import pandas as pd
import numpy as np

from .base_dataset import BaseDataset
from .utils.processors import ProcessorRegistry

from typing import List, Tuple, Dict, Any
from sklearn.decomposition import PCA

class TimeseriesDataset(BaseDataset):
    """
    A dataset class for handling time series data with support for lookback and forecast.

    This class extends BaseDataset to provide functionality specific to time series data,
    including the ability to retrieve past and future data points, handle episode boundaries,
    and transform features using custom processors.

    Attributes:
        lookback_timesteps (int): Number of past timesteps to include in each sample.
        forecast_timesteps (int): Number of future timesteps to predict.
        states (np.ndarray): Transformed state features.
        actions (np.ndarray): Transformed action features.
        episode_start_index (np.ndarray): Starting indices of each episode.
        episode_end_index (np.ndarray): Ending indices of each episode.
        timestep_start_index (np.ndarray): Starting indices of each timestep within its episode.
        timestep_end_index (np.ndarray): Ending indices of each timestep within its episode.
        min_episode_length (int): Minimum length of episodes in the dataset.
        max_episode_length (int): Maximum length of episodes in the dataset.
        median_episode_length (float): Median length of episodes in the dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_config: Dict[str, Any] = None,
        unit_index: str = None, 
        timestep_column: str = None, 
        reward_column: str = None, 
        state_columns: List[str] = None, 
        action_columns: List[str] = None,
        lookback_timesteps: int = 4,
        forecast_timesteps: int = 1,
        train_processors: bool = True,
        is_inference: bool = False,
        noise_intensity_on_past_states: float = 0.0,
    ):
        """
        Initialize the TimeseriesDataset.

        Args:
            df (pd.DataFrame): The input dataframe containing the time series data.
            dataset_config (Dict[str, Any]): Configuration dictionary for the dataset.
            lookback_timesteps (int, optional): Number of past timesteps to include. Defaults to 1.
            forecast_timesteps (int, optional): Number of future timesteps to predict. Defaults to 1.
            train_processors (bool, optional): Whether to train the feature processors. Defaults to True.
            is_inference (bool, optional): Whether the dataset is used for inference. Defaults to False.
            noise_intensity_on_past_states (float, optional): Intensity of noise to add to past states. Defaults to 0.0.
        """
        super().__init__(df, dataset_config, unit_index, timestep_column, reward_column, state_columns, action_columns)
        
        self.is_inference = is_inference
        self.noise_intensity_on_past_states = noise_intensity_on_past_states

        self.lookback_timesteps = lookback_timesteps
        self.forecast_timesteps = forecast_timesteps

        self.reward_column = reward_column

        self.logger.info(f"Dataset has {len(self.df)} rows and {len(self.df.columns)} columns.", "INFO", indent_level=1)
        self.logger.info(f"Dataset has {self.df[self.dataset_config['episode_column']].nunique()} episodes.", "INFO", indent_level=1)
        self.logger.info(f"Dataset has {len(self.dataset_config['states'])} state features and {len(self.dataset_config['actions'])} actions.", "INFO", indent_level=1)

        self._calculate_episode_boundaries()
        self._calculate_timestep_boundaries()

        self.train_processors = train_processors
        if self.train_processors:
            self.logger.info("Fitting feature processors...", "PROCESS", indent_level=1)
            self._setup_processors()
            self.logger.info("Processors created and fitted", "CHECK", indent_level=2)
        else:
            self.logger.info("Using processors provided in the dataset_configuration.", "INFO", indent_level=1)

        self.logger.info("Transforming features...", "PROCESS", indent_level=1)    
        self.states = self.transform_features("states", train_processors)
        self.actions = self.transform_features("actions", train_processors)

        self.num_episodes = len(self.episode_start_index)

        self.min_episode_length = np.min(np.diff(np.r_[0, self.episode_end_index]))
        self.max_episode_length = np.max(np.diff(np.r_[0, self.episode_end_index]))
        self.median_episode_length = np.median(
            np.diff(np.r_[0, self.episode_end_index])
        )
        self.logger.info("Dataset was created successfully!", "SUCCESS", indent_level=0)  

    def get_episode(
        self, episode_id: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get the states and actions for a specific episode.

        Args:
            episode_id (int): The ID of the episode to retrieve.

        Returns:
            Tuple containing lists of past states, past actions, future states, and future actions for the episode.
        """
        episode_start_index = self.episode_start_index[episode_id]
        episode_end_index = self.episode_end_index[episode_id]

        past_states, past_actions, future_states, future_actions = [], [], [], []

        for i in range(episode_start_index, episode_end_index):
            ps, pa, fs, fa = self.__getitem__(i)
            past_states.append(ps)
            past_actions.append(pa)
            future_states.append(fs)
            future_actions.append(fa)

        past_states = np.array(past_states)
        past_actions = np.array(past_actions)
        future_states = np.array(future_states)
        future_actions = np.array(future_actions)

        return past_states, past_actions, future_states, future_actions

    def get_existing_episodes_at_timestep(self, i: int) -> np.ndarray:
        """
        Get the indices of episodes that exist at a specific timestep.

        Args:
            i (int): The timestep index to check.

        Returns:
            np.ndarray: Array of episode indices.
        """
        
        timestep_idx = self.episode_start_index + i
        return np.where(timestep_idx < self.episode_end_index)[0]


    def get_timestep_from_all_episodes(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the states and actions for a specific timestep across all episodes.

        Args:
            i (int): The timestep index to retrieve.

        Returns:
            Tuple containing past states, past actions, future states, and future actions for the timestep.
        """
        timestep_idx = self.episode_start_index + i
        timestep_idx = timestep_idx[timestep_idx < self.episode_end_index]
        past_states, past_actions, future_states, future_actions = [], [], [], []

        for idx in timestep_idx:
            ps, pa, fs, fa = self.__getitem__(idx)
            past_states.append(ps)
            past_actions.append(pa)
            future_states.append(fs)
            future_actions.append(fa)

        return np.array(past_states), np.array(past_actions), np.array(future_states), np.array(future_actions)


    def get_all_episodes(self, return_only_done=False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get the states and actions for all episodes in the dataset seperated by episodes.

        Returns:
            Tuple containing lists of past states, past actions, future states, and future actions for all episodes.
        """
        all_states, all_next_states, all_actions, all_next_actions = [], [], [], []

        if return_only_done:
            episode_range = self.num_episodes - 1 # exclude the last episode
        else:
            episode_range = self.num_episodes

        for num_episodes in range(episode_range):
            states, actions, next_states, next_action = self.get_episode(num_episodes)
            all_states.append(states)
            all_actions.append(actions)
            all_next_states.append(next_states)
            all_next_actions.append(next_action)
        
        return all_states, all_actions, all_next_states, all_next_actions

    def transform_features(
        self, feature_type: str, train_processors: bool
    ) -> np.ndarray:
        """
        Transform features of a specific type (states or actions) using their associated processors.

        Args:
            feature_type (str): Type of features to transform ('states' or 'actions').
            train_processors (bool): Whether to train the processors on the data.

        Returns:
            np.ndarray: Transformed features.

        Raises:
            AssertionError: If feature_type is not 'states' or 'actions'.
        """
        assert feature_type in [
            "states",
            "actions",
        ], "Feature type must be 'states' or 'actions'"

        transformed_features = []
        current_index = 0

        for feature_idx in self.dataset_config[feature_type]:
            feature = self.dataset_config[feature_type][feature_idx]
            transformed = self._transform_single_feature(feature, train_processors)
            if transformed.ndim == 1:
                num_cols = 1
                transformed = transformed.reshape(-1, 1)
            else:
                num_cols = transformed.shape[1]

            feature["feature_begin_idx"] = current_index
            feature["feature_end_idx"] = current_index + num_cols
            current_index += num_cols

            if feature["name"] == self.reward_column:
                self.reward_column_idx = feature_idx
                self.dataset_config["reward_feature_idx"] = feature_idx
                self.dataset_config["reward_vector_idx"] = feature["feature_begin_idx"]

            # Save the state and action size to the dataset_config
            self.dataset_config[feature_type + "_size"] = feature["feature_end_idx"]


            transformed_features.append(transformed)

            if feature["processor"] is not None:
                if hasattr(feature["processor"], "n_components"):
                    processor_information = f"with {feature['processor'].n_components} components"
                elif hasattr(feature["processor"], "n_clusters"):
                    processor_information = f"with {feature['processor'].n_clusters} clusters"
                elif hasattr(feature["processor"], "mean_") and hasattr(feature["processor"], "var_"):
                    processor_information = f"with mean {round(feature['processor'].mean_[0], 2)} and std {round(feature['processor'].var_[0] **(1/2), 2)}"
                elif hasattr(feature["processor"], "_categories"):
                    processor_information = f"with categories {feature['processor'].categories_}"
                else:
                    processor_information = ""

            self.logger.info(
                f"Transformed {feature_type} feature '{feature['name']}' using preprocessor '{feature['processor']} {processor_information}'.",
                "CHECK",
                indent_level=2,
            )

        return np.hstack(transformed_features)

    def _transform_single_feature(
        self, feature: Dict[str, Any], train_processor: bool
    ) -> np.ndarray:
        """
        Transform a single feature using its associated processor.

        Args:
            feature (Dict[str, Any]): Feature configuration dictionary.
            train_processor (bool): Whether to train the processor on the data.

        Returns:
            np.ndarray: Transformed feature.
        """
        name = feature["name"]
        processor = feature["processor"]

        if train_processor and processor is not None:
            processor.fit(self.df[name].values.reshape(-1, 1))

        transformed = (
            processor.transform(self.df[name].values.reshape(-1, 1))
            if processor
            else self.df[name].values.reshape(-1, 1)
        )

        if hasattr(transformed, "toarray"):  # For sparse matrices
            transformed = transformed.toarray()
        return np.asarray(transformed)

    def inverse_transform_features(self, feature_type: str, transformed_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform features of a specific type (states or actions) using their associated processors.

        Args:
            feature_type (str): Type of features to inverse transform ('states' or 'actions').
            transformed_data (np.ndarray): The transformed data to be inverse transformed.

        Returns:
            np.ndarray: Inverse transformed features.

        Raises:
            AssertionError: If feature_type is not 'states' or 'actions'.
        """
        assert feature_type in [
            "states",
            "actions",
        ], "Feature type must be 'states' or 'actions'"

        inverse_transformed_features = []
        
        for feature_idx in self.dataset_config[feature_type]:
            feature = self.dataset_config[feature_type][feature_idx]
            begin_idx = feature["feature_begin_idx"]
            end_idx = feature["feature_end_idx"]
            
            feature_data = transformed_data[:, begin_idx:end_idx]
            inverse_transformed = self._inverse_transform_single_feature(feature, feature_data)
            
            if inverse_transformed.ndim == 1:
                inverse_transformed = inverse_transformed.reshape(-1, 1)
            
            inverse_transformed_features.append(inverse_transformed)

        return np.hstack(inverse_transformed_features)

    def _inverse_transform_single_feature(self, feature: Dict[str, Any], feature_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform a single feature using its associated processor.

        Args:
            feature (Dict[str, Any]): Feature configuration dictionary.
            feature_data (np.ndarray): Transformed feature data to be inverse transformed.

        Returns:
            np.ndarray: Inverse transformed feature.
        """
        processor = feature.get("processor")

        if feature.get("type") == "categorial" or feature.get("type") == "binary":
            feature_data = feature_data.astype(int)

        if processor is not None and hasattr(processor, 'inverse_transform'):
            inverse_transformed = processor.inverse_transform(feature_data)
        else:
            inverse_transformed = feature_data

        if hasattr(inverse_transformed, "toarray"):  # For sparse matrices
            inverse_transformed = inverse_transformed.toarray()
        
        return np.asarray(inverse_transformed)

    def __getitem__(
        self, index: int, use_padding: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.
            use_padding (bool, optional): Whether to pad the data if necessary. Defaults to True.

        Returns:
            Tuple containing past states, past actions, future states, and future actions.
        """
        episode_start_index = self.timestep_start_index[index]
        episode_end_index = self.timestep_end_index[index]

        past_data = self._get_past_data(index, episode_start_index)
        future_data = self._get_future_data(index, episode_end_index, past_data)

        if use_padding:
            past_data, future_data = self._pad_data(past_data, future_data)

        # Add noise to the past states
        if self.noise_intensity_on_past_states > 0:
            for state_idx in self.dataset_config["states"]:
                state = self.dataset_config["states"][state_idx]
                if state["type"] == 'numerical':
                    past_data["states"][:, state["feature_begin_idx"]:state["feature_end_idx"]] += np.random.normal(
                        0, self.noise_intensity_on_past_states, past_data["states"][:, state["feature_begin_idx"]:state["feature_end_idx"]].shape
                    )

        return (
            past_data["states"],
            past_data["actions"],
            future_data["states"],
            future_data["actions"],
        )

    def _get_past_data(
        self, index: int, episode_start_index: int
    ) -> Dict[str, np.ndarray]:
        """
        Get the past states and actions for a specific index. If the dataset is a inference dataset, 
        the past data will include the current timestep.

        Args:
            index (int): Index of the data point.
            episode_start_index (int): Starting index of the episode.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing past states and actions.
        """
        past_start = max(episode_start_index, index - self.lookback_timesteps)

        if self.is_inference:
            end_index = index + 1
            past_start += 1
        else:
            end_index = index
            past_start = past_start

        return {
            "states": self.states[past_start:end_index],
            "actions": self.actions[past_start:end_index],
        }

    def _get_future_data(
        self, index: int, episode_end_index: int, past_data: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get the future states and actions for a specific index.

        Args:
            index (int): Index of the data point.
            episode_end_index (int): Ending index of the episode.
            past_data (Dict[str, np.ndarray], optional): Dictionary containing past states and actions. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing future states and actions.
        """
        future_end = min(episode_end_index + 1, index + self.forecast_timesteps)
        future_states = self.states[index:future_end]
        future_actions = self.actions[index:future_end]

        return {
            "states": future_states,
            "actions": future_actions,
        }


    def _pad_data(
        self, past_data: Dict[str, np.ndarray], future_data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Pad the past and future data to ensure consistent dimensions.

        Args:
            past_data (Dict[str, np.ndarray]): Dictionary containing past states and actions.
            future_data (Dict[str, np.ndarray]): Dictionary containing future states and actions.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Padded past and future data.
        """
        past_data = self._pad_past_data(past_data)
        future_data = self._pad_future_data(future_data)
        return past_data, future_data

    def _pad_past_data(self, past_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Pad the past data to ensure consistent dimensions.

        Args:
            past_data (Dict[str, np.ndarray]): Dictionary containing past states and actions.

        Returns:
            Dict[str, np.ndarray]: Padded past data.
        """
        if len(past_data["states"]) < self.lookback_timesteps:
            pad_width = self.lookback_timesteps - len(past_data["states"])
            past_data["states"] = np.pad(
                past_data["states"], ((pad_width, 0), (0, 0)), mode="constant"
            )
            past_data["actions"] = np.pad(
                past_data["actions"], ((pad_width, 0), (0, 0)), mode="constant"
            )
        return past_data

    def _pad_future_data(
        self, future_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Pad the future data to ensure consistent dimensions.

        Args:
            future_data (Dict[str, np.ndarray]): Dictionary containing future states and actions.

        Returns:
            Dict[str, np.ndarray]: Padded future data.
        """
        if len(future_data["states"]) < self.forecast_timesteps:
            pad_width = self.forecast_timesteps - len(future_data["states"])
            future_data["states"] = np.pad(
                future_data["states"], ((0, pad_width), (0, 0)), mode="constant"
            )
            future_data["actions"] = np.pad(
                future_data["actions"], ((0, pad_width), (0, 0)), mode="constant"
            )
        return future_data

    def _calculate_episode_boundaries(self):
        """
        Calculate the start and end indices for each episode in the dataset.
        """
        episode_sizes = self.df.groupby(self.episode_column).size()
        self.episode_end_index = episode_sizes.cumsum().values
        self.episode_start_index = np.r_[0, self.episode_end_index[:-1]]


    def _calculate_timestep_boundaries(self):
        """
        Calculate the start and end indices for each timestep within its episode.
        """
        self.episode_lengths = np.diff(np.r_[0, self.episode_end_index])
        self.timestep_start_index = np.repeat(self.episode_start_index, self.episode_lengths)
        self.timestep_end_index = np.repeat(self.episode_end_index - 1, self.episode_lengths)

    def _setup_processors(self):
        """
        Set up the feature processors for states and actions based on the dataset configuration.
        """
        for feature_type in ["states", "actions"]:
            for _, feature in self.dataset_config[feature_type].items():
                if feature["processor"] is not None:
                    processor_config = feature["processor"]
                    processor_name = processor_config["name"]
                    processor_params = processor_config.get("params", {})
                    feature["processor"] = ProcessorRegistry.get(
                        processor_name, feature["type"], **processor_params
                    )

    def _calculate_state_values(self, df: pd.DataFrame, gamma: float) -> pd.DataFrame:
        """
        Calculate the state values for each episode in the dataset.

        Args:
            df (pd.DataFrame): The input dataframe containing the time series data.
            gamma (float): The discount factor for calculating state values.

        Returns:
            pd.DataFrame: The input dataframe with state values added.
        """

        def discounted_cumsum(rewards, gamma):
            """
            Compute discounted cumulative sums of rewards.

            Args:
                rewards (pd.Series): Series of rewards for an episode.
                gamma (float): Discount factor.

            Returns:
                pd.Series: Discounted cumulative sums.
            """
            discounted = np.zeros_like(rewards, dtype=np.float64)
            running_sum = 0.0
            # Iterate from the end to the beginning
            for t in reversed(range(len(rewards))):
                running_sum = rewards.iloc[t] + gamma * running_sum
                discounted[t] = running_sum
            return discounted

        episode_coulmn = self.dataset_config["episode_column"]
        timestep_column = self.dataset_config["timestep_column"]

        df = df.sort_values([episode_coulmn, timestep_column])

        # Apply the discounted_cumsum function to each episode
        df['state_value'] = df.groupby(episode_coulmn)[self.reward_column].transform(lambda x: discounted_cumsum(x, gamma))

        return df