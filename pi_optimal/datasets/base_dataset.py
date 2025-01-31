# base_dataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from pi_optimal.utils.logger import Logger

class BaseDataset(Dataset):
    """Base class for all datasets.

    Base class that sets up the dataset and provides some basic functionality. It ensures that
    the episode column is of type int, that the episode column starts from 0, that the number
    of episodes is equal to the maximum episode number, that the timestep column is of type int,
    and that the dataframe is sorted by episode and timestep.

    Attributes:
        df (pd.DataFrame): The dataframe containing the dataset.
        dataset_config (dict): The dataset configuration dictionary.
        episode_column (str): The name of the episode column.
        timestep_column (str): The name of the timestep column.
        num_episodes (int): The number of episodes in the dataset.
    """

    def __init__(self, df: pd.DataFrame, dataset_config: Dict[str, Any] = None, unit_index: str = None, timestep_column: str = None, reward_column: str = None, state_columns: List[str] = None, action_columns: List[str] = None):
        """Initialize the BaseDataset.

        Args:
            df (pd.DataFrame): The dataframe containing the dataset.
            dataset_config (Dict[str, Any]): The dataset configuration dictionary.

        Raises:
            AssertionError: If any of the input validations fail.
        """
        self.hash_id = np.random.randint(0, 100000)
        self.logger = Logger(f"Dataset-{self.hash_id}")
        self.logger.info("Initializing new dataset...", "PROCESS")
        self.df = df.copy()
        self.dataset_config = dataset_config
        self.unit_index = unit_index
        self.timestep_column = timestep_column


        if dataset_config is None:
            self.dataset_config = self._create_dataset_config(
                df, unit_index, timestep_column, reward_column, state_columns, action_columns
            )

        self._validate_input()
        self._setup_dataset()
        self._infer_action_type()

    def _create_dataset_config(
        self,
        df: pd.DataFrame,
        unit_index: str,
        timestep_column: str,
        reward_column: str,
        state_columns: List[str],
        action_columns: List[str],
    ) -> Dict[str, Any]:
        """Create a dataset configuration dictionary.

        Args:
            df (pd.DataFrame): The dataframe containing the dataset.
            unit_index (str): The name of the unit index column.
            timestep_column (str): The name of the time column.
            reward_column (str): The name of the reward column.
            state_columns (List[str]): A list of state column names.
            action_columns (List[str]): A list of action column names.

        Returns:
            Dict[str, Any]: The dataset configuration dictionary.
        """
        dataset_config = {
            "episode_column": unit_index,
            "timestep_column": timestep_column,
            "states": {},
            "actions": {},
        }

        # Process states
        for idx, column in enumerate(state_columns):
            col_dtype = df[column].dtype
            col_type, processor, eval_metric = self._infer_column_properties(col_dtype)
            dataset_config["states"][idx] = {
                "name": column,
                "type": col_type,
                "processor": processor,
                "evaluation_metric": eval_metric,
            }

        # Add reward column to states
        reward_dtype = df[reward_column].dtype
        reward_type, processor, eval_metric = self._infer_column_properties(reward_dtype)
        dataset_config["states"][len(dataset_config["states"])] = {
            "name": reward_column,
            "type": reward_type,
            "processor": processor,
            "evaluation_metric": eval_metric,
        }

        # Process actions
        for idx, column in enumerate(action_columns):
            col_dtype = df[column].dtype
            col_type, processor, eval_metric = self._infer_column_properties(col_dtype)
            dataset_config["actions"][idx] = {
                "name": column,
                "type": col_type,
                "processor": processor,
                "evaluation_metric": eval_metric,
            }

        return dataset_config

    def _infer_column_properties(self, dtype) -> (str, Optional[Dict[str, Any]], str):
        """Infer the column type, default processor, and evaluation metric based on dtype.

        Args:
            dtype: The data type of the column.

        Returns:
            Tuple[str, Optional[Dict[str, Any]], str]: A tuple containing the column type,
                default processor, and evaluation metric.
        """
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = "numerical"
            processor = {"name": "StandardScaler", "params": {}}
            eval_metric = "mae"
        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            col_type = "categorial"
            processor = {"name": "OrdinalEncoder"}
            eval_metric = "accuracy"
        elif pd.api.types.is_bool_dtype(dtype):
            col_type = "binary"
            processor = None
            eval_metric = "f1_binary"
        else:
            col_type = "unknown"
            processor = None
            eval_metric = None
        return col_type, processor, eval_metric


    def _validate_input(self):
        """Validate the input dataframe and configuration.

        Raises:
            ValueError: If any of the validations fail.
        """
        episode_col = self.dataset_config.get("episode_column")
        timestep_col = self.dataset_config.get("timestep_column")

        # Check if dataframe is empty
        if self.df.empty:
            self.logger.error("Input dataframe must not be empty.")
            raise ValueError("Input dataframe must not be empty.")

        # Check if episode and timestep columns exist
        if episode_col not in self.df.columns:
            self.logger.error(f"Episode column '{episode_col}' not found in dataframe.")
            raise ValueError(f"Episode column '{episode_col}' not found in dataframe.")
        if timestep_col not in self.df.columns:
            self.logger.error(f"Timestep column '{timestep_col}' not found in dataframe.")
            raise ValueError(f"Timestep column '{timestep_col}' not found in dataframe.")

        # Ensure episode column is of integer type
        if not pd.api.types.is_integer_dtype(self.df[episode_col]):
            try:
                self.logger.warning(f"Episode column '{episode_col}' is not of integer type.")
                self.df[episode_col] = self.df[episode_col].astype(int)
                self.logger.info(f"Converted episode column '{episode_col}' to integer type.")
            except ValueError:
                self.logger.error(f"Episode column '{episode_col}' must be of integer type.")
                raise ValueError(f"Episode column '{episode_col}' must be of integer type.")

        # Adjust episode numbers to start from 0 and be continuous
        unique_episodes = np.sort(self.df[episode_col].unique())
        expected_episodes = np.arange(unique_episodes.min(), unique_episodes.max() + 1)
        if not np.array_equal(unique_episodes, expected_episodes):
            self.logger.warning("Episodes are not continuous starting from the minimum episode number.")
            episode_mapping = {old: new for new, old in enumerate(unique_episodes)}
            self.df[episode_col] = self.df[episode_col].map(episode_mapping)
            self.logger.info("Adjusted episode numbering to start from 0 and be continuous.")

        # If timestep column is of datetime type, convert to integer indices starting from 0
        if pd.api.types.is_datetime64_any_dtype(self.df[timestep_col]):
            self.logger.warning(f"Timestep column '{timestep_col}' is of datetime type. Converting to integer indices.")

        # Check and adjust timestep column within each episode
        def process_timesteps(group):
            timesteps = group[timestep_col]
            if pd.api.types.is_datetime64_any_dtype(timesteps):
                # Convert datetime timesteps to integer indices starting from 0
                group = group.sort_values(by=timestep_col)
                group[timestep_col] = np.arange(len(group))
            elif pd.api.types.is_integer_dtype(timesteps):
                # Ensure timesteps start from 0 and increment by 1
                expected_timesteps = np.arange(len(group))
                if not np.array_equal(timesteps.values, expected_timesteps):
                    self.logger.warning(
                        f"Timesteps in episode {group[episode_col].iloc[0]} do not start from 0 and increment by 1. Adjusting timesteps."
                    )
                    group[timestep_col] = expected_timesteps
                    self.logger.info(
                        f"Adjusted timesteps in episode {group[episode_col].iloc[0]} to start from 0 and increment by 1."
                    )
            else:
                raise ValueError(f"Timestep column '{timestep_col}' must be of integer or datetime type.")
            return group

        self.df = self.df.groupby(episode_col, group_keys=False).apply(process_timesteps)

    def _setup_dataset(self):
        """Set up the dataset by sorting and resetting the index."""
        self.episode_column = self.dataset_config["episode_column"]
        self.timestep_column = self.dataset_config["timestep_column"]
        self.df = self.df.sort_values(by=[self.episode_column, self.timestep_column])
        self.df = self.df.reset_index(drop=True)
        self.num_episodes = len(self.df[self.episode_column].unique())

    def _infer_action_type(self):
        """Infer the type of actions and assign it to dataset.action_type."""
        action_types = {action["type"] for action in self.dataset_config["actions"].values()}
        if all(action_type == "numerical" for action_type in action_types):
            self.action_type = "mpc-continuous"
        elif all(action_type in {"categorial", "binary"} for action_type in action_types):
            self.action_type = "mpc-discrete"
        else:
            # If there are mixed action types, default to continuous
            self.action_type = "mpc-continuous"

    def __len__(self) -> int:
        """Return the number of rows in the dataset.

        Returns:
            int: The number of rows in the dataset.
        """
        return len(self.df)
