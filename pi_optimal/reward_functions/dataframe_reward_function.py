"""
DataFrame-based reward function for external reward calculation.

This module provides support for pandas DataFrame-based reward functions, reward calculation
is done via a function that operates on DataFrame rows.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List
from .base_reward_function import StateBasedRewardFunction


class DataFrameRewardFunction(StateBasedRewardFunction):
    """
    Reward function that operates on DataFrame rows using external reward calculation logic.
    
    This class bridges the gap between numpy-based model predictions and pandas-based
    business logic reward calculations, similar to the pattern used in adpilot_init.ipynb.
    """
    
    def __init__(self, 
                 reward_function: Callable[[pd.Series], float],
                 feature_mapping: Dict[int, str],
                 dataset_config: Dict[str, Any] = None,
                 name: str = "DataFrameReward"):
        """
        Initialize DataFrame-based reward function.
        
        Args:
            reward_function: Function that takes a pandas Series (row) and returns scalar reward
            feature_mapping: Maps state indices to DataFrame column names
                           e.g., {0: "adset_impressions_diff", 1: "adset_contactdose", ...}
            dataset_config: Dataset configuration for inverse transformation
            name: Name for the reward function
            
        Example:
            def calculate_reward(row):
                reward = 0
                if row['adset_impressions_diff'] < 0.9 * row['adset_settings_maximum_daily_impressions']:
                    reward -= 2
                if row['adset_contactdose'] < 4 or row['adset_contactdose'] > 6:
                    reward -= 2
                return reward
            
            reward_func = DataFrameRewardFunction(
                reward_function=calculate_reward,
                feature_mapping={
                    0: "adset_impressions_diff",
                    1: "adset_settings_maximum_daily_impressions", 
                    2: "adset_contactdose"
                }
            )
        """
        super().__init__(name)
        self.reward_function = reward_function
        self.feature_mapping = feature_mapping
        self.dataset_config = dataset_config
        self._reverse_mapping = {v: k for k, v in feature_mapping.items()}
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """
        Calculate reward by converting state array to DataFrame row and applying reward function.
        
        Args:
            state: State vector (1D array)
            
        Returns:
            Scalar reward value
        """
        # Convert state array to DataFrame row
        row_data = {}
        
        for state_idx, column_name in self.feature_mapping.items():
            if state_idx < len(state):
                raw_value = state[state_idx]
                
                # Apply inverse transformation if dataset config is available
                if self.dataset_config and 'states' in self.dataset_config:
                    if state_idx in self.dataset_config['states']:
                        state_config = self.dataset_config['states'][state_idx]
                        if 'processor' in state_config:
                            processor = state_config['processor']
                            # Inverse transform the value
                            if hasattr(processor, 'inverse_transform'):
                                try:
                                    transformed_value = processor.inverse_transform([[raw_value]])[0][0]
                                    row_data[column_name] = transformed_value
                                except:
                                    # Fall back to raw value if transformation fails
                                    row_data[column_name] = raw_value
                            else:
                                row_data[column_name] = raw_value
                        else:
                            row_data[column_name] = raw_value
                    else:
                        row_data[column_name] = raw_value
                else:
                    row_data[column_name] = raw_value
        
        # Create pandas Series and calculate reward
        row = pd.Series(row_data)
        
        try:
            reward = self.reward_function(row)
            return float(reward)
        except Exception as e:
            # Log error and return zero reward as fallback
            print(f"Warning: Reward calculation failed for row {row_data}: {e}")
            return 0.0
    
    def fit(self, dataset_config: Dict[str, Any] = None, **kwargs):
        """Store dataset configuration for inverse transformations."""
        if dataset_config is not None:
            self.dataset_config = dataset_config
        super().fit(dataset_config, **kwargs)
    
    def add_feature_mapping(self, state_idx: int, column_name: str):
        """Add a new feature mapping."""
        self.feature_mapping[state_idx] = column_name
        self._reverse_mapping[column_name] = state_idx
    
    def get_required_features(self) -> List[str]:
        """Get list of required DataFrame column names."""
        return list(self.feature_mapping.values())
    
    def validate_feature_mapping(self, dataset_config: Dict[str, Any]) -> bool:
        """
        Validate that feature mapping matches dataset configuration.
        
        Args:
            dataset_config: Dataset configuration to validate against
            
        Returns:
            True if mapping is valid, False otherwise
        """
        if 'states' not in dataset_config:
            return False
        
        states_config = dataset_config['states']
        
        for state_idx, column_name in self.feature_mapping.items():
            if state_idx not in states_config:
                print(f"Warning: State index {state_idx} not found in dataset config")
                return False
            
            state_config = states_config[state_idx]
            if state_config.get('name') != column_name:
                print(f"Warning: Column name mismatch for state {state_idx}: "
                      f"expected '{column_name}', got '{state_config.get('name')}'")
                return False
        
        return True
