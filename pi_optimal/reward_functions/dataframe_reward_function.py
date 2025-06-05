"""
DataFrame-based reward function for external reward calculation.

This module provides support for pandas DataFrame-based reward functions, reward calculation
is done via a function that operates on DataFrame rows.
"""

import numpy as np
import pandas as pd
from .base_reward_function import StateBasedRewardFunction

class DataFrameRewardFunction(StateBasedRewardFunction):
    """
    Optimized version of DataFrameRewardFunction that caches transformations
    and minimizes pandas overhead.
    """
    
    def __init__(self, reward_function, feature_mapping, dataset_config=None, name="OptimizedDataFrameReward"):
        super().__init__(name)
        self.reward_function = reward_function
        self.feature_mapping = feature_mapping
        self.dataset_config = dataset_config
        
        # Cache inverse transformation functions
        self._cached_transformers = {}
        self._prepare_transformers()
        
    def _prepare_transformers(self):
        """Pre-compute and cache inverse transformation functions."""
        if self.dataset_config and 'states' in self.dataset_config:
            for state_idx, column_name in self.feature_mapping.items():
                if state_idx in self.dataset_config['states']:
                    state_config = self.dataset_config['states'][state_idx]
                    if 'processor' in state_config:
                        processor = state_config['processor']
                        if hasattr(processor, 'inverse_transform'):
                            self._cached_transformers[state_idx] = processor
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Optimized reward calculation with cached transformations."""
        try:
            # Pre-allocate row data dict with expected size
            row_data = {}
            
            # Vectorized inverse transformation where possible
            for state_idx, column_name in self.feature_mapping.items():
                if state_idx < len(state):
                    raw_value = state[state_idx]
                    
                    # Use cached transformer if available
                    if state_idx in self._cached_transformers:
                        try:
                            transformer = self._cached_transformers[state_idx]
                            transformed_value = transformer.inverse_transform([[raw_value]])[0][0]
                            row_data[column_name] = transformed_value
                        except:
                            row_data[column_name] = raw_value
                    else:
                        row_data[column_name] = raw_value
            
            # Use dict directly instead of pandas Series for simple reward functions
            # This avoids pandas overhead for basic calculations
            if hasattr(self.reward_function, '__code__') and len(self.reward_function.__code__.co_names) < 10:
                # For simple reward functions, pass dict-like object
                class DictRow:
                    def __init__(self, data):
                        self.__dict__.update(data)
                    def __getitem__(self, key):
                        return self.__dict__[key]
                    def get(self, key, default=None):
                        return self.__dict__.get(key, default)
                
                row = DictRow(row_data)
            else:
                # Fall back to pandas Series for complex functions
                row = pd.Series(row_data)
            
            reward = self.reward_function(row)
            return float(reward)
            
        except Exception as e:
            # Minimal error handling to avoid performance overhead
            return 0.0
