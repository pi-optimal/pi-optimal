"""
Common reward function implementations for typical use cases.

This module provides ready-to-use reward functions that can be configured
and used during planning without model retraining.
"""

import numpy as np
from typing import Dict, List, Union, Callable
from .base_reward_function import StateBasedRewardFunction, ActionBasedRewardFunction


class LinearStateRewardFunction(StateBasedRewardFunction):
    """
    Linear combination of state features with configurable weights.
    
    Reward = sum(weights[i] * state[i] for i in selected_features)
    """
    
    def __init__(self, 
                 weights: Union[np.ndarray, List[float]], 
                 feature_indices: List[int] = None,
                 name: str = "LinearStateReward"):
        """
        Initialize linear state reward function.
        
        Args:
            weights: Coefficients for linear combination
            feature_indices: Which state features to use (None = use all)
            name: Name for the reward function
        """
        super().__init__(name)
        self.weights = np.array(weights)
        self.feature_indices = feature_indices
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Calculate linear combination of selected state features."""
        if self.feature_indices is not None:
            selected_state = state[self.feature_indices]
        else:
            selected_state = state
        
        if len(selected_state) != len(self.weights):
            raise ValueError(f"State dimension {len(selected_state)} doesn't match weights dimension {len(self.weights)}")
        
        return float(np.dot(self.weights, selected_state))


class TargetStateRewardFunction(StateBasedRewardFunction):
    """
    Reward function that maximizes proximity to target state values.
    
    Uses negative distance to target as reward (closer = higher reward).
    """
    
    def __init__(self, 
                 target_values: Union[np.ndarray, List[float]],
                 feature_indices: List[int] = None,
                 distance_metric: str = "euclidean",
                 scale_factor: float = 1.0,
                 name: str = "TargetStateReward"):
        """
        Initialize target state reward function.
        
        Args:
            target_values: Target values for selected features
            feature_indices: Which state features to use (None = use all)
            distance_metric: "euclidean", "manhattan", or "squared"
            scale_factor: Scaling factor for the reward
            name: Name for the reward function
        """
        super().__init__(name)
        self.target_values = np.array(target_values)
        self.feature_indices = feature_indices
        self.distance_metric = distance_metric
        self.scale_factor = scale_factor
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Calculate negative distance to target state."""
        if self.feature_indices is not None:
            selected_state = state[self.feature_indices]
        else:
            selected_state = state
        
        if len(selected_state) != len(self.target_values):
            raise ValueError(f"State dimension {len(selected_state)} doesn't match target dimension {len(self.target_values)}")
        
        diff = selected_state - self.target_values
        
        if self.distance_metric == "euclidean":
            distance = np.sqrt(np.sum(diff ** 2))
        elif self.distance_metric == "manhattan":
            distance = np.sum(np.abs(diff))
        elif self.distance_metric == "squared":
            distance = np.sum(diff ** 2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return float(-distance * self.scale_factor)


class ThresholdRewardFunction(StateBasedRewardFunction):
    """
    Reward function based on state features crossing thresholds.
    
    Provides positive/negative rewards when features cross specified thresholds.
    """
    
    def __init__(self, 
                 thresholds: Dict[int, Dict[str, float]],
                 name: str = "ThresholdReward"):
        """
        Initialize threshold-based reward function.
        
        Args:
            thresholds: Dict mapping feature_idx to threshold config:
                       {feature_idx: {"threshold": value, "reward_above": r1, "reward_below": r2}}
            name: Name for the reward function
        """
        super().__init__(name)
        self.thresholds = thresholds
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Calculate reward based on threshold crossings."""
        total_reward = 0.0
        
        for feature_idx, config in self.thresholds.items():
            if feature_idx >= len(state):
                continue
                
            feature_value = state[feature_idx]
            threshold = config["threshold"]
            
            if feature_value >= threshold:
                total_reward += config.get("reward_above", 0.0)
            else:
                total_reward += config.get("reward_below", 0.0)
        
        return float(total_reward)


class CustomFunctionRewardFunction(StateBasedRewardFunction):
    """
    Wrapper for custom user-defined reward functions.
    
    Allows users to provide arbitrary Python functions as rewards.
    """
    
    def __init__(self, 
                 reward_function: Callable[[np.ndarray], float],
                 name: str = "CustomReward"):
        """
        Initialize custom function reward.
        
        Args:
            reward_function: Function that takes state array and returns scalar reward
            name: Name for the reward function
        """
        super().__init__(name)
        self.reward_function = reward_function
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Delegate to user-provided function."""
        return float(self.reward_function(state))


class ActionPenaltyRewardFunction(ActionBasedRewardFunction):
    """
    Reward function that penalizes large or frequent actions.
    
    Useful for encouraging smooth control or energy efficiency.
    """
    
    def __init__(self, 
                 action_weights: Union[np.ndarray, List[float]] = None,
                 penalty_type: str = "l2",
                 scale_factor: float = 1.0,
                 name: str = "ActionPenalty"):
        """
        Initialize action penalty reward function.
        
        Args:
            action_weights: Weights for different action dimensions
            penalty_type: "l1", "l2", or "max"
            scale_factor: Scaling factor for the penalty
            name: Name for the reward function
        """
        super().__init__(name)
        self.action_weights = np.array(action_weights) if action_weights is not None else None
        self.penalty_type = penalty_type
        self.scale_factor = scale_factor
    
    def calculate_reward(self, 
                        state: np.ndarray, 
                        action: np.ndarray, 
                        next_state: np.ndarray = None) -> float:
        """Calculate action penalty."""
        if len(action) == 0:
            return 0.0
        
        weighted_action = action
        if self.action_weights is not None:
            if len(self.action_weights) != len(action):
                raise ValueError(f"Action weights dimension {len(self.action_weights)} doesn't match action dimension {len(action)}")
            weighted_action = action * self.action_weights
        
        if self.penalty_type == "l1":
            penalty = np.sum(np.abs(weighted_action))
        elif self.penalty_type == "l2":
            penalty = np.sqrt(np.sum(weighted_action ** 2))
        elif self.penalty_type == "max":
            penalty = np.max(np.abs(weighted_action))
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")
        
        return float(-penalty * self.scale_factor)


class CompositeRewardFunction(StateBasedRewardFunction):
    """
    Combines multiple reward functions with configurable weights.
    
    Allows building complex reward functions from simpler components.
    """
    
    def __init__(self, 
                 reward_functions: List[StateBasedRewardFunction],
                 weights: Union[np.ndarray, List[float]] = None,
                 name: str = "CompositeReward"):
        """
        Initialize composite reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Weights for combining rewards (default: equal weights)
            name: Name for the reward function
        """
        super().__init__(name)
        self.reward_functions = reward_functions
        
        if weights is None:
            self.weights = np.ones(len(reward_functions)) / len(reward_functions)
        else:
            self.weights = np.array(weights)
            
        if len(self.weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
    
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """Calculate weighted combination of component rewards."""
        total_reward = 0.0
        
        for weight, reward_func in zip(self.weights, self.reward_functions):
            component_reward = reward_func.calculate_state_reward(state)
            total_reward += weight * component_reward
        
        return float(total_reward)
    
    def fit(self, dataset_config: Dict = None, **kwargs):
        """Fit all component reward functions."""
        super().fit(dataset_config, **kwargs)
        for reward_func in self.reward_functions:
            reward_func.fit(dataset_config, **kwargs)