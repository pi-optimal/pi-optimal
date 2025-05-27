"""
Base classes for modular reward function system.

This module provides the foundation for decoupling reward calculation from model training,
allowing explicit reward definition during planning phase without requiring model retraining.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Dict, Any


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions that can be used during planning
    without requiring model retraining.
    
    Reward functions take state and action information and return scalar rewards.
    They operate independently of the model's prediction pipeline.
    """
    
    def __init__(self, name: str = "BaseRewardFunction"):
        """
        Initialize the reward function.
        
        Args:
            name: Human-readable name for the reward function
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def calculate_reward(self, 
                        state: np.ndarray, 
                        action: np.ndarray, 
                        next_state: np.ndarray = None) -> float:
        """
        Calculate reward based on state, action, and optionally next state.
        
        Args:
            state: Current state vector (1D array)
            action: Action taken (1D array)
            next_state: Next state vector (1D array), optional
            
        Returns:
            Scalar reward value
        """
        pass
    
    def calculate_trajectory_reward(self, trajectory: np.ndarray) -> float:
        """
        Calculate total reward for a complete trajectory.
        
        This is used by planners during trajectory evaluation.
        
        Args:
            trajectory: Array of shape (horizon, state_dim) containing predicted states
            
        Returns:
            Total reward for the trajectory
        """
        # Default implementation sums rewards across trajectory
        total_reward = 0.0
        for i in range(len(trajectory)):
            # For trajectory evaluation, we only have states
            # Some reward functions may need to be overridden for this use case
            total_reward += self.calculate_reward(
                state=trajectory[i], 
                action=np.array([]),  # Empty action for state-only evaluation
                next_state=trajectory[i+1] if i < len(trajectory)-1 else None
            )
        return total_reward
    
    def fit(self, dataset_config: Dict[str, Any] = None, **kwargs):
        """
        Optional fitting method for reward functions that need dataset information.
        
        Args:
            dataset_config: Dataset configuration dictionary
            **kwargs: Additional fitting parameters
        """
        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if the reward function has been fitted (if required)."""
        return self._is_fitted
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class StateBasedRewardFunction(BaseRewardFunction):
    """
    Base class for reward functions that depend only on state information.
    
    This is useful for rewards that can be calculated directly from the predicted
    state without needing action information.
    """
    
    @abstractmethod
    def calculate_state_reward(self, state: np.ndarray) -> float:
        """
        Calculate reward based only on state.
        
        Args:
            state: State vector (1D array)
            
        Returns:
            Scalar reward value
        """
        pass
    
    def calculate_reward(self, 
                        state: np.ndarray, 
                        action: np.ndarray, 
                        next_state: np.ndarray = None) -> float:
        """Implementation that delegates to state-only calculation."""
        return self.calculate_state_reward(state)
    
    def calculate_trajectory_reward(self, trajectory: np.ndarray) -> float:
        """Optimized trajectory evaluation for state-only rewards."""
        return sum(self.calculate_state_reward(state) for state in trajectory)


class ActionBasedRewardFunction(BaseRewardFunction):
    """
    Base class for reward functions that depend on both state and action information.
    
    This requires access to the action sequence during trajectory evaluation.
    """
    
    def calculate_trajectory_reward(self, trajectory: np.ndarray, actions: np.ndarray = None) -> float:
        """
        Calculate total reward for trajectory with action sequence.
        
        Args:
            trajectory: Array of shape (horizon, state_dim)
            actions: Array of shape (horizon-1, action_dim) or (horizon, action_dim)
            
        Returns:
            Total reward for the trajectory
        """
        if actions is None:
            raise ValueError("ActionBasedRewardFunction requires actions for trajectory evaluation")
        
        total_reward = 0.0
        for i in range(len(trajectory)):
            action = actions[i] if i < len(actions) else np.array([])
            next_state = trajectory[i+1] if i < len(trajectory)-1 else None
            total_reward += self.calculate_reward(
                state=trajectory[i],
                action=action,
                next_state=next_state
            )
        return total_reward