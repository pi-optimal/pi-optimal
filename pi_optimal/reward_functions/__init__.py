"""
Modular reward function system for pi_optimal.

This module provides a flexible architecture for defining reward functions
that can be used during planning without requiring model retraining.

Key components:
- BaseRewardFunction: Abstract base class for all reward functions
- StateBasedRewardFunction: For rewards that depend only on state
- ActionBasedRewardFunction: For rewards that depend on state and action
- Common implementations: Linear, target-based, threshold, custom, and composite rewards

Usage:
    from pi_optimal.reward_functions import LinearStateRewardFunction
    
    # Define a reward function
    reward_func = LinearStateRewardFunction(
        weights=[1.0, -0.5, 2.0],
        feature_indices=[0, 2, 5]
    )
    
    # Use in agent planning
    agent.predict(..., reward_function=reward_func)
"""

from .base_reward_function import (
    BaseRewardFunction,
    StateBasedRewardFunction,
    ActionBasedRewardFunction
)

from .common_reward_functions import (
    LinearStateRewardFunction,
    TargetStateRewardFunction,
    ThresholdRewardFunction,
    CustomFunctionRewardFunction,
    ActionPenaltyRewardFunction,
    CompositeRewardFunction
)

from .dataframe_reward_function import (
    DataFrameRewardFunction
)

__all__ = [
    "BaseRewardFunction",
    "StateBasedRewardFunction", 
    "ActionBasedRewardFunction",
    "LinearStateRewardFunction",
    "TargetStateRewardFunction",
    "ThresholdRewardFunction",
    "CustomFunctionRewardFunction",
    "ActionPenaltyRewardFunction",
    "CompositeRewardFunction",
    "DataFrameRewardFunction"
]