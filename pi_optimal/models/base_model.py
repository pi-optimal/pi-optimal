# pi_optimal/models/base_model.py
from abc import ABC

class BaseModel(ABC):
    def __init__(self):
        pass
    
    def fit(self, dataset):
        """Fits the model to the dataset."""
        raise NotImplementedError

    def predict(self, X):
        """Predicts the next state given the current state and action which are 
           already in a ready to predict form X."""#
        raise NotImplementedError

    def forward(self, state, action):
        """Predicts the next state given the current state and action."""
        raise NotImplementedError
    
    def forward_with_reward_function(self, state, action, reward_function):
        """
        Predicts the next state and uses an external reward function instead of the model's reward prediction.
        
        Args:
            state: The current state
            action: The action to take
            reward_function: An external reward function to use instead of model's reward prediction
            
        Returns:
            The next state with reward calculated from the external reward function
        """
        # By default, predict without reward and then add external reward
        next_state = self.forward(state, action)
        
        # Replace the reward component in the predicted state with the external reward calculation
        if reward_function is not None:
            for i in range(len(state)):
                # Calculate reward based on the predicted next state
                reward = reward_function.calculate_state_reward(next_state[i])
                
                # Get the reward index from the reward function
                reward_idx = reward_function.reward_vector_idx
                
                # Replace the model's reward prediction with the calculated reward
                next_state[i, reward_idx] = reward
                
        return next_state
    
    def forward_n_steps(self, inital_state, actions, n_steps, backtransform=True):
        """Predicts the next n states given the initial state and sequence of actions."""
        raise NotImplementedError
        
    def save(self, filepath):
        """Secure model saving depending on the model type."""
        raise NotImplementedError

    @classmethod
    def load(cls, filepath):
        """Load the model from the given filepath."""
        raise NotImplementedError