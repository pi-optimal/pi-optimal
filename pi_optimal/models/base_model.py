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