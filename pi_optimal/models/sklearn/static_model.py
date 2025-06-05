from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
from .base_sklearn_model import BaseSklearnModel


class StaticRegressor(BaseEstimator, RegressorMixin):
    """A regressor that returns the static value from the input X at the specified feature index."""
    
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx
    
    def fit(self, X, y):
        """No fitting needed for static regressor."""
        # X, y are required by sklearn interface but not used for static prediction
        return self
    
    def predict(self, X):
        """Return the static feature value from the input for all samples."""
        if len(X) == 0:
            return np.array([])
        # Extract the static feature value from the input
        return X[:, self.feature_idx]


class StaticClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that returns the static value from the input X at the specified feature index."""
    
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx
        self.classes_ = None
    
    def fit(self, X, y):
        """Determine unique classes from y for predict_proba."""
        # X is required by sklearn interface but not used for static prediction
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """Return the static feature value from the input for all samples."""
        if len(X) == 0:
            return np.array([], dtype=int)
        # Extract the static feature value from the input
        return X[:, self.feature_idx].astype(int)
    
    def predict_proba(self, X):
        """Return probability of 1.0 for the input class and 0.0 for others."""
        if len(X) == 0 or self.classes_ is None:
            return np.zeros((0, len(self.classes_) if self.classes_ is not None else 1))
        
        input_values = X[:, self.feature_idx]
        probas = np.zeros((len(X), len(self.classes_)))
        
        for i, val in enumerate(input_values):
            class_idx = np.where(self.classes_ == val)[0]
            if len(class_idx) > 0:
                probas[i, class_idx[0]] = 1.0
        
        return probas


class StaticModel(BaseSklearnModel):
    def __init__(
        self,
        params: dict = {},
    ):
        """Static Model class that predicts the last seen value for each feature.
        
        Args:
            params (dict): Configuration parameters.
                use_past_states_for_reward (bool): Whether to use past states for reward prediction.
        """
        self.params = params
        self.use_past_states_for_reward = params.get("use_past_states_for_reward", True)
        self.params.pop("use_past_states_for_reward", None)
        
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs, state_idx):
        """
        Create a static estimator for the given state index and append it to self.models.
        
        Args:
            state_configs: Configuration for different states/features
            state_idx: Index of the state/feature to create an estimator for
        
        Returns:
            An appropriate static estimator (regressor or classifier)
        """
        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        
        # Calculate the correct input index for static models
        # Static models need to extract from the most recent timestep in the flattened past_states
        # feature_begin_idx is for output indexing, but input X has structure [flatten_past_states, flatten_past_actions]
        # where flatten_past_states has shape (batch_size, lookback_timesteps * n_state_features)
        # and is organized as [t0_f0, t0_f1, ..., t0_fn, t1_f0, t1_f1, ..., t1_fn, ..., t(L-1)_f0, ..., t(L-1)_fn]
        feature_begin_idx = state_config["feature_begin_idx"]
        lookback_timesteps = self.dataset_config["lookback_timesteps"]
        n_state_features = self.dataset_config["states_size"]
        
        # For static models: extract from the most recent timestep (last timestep) for this feature
        # Input index = (lookback_timesteps - 1) * n_state_features + feature_begin_idx
        correct_input_idx = (lookback_timesteps - 1) * n_state_features + feature_begin_idx
        
        if feature_type == "numerical":
            return StaticRegressor(correct_input_idx)
        elif feature_type in ["categorial", "binary"]:
            return StaticClassifier(correct_input_idx)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
