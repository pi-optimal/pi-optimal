from .base_sklearn_model import BaseSklearnModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearModel(BaseSklearnModel):
    def __init__(
        self,
        params: dict = {},
    ):  
        """ Linear Model class that uses the underlying sklearn LinearRegression or LogisticRegression. 
            Check the documentation of LinearRegression and LogisticRegression for the available 
            parameters (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
        """
        self.params = params
        self.use_past_states_for_reward = params.get("use_past_states_for_reward", True)
        self.params.pop("use_past_states_for_reward", None)
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs, state_idx):
        """
        Create an estimator for the given state index and append it to self.models.
        """
        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        if feature_type == "numerical":
            return LinearRegression(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return LogisticRegression(**self.params, class_weight="balanced")
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
