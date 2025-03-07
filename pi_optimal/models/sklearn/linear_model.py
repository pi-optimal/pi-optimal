from .base_sklearn_model import BaseSklearnModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearModel(BaseSklearnModel):
    def __init__(
        self,
    ):
        self.params = {}
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs, state_idx):
        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        if feature_type == "numerical":
            return LinearRegression(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return LogisticRegression(**self.params, class_weight="balanced")
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
