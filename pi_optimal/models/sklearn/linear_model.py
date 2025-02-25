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

    def _create_estimator(self, feature_type):
        if feature_type == "numerical":
            return LinearRegression(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return LogisticRegression(**self.params, class_weight="balanced")
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
