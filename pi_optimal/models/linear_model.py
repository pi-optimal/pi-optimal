from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearModel(BaseModel):
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
