from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class RandomForest(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
        }
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, feature_type):
        if feature_type == "numerical":
            return RandomForestRegressor(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return RandomForestClassifier(**self.params, class_weight="balanced")
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
