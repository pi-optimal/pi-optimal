from .base_sklearn_model import BaseSklearnModel
from sklearn.svm import SVR, SVC

class SupportVectorMachine(BaseSklearnModel):
    def __init__(
        self,
        kernel='rbf',
        C=1.0,
        gamma='scale',
        tol=1e-3,
        max_iter=-1,
        verbose=0,
    ):
        self.params = {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "tol": tol,
            "max_iter": max_iter,
            "verbose": verbose,
        }
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs, state_idx):
        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        if feature_type == "numerical":
            return SVR(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return SVC(**self.params, probability=True)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")