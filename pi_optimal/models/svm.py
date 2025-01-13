from .base_model import BaseModel
from sklearn.svm import SVR, SVC

class SupportVectorMachine(BaseModel):
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

    def _create_estimator(self, feature_type):
        if feature_type == "numerical":
            return SVR(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return SVC(**self.params, probability=True)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")