from .base_sklearn_model import BaseSklearnModel
from sklearn.svm import SVR, SVC

class SupportVectorMachine(BaseSklearnModel):
    def __init__(
        self,
        params: dict = {},
    ):
        """ Support Vector Machine class that uses the underlying sklearn SVR or SVC. Check the
            documentation of SVR and SVC for the available parameters 
            (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).
        """
        self.params = params
        self.use_past_states_for_reward = params.get("use_past_states_for_reward", True)
        self.params.pop("use_past_states_for_reward", None)
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