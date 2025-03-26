from .base_sklearn_model import BaseSklearnModel
from sklearn.neural_network import MLPRegressor, MLPClassifier


class NeuralNetwork(BaseSklearnModel):
    def __init__(
        self,
        params: dict = {},
    ):
        """ Neural Network class that uses the underlying sklearn MLPRegressor or MLPClassifier. Check the
            documentation of MLPRegressor and MLPClassifier for the available parameters 
            (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
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
            return MLPRegressor(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return MLPClassifier(**self.params)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    