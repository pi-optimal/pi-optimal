from .base_sklearn_model import BaseSklearnModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from typing import Dict, Any

class HybridModel(BaseSklearnModel):
    def __init__(self, params: dict):
        if not HybridModel._validate_hybrid_params(params):
            raise ValueError("Invalid configuration for hybrid model.")
        
        self.params = params
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs: Dict[int, dict], state_idx: int) -> Any:
        """
        Create an estimator for the given state index and append it to self.models.
        Returns the created estimator.
        """
        if state_idx not in state_configs:
            raise KeyError(f"State index {state_idx} not found in state_configs.")

        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        model_config = self.params[state_idx]
        model_type = model_config["type"]
        model_params = model_config["params"]

        # Mapping feature type and model type to corresponding estimator classes
        if feature_type == "numerical":
            estimators = {
                "mlp": lambda p: MLPRegressor(**p),
                "rf": lambda p: RandomForestRegressor(**p),
                "svm": lambda p: SVR(**p),
                "linear": lambda p: LinearRegression(**p),
            }
        elif feature_type in ["categorial", "binary"]:
            estimators = {
                "mlp": lambda p: MLPClassifier(**p),
                "rf": lambda p: RandomForestClassifier(**p, class_weight="balanced"),
                "svm": lambda p: SVC(**p, probability=True),
                "linear": lambda p: LogisticRegression(**p, class_weight="balanced"),
            }
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        if model_type not in estimators:
            raise ValueError(f"Unsupported model type '{model_type}' for state index {state_idx}.")

        estimator = estimators[model_type](model_params)
        return estimator

    @staticmethod
    def _validate_hybrid_params(params: dict) -> bool:
        """
        Validate the format of the hybrid model configuration.

        The expected format is:
        
        {
            state_idx (int): {
                "type": <model_type: str>,  # allowed values: "mlp", "rf", "svm", "linear"
                "params": <dict of model parameters>
            },
            ...
        }
        
        Raises:
            ValueError: If the configuration does not follow the expected format.
        
        Returns:
            bool: True if the configuration is valid.
        """
        if not isinstance(params, dict):
            raise ValueError("The configuration must be a dictionary.")

        allowed_types = {"mlp", "rf", "svm", "linear"}

        for key, value in params.items():
            if not isinstance(key, int):
                raise ValueError("All keys in the configuration must be integers representing state indices.")
            if not isinstance(value, dict):
                raise ValueError(f"Configuration for state index {key} must be a dictionary.")
            if "type" not in value:
                raise ValueError(f"Missing 'type' key for state index {key}.")
            if "params" not in value:
                raise ValueError(f"Missing 'params' key for state index {key}.")
            if not isinstance(value["type"], str):
                raise ValueError(f"'type' for state index {key} must be a string.")
            if value["type"] not in allowed_types:
                raise ValueError(
                    f"Unsupported model type '{value['type']}' for state index {key}. "
                    f"Allowed types are: {', '.join(allowed_types)}."
                )
            if not isinstance(value["params"], dict):
                raise ValueError(f"'params' for state index {key} must be a dictionary.")
        
        return True
