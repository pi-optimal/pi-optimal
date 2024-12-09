from tqdm import tqdm
import pickle
import numpy as np
from .base_model import BaseModel
from sklearn.neural_network import MLPRegressor, MLPClassifier
from torch.utils.data import DataLoader

class NeuralNetwork(BaseModel):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=None,
        verbose=0,
    ):
        self.params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "random_state": random_state,
            "verbose": verbose,
        }
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, feature_type):
        if feature_type == "numerical":
            return MLPRegressor(**self.params)
        elif feature_type in ["categorial", "binary"]:
            return MLPClassifier(**self.params)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def fit(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False, num_workers=0
        )
        past_states, past_actions, future_states, _ = next(iter(dataloader))
        X = self._prepare_input_data(past_states, past_actions)
        y = self._prepare_target_data(future_states)
        
        self.dataset_config = dataloader.dataset.dataset_config
        self.models = [
            self._create_estimator(self.dataset_config["states"][state_idx]["type"])
            for state_idx in self.dataset_config["states"]
        ]
        
        for i, model in enumerate(tqdm(self.models)):
            y_target = self._get_target_for_feature(y, i)
            model.fit(X, y_target)

