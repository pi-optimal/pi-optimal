from tqdm import tqdm
import pickle
import numpy as np
from .base_model import BaseModel
from sklearn.svm import SVR, SVC
from torch.utils.data import DataLoader

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