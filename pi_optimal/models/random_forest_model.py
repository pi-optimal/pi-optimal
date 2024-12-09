from tqdm import tqdm
import pickle
import numpy as np
from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from torch.utils.data import DataLoader


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

    def fit(self, dataset):

        self.dataset_config = dataset.dataset_config

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
