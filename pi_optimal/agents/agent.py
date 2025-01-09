from pi_optimal.datasets.base_dataset import BaseDataset
from pi_optimal.planners.cem_discrete import CEMDiscretePlanner
from pi_optimal.planners.cem_continuous import CEMContinuousPlanner
from pi_optimal.models.random_forest_model import RandomForest
from pi_optimal.models.svm import SupportVectorMachine
from pi_optimal.models.mlp import NeuralNetwork
import numpy as np

class Agent():
    def __init__(self, dataset: BaseDataset, type: str, config: dict = None):
        self.dataset = dataset
        self.type = type
        self.config = self._init_config(config)

        if type == "mpc-discrete":
            self.policy = CEMDiscretePlanner(action_dim=dataset.actions.shape[1],
                                            horizon=self.config["horizon"],
                                            population_size=self.config["population_size"],
                                            topk=self.config["topk"],
                                            uncertainty_weight=self.config["uncertainty_weight"],
                                            )
        elif type == "mpc-continuous":
            self.policy = CEMContinuousPlanner(action_dim=dataset.actions.shape[1],
                                            horizon=self.config["horizon"],
                                            population_size=self.config["population_size"],
                                            topk=self.config["topk"],
                                            uncertainty_weight=self.config["uncertainty_weight"],
                                            constraints=self.config["constraints"])
        else:
            raise NotImplementedError
    
    def _init_config(self, config):
        self.default_params = {"horizon": 4,
                        "population_size": 1000,
                        "topk": 100,
                        "uncertainty_weight": 0.1,
                        "constraints": None,
                        }
        config_with_defaults = self.default_params.copy()
        config_with_defaults.update(config or {})

        
        min_values = []
        max_values = []
        for action_key in self.dataset.dataset_config["actions"]:
            action = self.dataset.dataset_config["actions"][action_key]
            action_name = action["name"]

            if "constraints" not in config_with_defaults or config_with_defaults["constraints"] is None:
                action_min, action_max = self.dataset.df[action_name].min(), self.dataset.df[action_name].max()
            else:
                action_min, action_max = config_with_defaults["constraints"]["min"][action_key], config_with_defaults["constraints"]["max"][action_key]

            transformed_min, transformed_max = action["processor"].transform([[action_min], [action_max]])
            min_values.append(transformed_min[0])
            max_values.append(transformed_max[0])
    
        config_with_defaults["constraints"] = {"min": np.array(min_values), "max": np.array(max_values)}

        return config_with_defaults

    def train(self):
        if self.type == "mpc-discrete" or self.type == "mpc-continuous":

            self.models = []
            rf_reg = RandomForest(n_estimators=100, 
                                  max_depth=None, 
                                  n_jobs=-1,
                                  verbose=0)
            rf_reg.fit(self.dataset)
            self.models.append(rf_reg)

            svm_reg = SupportVectorMachine(kernel="rbf",
                                 C=4,
                                 gamma="scale",
                                 verbose=False)
            svm_reg.fit(self.dataset)
            self.models.append(svm_reg)

    def objective_function(self, traj):
        reward_idx = self.dataset.reward_column_idx
        return -sum(traj[:, reward_idx])       

    def predict(self, dataset: BaseDataset, inverse_transform=True, n_iter=10):
        if self.type == "mpc-discrete" or self.type == "mpc-continuous":
            last_state, last_action, _, _ = dataset[len(dataset) - 1]
            actions = self.policy.plan(
                models=self.models,                  
                starting_state=last_state,
                action_history=last_action,
                objective_function=self.objective_function,
                n_iter=n_iter,
                allow_sigma=False,)
            transformed_actions = []
            if inverse_transform:
                for action_idx in dataset.dataset_config["actions"]:
                    action_config = dataset.dataset_config["actions"][action_idx]
                    if action_config["type"] == "categorial":
                        transformed_actions.append(action_config["processor"].inverse_transform(actions[:, action_idx].astype(int).reshape(-1,1)).reshape(1, -1))
                    else:
                        transformed_actions.append(action_config["processor"].inverse_transform([actions[:, action_idx]]))
                return np.array(transformed_actions)[: ,0].T
            return actions
