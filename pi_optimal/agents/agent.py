from pi_optimal.datasets.base_dataset import BaseDataset
from pi_optimal.planners.cem_discrete import CEMDiscretePlanner
from pi_optimal.planners.cem_continuous import CEMContinuousPlanner
from pi_optimal.models.random_forest_model import RandomForest
from pi_optimal.models.svm import SupportVectorMachine
from pi_optimal.models.mlp import NeuralNetwork
from pi_optimal.utils.logger import Logger
from torch.utils.data import Subset
import numpy as np

class Agent():
    def __init__(self, dataset: BaseDataset, type: str, constraints: dict = None):
        self.dataset = dataset
        self.type = type

        self.hash_id = np.random.randint(0, 100000)
        self.logger = Logger(f"Agent-{self.hash_id}")

        self.logger.info(f"Creating agent of type {type}", "PROCESS")

        if type == "mpc-discrete":
            self.policy = CEMDiscretePlanner(action_dim=dataset.actions.shape[1])
        elif type == "mpc-continuous":
            constraints = self._init_constrains(constraints)
            self.policy = CEMContinuousPlanner(action_dim=dataset.actions.shape[1],
                                                constraints=constraints)
        else:
            raise NotImplementedError
        
        self.logger.info(f"Agent of type {type} created.", "SUCCESS")

    
    def _init_constrains(self, constraints):
        
        min_values = []
        max_values = []
        for action_key in self.dataset.dataset_config["actions"]:
            action = self.dataset.dataset_config["actions"][action_key]
            action_name = action["name"]

            if constraints is None:
                action_min, action_max = self.dataset.df[action_name].min(), self.dataset.df[action_name].max()
            else:
                action_min, action_max = constraints["min"][action_key], constraints["max"][action_key]

            transformed_min, transformed_max = action["processor"].transform([[action_min], [action_max]])
            min_values.append(transformed_min[0])
            max_values.append(transformed_max[0])
    
        constraints = {"min": np.array(min_values), "max": np.array(max_values)}

        return constraints

    def train(self):
        self.logger_training = Logger(f"Agent-Training-{self.hash_id}")
        self.logger_training.info(f"Training agent of type {self.type}", "PROCESS")
        if self.type == "mpc-discrete" or self.type == "mpc-continuous":

            self.models = []
            model_1 = RandomForest(n_estimators=100, 
                                  max_depth=None, 
                                  n_jobs=-1,
                                  verbose=0,
                                  random_state=0)
            model_1 = NeuralNetwork()
            self.models.append(model_1)

            model_2 = RandomForest(n_estimators=100, 
                                  max_depth=None, 
                                  n_jobs=-1,
                                  verbose=0,
                                  random_state=1)
            model_2 = NeuralNetwork()
            self.models.append(model_2)

            n_models = len(self.models)

            # Split the dataset into n_models
            len_dataset = len(self.dataset)
            subset_size = len_dataset // n_models  # integer division

            for i in range(n_models):
                # Compute start and end indices for this model's subset
                start_idx = i * subset_size
                # For the last model, make sure we include all remaining data
                end_idx = (i + 1) * subset_size if i < n_models - 1 else len_dataset
                
                # Create a Subset of the dataset
                current_subset = Subset(self.dataset, range(start_idx, end_idx))
                current_subset.dataset_config = self.dataset.dataset_config
                # Fit the model on this subset
                self.models[i].fit(current_subset)

        self.logger_training.info(f"The agent of type {self.type} has been trained.", "SUCCESS")

    def objective_function(self, traj):
        reward_idx = self.dataset.reward_column_idx
        return -sum(traj[:, reward_idx])       

    def predict(self, 
                dataset: BaseDataset, 
                inverse_transform: bool = True, 
                n_iter: int = 10,
                horizon: int = 4,
                population_size: int = 1000,
                topk: int = 100,
                uncertainty_weight: float = 0.5,
                reset_planer: bool = True,
                allow_sigma: bool = False):
        self.logger_inference = Logger(f"Agent-Inference-{self.hash_id}")
        self.logger_inference.info(f"Searching for the optimal action sequence over a horizon of {horizon} steps.", "PROCESS")

        if self.type == "mpc-discrete" or self.type == "mpc-continuous":
            last_state, last_action, _, _ = dataset[len(dataset) - 1]

            actions = self.policy.plan(
                models=self.models,                  
                starting_state=last_state,
                action_history=last_action,
                objective_function=self.objective_function,
                n_iter=n_iter,
                horizon=horizon,
                population_size=population_size,
                uncertainty_weight=uncertainty_weight,
                reset_planer=reset_planer,
                allow_sigma=allow_sigma)
            
            self.logger_inference.info(f"Optimal action sequence found.", "SUCCESS")

            transformed_actions = []
            if inverse_transform:
                for action_idx in dataset.dataset_config["actions"]:
                    action_config = dataset.dataset_config["actions"][action_idx]
                    if action_config["type"] == "categorial":
                        transformed_actions.append(action_config["processor"].inverse_transform(actions[:, action_idx].round().astype(int).reshape(-1,1)).reshape(1, -1))
                    else:
                        transformed_actions.append(action_config["processor"].inverse_transform([actions[:, action_idx]]))
                return np.array(transformed_actions)[: ,0].T
            
            return actions
        else:
            self.logger_inference.error(f"Agent of type {self.type} not implemented.", "ERROR")
            raise NotImplementedError
