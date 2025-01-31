from pi_optimal.datasets.base_dataset import BaseDataset
from pi_optimal.planners.cem_discrete import CEMDiscretePlanner
from pi_optimal.planners.cem_continuous import CEMContinuousPlanner
from pi_optimal.models.random_forest_model import RandomForest
from pi_optimal.models.svm import SupportVectorMachine
from pi_optimal.models.mlp import NeuralNetwork
from pi_optimal.utils.serialization import (
    serialize_processors,
    deserialize_processors,
    serialize_policy_dict,
    deserialize_policy_dict,
    NumpyEncoder
)
from pi_optimal.utils.validation import validate_agent_directory, validate_path
from pi_optimal.utils.logger import Logger
from torch.utils.data import Subset
import numpy as np
import json
import os
import glob
import datetime

class Agent():
    def __init__(self, name: str = "pi_optimal_agent"):                
        self.name = name
        self.status = "Initialized"

        self.hash_id = np.random.randint(0, 100000)
        self.logger = Logger(f"Agent-{self.hash_id}")
        self.logger.info(f"Agent of type {type} initialized.", "SUCCESS")
        
    def _init_constrains(self, dataset, constraints):
        
        min_values = []
        max_values = []
        for action_key in dataset.dataset_config["actions"]:
            action = dataset.dataset_config["actions"][action_key]
            action_name = action["name"]

            if constraints is None:
                action_min, action_max = dataset.df[action_name].min(), dataset.df[action_name].max()
            else:
                action_min, action_max = constraints["min"][action_key], constraints["max"][action_key]

            transformed_min, transformed_max = action["processor"].transform([[action_min], [action_max]])
            min_values.append(transformed_min[0])
            max_values.append(transformed_max[0])
    
        constraints = {"min": np.array(min_values), "max": np.array(max_values)}

        return constraints

    def train(self, dataset: BaseDataset, constraints: dict = None):
        
        self.type = dataset.action_type

        self.logger_training = Logger(f"Agent-Training-{self.hash_id}-{np.random.randint(0, 100000)}")
        self.logger_training.info(f"Training agent of type {self.type}", "PROCESS")

        if self.type == "mpc-discrete":
            self.policy = CEMDiscretePlanner(action_dim=dataset.actions.shape[1])
        elif self.type == "mpc-continuous":
            constraints = self._init_constrains(dataset, constraints)
            self.policy = CEMContinuousPlanner(action_dim=dataset.actions.shape[1],
                                                constraints=constraints)
        else:
            self.logger.error(f"Agent type {self.type} not supported.")
            raise NotImplementedError
        
        self.dataset_config = dataset.dataset_config
        self.models = []
        # rf_reg = RandomForest(n_estimators=100, 
        #                         max_depth=None, 
        #                         n_jobs=-1,
        #                         verbose=0,
        #                         random_state=0)
        rf_reg = NeuralNetwork()    
        self.models.append(rf_reg)

        # rf_reg = RandomForest(n_estimators=100, 
        #                         max_depth=None, 
        #                         n_jobs=-1,
        #                         verbose=0,
        #                         random_state=1)
        rf_reg = NeuralNetwork()
        self.models.append(rf_reg)

        n_models = len(self.models)

        # Split the dataset into n_models
        len_dataset = len(dataset)
        subset_size = len_dataset // n_models  # integer division

        for i in range(n_models):
            # Compute start and end indices for this model's subset
            start_idx = i * subset_size
            # For the last model, make sure we include all remaining data
            end_idx = (i + 1) * subset_size if i < n_models - 1 else len_dataset
            
            # Create a Subset of the dataset
            current_subset = Subset(dataset, range(start_idx, end_idx))
            current_subset.dataset_config = self.dataset_config
            # Fit the model on this subset
            self.models[i].fit(current_subset)

        self.status = "Trained"
        self.logger_training.info(f"The agent of type {self.type} has been trained.", "SUCCESS")


    def objective_function(self, traj):
        reward_idx = self.dataset_config['reward_vector_idx']
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
        self.logger_inference = Logger(f"Agent-Inference-{self.hash_id}-{np.random.randint(0, 100000)}")
        self.logger_inference.info(f"Searching for the optimal action sequence over a horizon of {horizon} steps.", "PROCESS")
        self.policy.logger = self.logger_inference
        
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

    def save(self, path = 'agents/'):
        """Save the agent configuration and models."""
        validate_path(path)
        if self.status != "Trained":
            self.logger.error("Agent must be trained before saving.")
            raise Exception("Agent must be trained before saving.")
        
        # Check if the directory exists
        agent_path = f"{path}/{self.name}"
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
            os.makedirs(f"{agent_path}/models")

        # Save agent configuration
        config = {
            'name': self.name,
            'type': self.type,
            'status': self.status,            
            'version': '0.1',
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_config': serialize_processors(self.dataset_config.copy(), agent_path)
        }
        
        with open(f"{agent_path}/agent_config.json", "w") as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)

        # Save policy if exists
        if hasattr(self, 'policy'):
            with open(f"{agent_path}/policy_config.json", "w") as f:
                policy_config = {
                    'type': self.policy.__class__.__name__,
                    'params': serialize_policy_dict(self.policy.__dict__)
                }
                json.dump(policy_config, f, indent=4, cls=NumpyEncoder)

        # Save models if they exist
        if hasattr(self, 'models') and self.models:
            for i, model in enumerate(self.models):
                model.save(f"{agent_path}/models/model_{i}.joblib")

    @classmethod 
    def load(cls, path: str):
        """Load an agent from saved configuration."""

        validate_agent_directory(path)

        with open(f"{path}/agent_config.json", "r") as f:
            config = json.load(f)

        # Create agent instance
        agent = cls(
            name=config['name']
        )
        agent.status = config['status']
        # Restore dataset configuration with deserialized processors
        agent.dataset_config = deserialize_processors(config['dataset_config'], path)

        # Load policy if exists
        if os.path.exists(f"{path}/policy_config.json"):
            with open(f"{path}/policy_config.json", "r") as f:
                policy_config = json.load(f)
                # Reconstruct policy based on type
                if policy_config['type'] == "CEMDiscretePlanner":
                    agent.policy = CEMDiscretePlanner(action_dim=policy_config['params']['action_dim'])
                    agent.type = "mpc-discrete"
                elif policy_config['type'] == "CEMContinuousPlanner":
                    agent.policy = CEMContinuousPlanner(action_dim=policy_config, constraints=policy_config['params']['constraints'])
                    agent.type = "mpc-continuous"
                # Restore policy parameters
                for key, value in deserialize_policy_dict(policy_config['params']).items():
                    setattr(agent.policy, key, value)

        # Load models if they exist
        model_files = glob.glob(f"{path}/models/model_*.joblib")
        if model_files:
            agent.models = []
            for model_path in sorted(model_files):
                model = NeuralNetwork.load(model_path) 
                agent.models.append(model)

        return agent
    

    def _validate_models(self):
        """Validate that all required models are loaded correctly."""
        if not self.models:
            self.logger.error("No models found in agent")
            raise ValueError("No models found in agent")
        for model in self.models:
            if not isinstance(model, (RandomForest, SupportVectorMachine, NeuralNetwork)):
                self.logger.error(f"Invalid model type: {type(model)}")
                raise TypeError(f"Invalid model type: {type(model)}")
