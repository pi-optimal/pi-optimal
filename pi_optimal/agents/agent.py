# pi_optimal/agents/agent.py

from pi_optimal.datasets.base_dataset import BaseDataset
from pi_optimal.planners.cem_discrete import CEMDiscretePlanner
from pi_optimal.planners.cem_continuous import CEMContinuousPlanner
from pi_optimal.models.sklearn.random_forest_model import RandomForest
from pi_optimal.models.sklearn.svm import SupportVectorMachine
from pi_optimal.models.sklearn.mlp import NeuralNetwork
from pi_optimal.models.sklearn.hybrid_model import HybridModel
from pi_optimal.models.sklearn.static_model import StaticModel

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
from typing import List, Dict


class Agent():
    # The MODEL_REGISTRY now includes both scikit-learn models and the new PyTorch ones.
    MODEL_REGISTRY = {
        "NeuralNetwork": NeuralNetwork,
        "SupportVectorMachine": SupportVectorMachine, 
        "RandomForest": RandomForest,
        "HybridModel": HybridModel,
        "StaticModel": StaticModel
    }

    def __init__(self, name: str = "pi_optimal_agent"):                
        self.name = name
        self.status = "Initialized"

        self.hash_id = np.random.randint(0, 100000)
        self.logger = Logger(f"Agent-{self.hash_id}")
        self.logger.info(f"Agent of type {type} initialized with name '{self.name}'.", "SUCCESS")        

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

    def train(self, dataset: BaseDataset, constraints: dict = None, model_config: List[Dict] = None):
        self.type = dataset.action_type

        self.logger_training = Logger(f"Agent-Training-{self.hash_id}-{np.random.randint(0, 100000)}")
        self.logger_training.info(f"Training agent '{self.name}' of type {self.type}", "PROCESS")

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

        if model_config is None:
            # Default configuration: by default, using two scikit-learn NeuralNetwork models.
            model_config = [
                {"model_type": "NeuralNetwork", "params": {}},
                {"model_type": "NeuralNetwork", "params": {}}
            ]

        self._validate_models(model_config)

        self.models = []
        for config in model_config:
            model_cls = self.MODEL_REGISTRY[config["model_type"]]
            model = model_cls(params=config.get("params", {}))
            self.models.append(model)

        # Split the dataset into n_models
        n_models = len(self.models)
        len_dataset = len(dataset)
        subset_size = len_dataset // n_models  # integer division

        for i in range(n_models):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if i < n_models - 1 else len_dataset
            current_subset = Subset(dataset, range(start_idx, end_idx))
            current_subset.dataset_config = self.dataset_config
            self.models[i].fit(current_subset)

        self.status = "Trained"
        self.logger_training.info(f"The agent '{self.name}' of type {self.type} has been trained.", "SUCCESS")

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
                allow_sigma: bool = False,
                reward_function=None):
        self.logger_inference = Logger(f"Agent-Inference-{self.hash_id}-{np.random.randint(0, 100000)}")
        self.logger_inference.info(f"Searching for the optimal action sequence over a horizon of {horizon} steps.", "PROCESS")
        self.policy.logger = self.logger_inference
        
        # Choose objective function based on whether reward_function is provided
        if reward_function is not None:
            # Fit the reward function if needed
            if hasattr(reward_function, 'fit') and not reward_function.is_fitted:
                reward_function.fit(dataset_config=self.dataset_config)
            
            # Create objective function that uses external reward function
            def reward_function_objective(traj):
                return -reward_function.calculate_trajectory_reward(traj)
            
            objective_func = reward_function_objective
            self.logger_inference.info("Using external reward function for planning.", "INFO")
        else:
            objective_func = self.objective_function
            self.logger_inference.info("Using model-predicted rewards for planning.", "INFO")
        
        if self.type == "mpc-discrete" or self.type == "mpc-continuous":
            last_state, last_action, _, _ = dataset[len(dataset) - 1]

            # Set reward function as an attribute on the policy object
            if reward_function is not None:
                self.policy.reward_function = reward_function
            
            # Call plan without passing reward_function parameter
            actions = self.policy.plan(
                models=self.models,                  
                starting_state=last_state,
                action_history=last_action,
                objective_function=objective_func,
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
                        transformed_actions.append(
                            action_config["processor"]
                            .inverse_transform(actions[:, action_idx].round().astype(int).reshape(-1, 1))
                            .reshape(1, -1)
                        )
                    else:
                        transformed_actions.append(
                            action_config["processor"].inverse_transform([actions[:, action_idx]])
                        )
                return np.array(transformed_actions)[:, 0].T
            
            return actions

    def save(self, path='agents/'):
        """Save the agent configuration and models."""
        
        if self.status != "Trained":
            self.logger.error("Agent must be trained before saving.")
            raise Exception("Agent must be trained before saving.")
        
        agent_path = path + '/' + self.name
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        
        models_dir = f"{agent_path}/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        models_config = []
        if hasattr(self, 'models') and self.models:
            for i, model in enumerate(self.models):
                model_filename = f"model_{i}.pkl"
                model_path = f"{models_dir}/{model_filename}" 
                model.save(model_path)
                models_config.append({
                    "model_type": model.__class__.__name__,
                    "model_filename": model_filename
                })
        
        config = {
            'name': self.name, 
            'type': self.type,
            'status': self.status,            
            'version': '0.1',
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_config': serialize_processors(self.dataset_config.copy(), agent_path), 
            'models_config': models_config
        }
        
        # Save config file directly in agent_path
        with open(f"{agent_path}/agent_config.json", "w") as f: 
            json.dump(config, f, indent=4, cls=NumpyEncoder)

        if hasattr(self, 'policy'):
             # Save policy config file directly in agent_path
            with open(f"{agent_path}/policy_config.json", "w") as f:
                policy_config = {
                    'type': self.policy.__class__.__name__,
                    'params': serialize_policy_dict(self.policy.__dict__)
                }
                json.dump(policy_config, f, indent=4, cls=NumpyEncoder)

    @classmethod 
    def load(cls, path: str):
        """Load an agent from saved configuration."""
        validate_agent_directory(path)
        with open(f"{path}/agent_config.json", "r") as f:
            config = json.load(f)

        agent = cls(name=config['name'])
        agent.status = config['status']
        agent.dataset_config = deserialize_processors(config['dataset_config'], path)

        if os.path.exists(f"{path}/policy_config.json"):
            with open(f"{path}/policy_config.json", "r") as f:
                policy_config = json.load(f)
                if policy_config['type'] == "CEMDiscretePlanner":
                    agent.policy = CEMDiscretePlanner(action_dim=policy_config['params']['action_dim'])
                    agent.type = "mpc-discrete"
                elif policy_config['type'] == "CEMContinuousPlanner":
                    agent.policy = CEMContinuousPlanner(action_dim=policy_config['params']['action_dim'],
                                                        constraints=policy_config['params']['constraints'])
                    agent.type = "mpc-continuous"
                for key, value in deserialize_policy_dict(policy_config['params']).items():
                    setattr(agent.policy, key, value)

        agent.models = []
        models_config = config.get('models_config', [])
        for model_entry in models_config:
            model_type = model_entry['model_type']
            model_filename = model_entry['model_filename']
            model_path = f"{path}/models/{model_filename}"
            if model_type not in cls.MODEL_REGISTRY:
                raise ValueError(f"Unknown model type '{model_type}' found in saved configuration.")
            model_cls = cls.MODEL_REGISTRY[model_type]
            model = model_cls.load(model_path)
            agent.models.append(model)

        return agent

    def _validate_models(self, model_config: list) -> None:
        """Validate model configuration structure and parameters."""
        required_keys = {"model_type", "params"}
        for i, config in enumerate(model_config):
            missing_keys = required_keys - config.keys()
            if missing_keys:
                self.logger.error(f"Model config #{i+1} missing required keys: {missing_keys}")
                raise ValueError(
                    f"Model config #{i+1} missing required keys: {missing_keys}"
                )
            model_type = config["model_type"]
            if model_type not in self.MODEL_REGISTRY:
                available_models = list(self.MODEL_REGISTRY.keys())
                self.logger.error(f"Invalid model type '{model_type}' in config #{i+1}.")
                raise ValueError(
                    f"Invalid model type '{model_type}' in config #{i+1}. "
                    f"Available models: {available_models}"
                )
            if not isinstance(config["params"], dict):
                self.logger.error(
                    f"Parameters for model #{i+1} must be a dictionary, got {type(config['params']).__name__}"
                )
                raise TypeError(
                    f"Parameters for model #{i+1} must be a dictionary, got {type(config['params']).__name__}"
                )
