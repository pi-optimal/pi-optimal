from typing import Dict, List, Any, Union
import numpy as np

from ..models.base_model import BaseModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

class BaseEvaluator:
    """
    A base class for evaluating machine learning models on different types of data.

    This class provides functionality to evaluate models on numerical, binary, and 
    categorical data using various metrics from scikit-learn.

    Attributes:
        dataset_config (Dict[str, Any]): Configuration of the dataset, including feature types and indices.
        default_metrics (Dict[str, str]): Default evaluation metrics for each data type.
        validation_metrics (Dict[str, List[str]]): Available evaluation metrics for each data type.
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        default_metrics: Dict[str, str] = {
            "numerical": "mse",
            "binary": "accuracy",
            "categorical": "f1_weighted",
        },
    ):
        """
        Initialize the BaseEvaluator.

        Args:
            dataset_config (Dict[str, Any]): Configuration of the dataset, including feature types and indices.
            default_metrics (Dict[str, str], optional): Default evaluation metrics for each data type. 
                                              Defaults to MSE for numerical, accuracy for binary, 
                                              and F1 weighted for categorical data.
        
        Raises:
            ValueError: If an unsupported metric is specified for a data type.
        """
        self.dataset_config: Dict[str, Any] = dataset_config
        self.default_metrics: Dict[str, str] = default_metrics
        self.validation_metrics: Dict[str, List[str]] = {
            "numerical": ["mse", "rmse", "mae"],
            "binary": ["accuracy", "f1_binary", "roc_auc"],
            "categorical": ["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
        }

        # Initialize evaluation metrics
        for idx, item in self.dataset_config["states"].items():
            data_type: str = item["type"]
            if "evaluation_metric" not in item:
                item["evaluation_metric"] = self.default_metrics[data_type]
            elif item["evaluation_metric"] not in self.validation_metrics[data_type]:
                raise ValueError(
                    f"Unsupported metric for data type {data_type}: {item['evaluation_metric']}"
                )
    
    def evaluate_one_step(self, dataset: Dataset, model: BaseModel, backtransform: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the model on the given dataset.

        Args:
            dataset (Dataset): The dataset to evaluate the model on.
            model (BaseModel): The model to evaluate.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing evaluation results for each feature.

        Raises:
            ValueError: If there's a shape mismatch between true and predicted values.
        """
        dataloader: DataLoader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        past_states, past_actions, next_states, _ = next(iter(dataloader))

        next_states: np.array = model._prepare_target_data(next_states)
        next_states_hat: np.array = model.forward(past_states, past_actions)

        if backtransform:
            next_states = dataset.inverse_transform_features("states", next_states)
            next_states_hat = dataset.inverse_transform_features("states", next_states_hat)

        evaluation = self._evaluate_next_state_prediction(next_states, next_states_hat)

        return evaluation

    def evaluate_episode(self, dataset: Dataset, model: BaseModel, episode_idx: int, initial_state_idx: int, n_rollout_steps: int, backtransform: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the model on a single episode from the dataset.

        Args:
            dataset (Dataset): The dataset to evaluate the model on.
            model (BaseModel): The model to evaluate.
            episode_idx (int): Index of the episode to evaluate.
            initial_state_idx (int): Index of the initial state in the episode.
            n_rollout_steps (int): Number of steps to rollout.
            backtransform (bool, optional): Whether to apply inverse transformation to states. Defaults to True.
        
        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results for each feature.
        """
        next_states, next_states_hat = self._rollout_episode_n_steps(dataset, model, episode_idx, initial_state_idx, n_rollout_steps, backtransform)
        return self._evaluate_next_state_prediction(next_states[0, :, 0, :], next_states_hat[0, :, 0, :])

    def _rollout_episode_n_steps(self, dataset: Dataset, model: BaseModel, episode_idx: int, initial_state_idx: int, n_rollout_steps: int, backtransform: bool = True) -> Union[np.ndarray, np.ndarray]:
        
        states, actions, next_states, next_action = dataset.get_episode(episode_idx)
        episode_length = dataset.episode_lengths[episode_idx]

        assert initial_state_idx < episode_length
        assert n_rollout_steps > 0
        assert initial_state_idx + n_rollout_steps <= episode_length

        inital_state = states[initial_state_idx]
        next_states = next_states[initial_state_idx:(initial_state_idx + n_rollout_steps)]
        actions = actions[initial_state_idx:(initial_state_idx + n_rollout_steps)]

        inital_state = np.expand_dims(inital_state, axis=0)
        next_states = np.expand_dims(next_states, axis=0)
        actions = np.expand_dims(actions, axis=0)

        next_states_hat = model.forward_n_steps(inital_state, actions, n_steps=n_rollout_steps)

        if backtransform:
            for step in range(n_rollout_steps):
                next_states[:,step,0,:] = dataset.inverse_transform_features("states", next_states[:,step,0,:])
                next_states_hat[:,step,0,:] = dataset.inverse_transform_features("states", next_states_hat[:,step,0,:])
            
        return next_states, next_states_hat

    def evaluate_dataset(self, dataset: Dataset, model: BaseModel, n_steps: int, backtransform: bool = True) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model predictions over multiple timesteps.

        Args:
            dataset (Dataset): The dataset containing states, actions, and next states.
            model (BaseModel): The model to evaluate.
            n_steps (int): Number of steps to evaluate.
            backtransform (bool, optional): Whether to apply inverse transformation to states. Defaults to True.

        Returns:
            Dict[int, Dict[str, float]]: Evaluation results for each step.
        """
        max_timesteps = dataset.max_episode_length
        next_states_per_rollout, next_states_hat_per_rollout = self._rollout_dataset_n_steps(dataset, model, n_steps, max_timesteps, backtransform)
        return self._evaluate_dataset_rollout(next_states_per_rollout, next_states_hat_per_rollout)
    
    def _evaluate_dataset_rollout(self, next_states_per_rollout: List[List[np.ndarray]], next_states_hat_per_rollout: List[List[np.ndarray]]) -> Dict[int, Dict[str, float]]:
        """
        Evaluate the results of a rollout.

        Args:
            next_states_per_rollout (List[List[np.ndarray]]): Actual next states for each rollout step.
            next_states_hat_per_rollout (List[List[np.ndarray]]): Predicted next states for each rollout step.

        Returns:
            Dict[int, Dict[str, float]]: Evaluation metrics for each rollout step.
        """
        assert len(next_states_per_rollout) == len(next_states_hat_per_rollout)
        n_steps = len(next_states_per_rollout)

        evaluations = {}
        for rollout_step in range(n_steps):
            # Concatenate and reshape results
            next_states = np.vstack(next_states_per_rollout[rollout_step])
            next_states_hat = np.vstack(next_states_hat_per_rollout[rollout_step])
            
            next_states = next_states.reshape(-1, 1, next_states.shape[-1])
            next_states_hat = next_states_hat.reshape(-1, 1, next_states_hat.shape[-1])

            # Evaluate predictions
            evaluation = self._evaluate_next_state_prediction(next_states[:, 0, :], next_states_hat[:, 0, :])
            evaluations[rollout_step] = evaluation

        return evaluations
        
    def _rollout_dataset_n_steps(self, dataset: Dataset, model: BaseModel, n_steps: int, max_timesteps: int, backtransform: bool = True):
        """
        Perform rollouts for the whole dataset.

        Iterates over each timestep in the dataset and rollout from this timestep for n steps.

        Args:
            dataset (Dataset): The dataset containing states, actions, and next states.
            model (BaseModel): The model to use for predictions.
            n_steps (int): Number of steps to roll out.
            max_timesteps (int): Maximum number of timesteps in the dataset.
            backtransform (bool, optional): Whether to apply inverse transformation to states. Defaults to True.

        Returns:
            Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]: Actual and predicted next states for each rollout step.
        """
        next_states_per_rollout = [[] for _ in range(n_steps)]
        next_states_hat_per_rollout = [[] for _ in range(n_steps)]

        # Iterate through all starting timesteps
        for initial_timestep in tqdm(range(0, max_timesteps + 1), desc="Performing rollouts"):
            existing_episodes = dataset.get_existing_episodes_at_timestep(i=initial_timestep)
            
            # Perform rollout for n steps
            for rollout_step in range(n_steps):
                current_timestep = initial_timestep + rollout_step
                
                # Get data for the current timestep
                if rollout_step == 0:
                    states, actions, next_states, _ = dataset.get_timestep_from_all_episodes(i=current_timestep)
                else:
                    _, actions, next_states, _ = dataset.get_timestep_from_all_episodes(i=current_timestep)
                    current_existing_episodes = dataset.get_existing_episodes_at_timestep(i=current_timestep)
                    
                    # Handle missing episodes
                    terminated_episodes = list(set(existing_episodes) - set(current_existing_episodes))
                    if terminated_episodes:
                        indices_to_remove = np.concatenate([np.where(episode == existing_episodes)[0] for episode in terminated_episodes])
                        states = np.delete(states, indices_to_remove, axis=0)
                        existing_episodes = current_existing_episodes

                # Skip if no data available
                if states.shape[0] == 0 or actions.shape[0] == 0:
                    continue

                # Forward pass through the model
                next_states_hat = model.forward(states, actions)

                # Update states for next iteration
                states = np.roll(states, -1, axis=1)
                states[:, -1] = next_states_hat

                # Apply backtransformation if required
                if backtransform:
                    next_states = dataset.inverse_transform_features("states", next_states[:, 0, :])
                    next_states_hat = dataset.inverse_transform_features("states", next_states_hat)

                # Store results
                next_states_per_rollout[rollout_step].append(next_states)
                next_states_hat_per_rollout[rollout_step].append(next_states_hat)
        return next_states_per_rollout, next_states_hat_per_rollout
    
    def _evaluate_next_state_prediction(self, 
                                   next_states: np.ndarray, 
                                   next_states_hat: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the model's predictions for each state feature.

        Args:
            next_states (np.ndarray): True future state values.
            next_states_hat (np.ndarray): Predicted future state values.
            backtransform (bool, optional): Whether to backtransform the predicted values. Defaults to True.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing evaluation results for each feature.

        Raises:
            ValueError: If there's a shape mismatch between true and predicted values.
        """

        evaluation: Dict[str, Dict[str, Any]] = {}
        for idx, item in self.dataset_config["states"].items():
            feature_begin_idx: int = item["feature_begin_idx"]
            feature_end_idx: int = item["feature_end_idx"]
            data_type: str = item["type"]
            evaluation_metric: str = item["evaluation_metric"]

            evaluation[idx] = {
                "name": item["name"],
                "type": data_type,
                "metric": evaluation_metric,
            }

            y_true: np.ndarray = next_states[:, feature_begin_idx:feature_end_idx]
            y_pred: np.ndarray = next_states_hat[:, feature_begin_idx:feature_end_idx]

            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f"Shape mismatch for feature {item['name']}: true {y_true.shape}, pred {y_pred.shape}"
                )

            value: float = self._calculate_metric(y_true, y_pred, data_type, evaluation_metric)
            evaluation[idx]["value"] = value

        return evaluation

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, data_type: str, metric: str) -> float:
        """
        Calculate the specified metric for the given true and predicted values using sklearn metrics.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            data_type (str): Type of the data ('numerical', 'binary', or 'categorical').
            metric (str): The metric to calculate.

        Returns:
            float: The calculated metric value.

        Raises:
            ValueError: If an unsupported metric is specified for the given data type.
        """
        if data_type == "numerical":
            if metric == "mse":
                return float(mean_squared_error(y_true, y_pred))
            elif metric == "rmse":
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric == "mae":
                return float(mean_absolute_error(y_true, y_pred))
        elif data_type in ["binary", "categorical"]:
            y_pred_classes = np.round(y_pred).astype(int)
            if metric == "accuracy":
                return float(accuracy_score(y_true, y_pred_classes))
            elif metric.startswith("f1_"):
                if metric == "f1_binary" and data_type == "binary":
                    return float(f1_score(y_true, y_pred_classes, average='binary', zero_division=0))
                elif metric == "f1_weighted":
                    return float(f1_score(y_true, y_pred_classes, average='weighted', zero_division=0))
                elif metric == "f1_macro":
                    return float(f1_score(y_true, y_pred_classes, average='macro', zero_division=0))
                elif metric == "f1_micro":
                    return float(f1_score(y_true, y_pred_classes, average='micro', zero_division=0))
            elif metric == "roc_auc" and data_type == "binary":
                return float(roc_auc_score(y_true, y_pred))

        raise ValueError(f"Unsupported metric for data type {data_type}: {metric}")