import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from sklearn import metrics
import torch


from..datasets.timeseries_dataset import TimeseriesDataset

class BaseEvaluator:
    def __init__(
        self,
        dataset_config: Dict[str, Dict[str, Any]],
        default_metrics: Dict[str, str] = {
            "numerical": "rmse",
            "binary": "accuracy",
            "categorical": "accuracy",
        },
        n_steps: int = 1
    ):
        self.dataset_config = dataset_config
        self.default_metrics = default_metrics
        self.validation_metrics = {
            "numerical": ["rmse", "mae", "r2"],
            "binary": ["accuracy", "f1", "precision", "recall"],
            "categorical": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        }
        self.n_steps = n_steps
        self._initialize_evaluation_metrics()

    def _initialize_evaluation_metrics(self) -> None:
        for item in self.dataset_config["states"].values():
            data_type = item["type"]
            if "evaluation_metric" not in item:
                item["evaluation_metric"] = self.default_metrics[data_type]
            elif item["evaluation_metric"] not in self.validation_metrics[data_type]:
                raise ValueError(
                    f"Unsupported metric for data type {data_type}: {item['evaluation_metric']}"
                )

    def evaluate(self, dataset: TimeseriesDataset, model: Any) -> Dict[str, List[Dict[str, Any]]]:
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        past_states, past_actions, future_states, future_actions = next(iter(dataloader))

        episode_mask = self._create_episode_mask(dataset)
        
        evaluations = []
        
        for step in range(self.n_steps):
            if step == 0:
                predicted_states = model(past_states, past_actions)
            else:
                predicted_states = model(predicted_states, future_actions[:, step-1:step])
            
            step_mask = self._create_step_mask(episode_mask, step)
            
            evaluation = self._evaluate_step(future_states[:, step], predicted_states, step_mask)
            evaluations.append(evaluation)

        return self._aggregate_evaluations(evaluations)

    def _create_episode_mask(self, dataset: 'TimeseriesDataset') -> torch.Tensor:
        mask = torch.ones(len(dataset), dtype=torch.bool)
        for start, end in zip(dataset.episode_start_index[1:], dataset.episode_end_index):
            mask[start:end] = False
        return mask

    def _create_step_mask(self, episode_mask: torch.Tensor, step: int) -> torch.Tensor:
        return episode_mask.roll(-step, dims=0)

    def _evaluate_step(self, true_states: torch.Tensor, predicted_states: torch.Tensor, mask: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        evaluation = {}
        for idx, item in self.dataset_config["states"].items():
            feature_slice = slice(item["feature_begin_idx"], item["feature_end_idx"])
            y_true = true_states[:, feature_slice][mask].numpy()
            y_pred = predicted_states[:, feature_slice][mask].detach().numpy()

            self._validate_shapes(y_true, y_pred, item["name"])

            metric_value = self._calculate_metric(y_true, y_pred, item["type"], item["evaluation_metric"])

            evaluation[idx] = {
                "name": item["name"],
                "type": item["type"],
                "metric": item["evaluation_metric"],
                "value": metric_value
            }
        return evaluation

    def _aggregate_evaluations(self, evaluations: List[Dict[str, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        aggregated = {}
        for idx in self.dataset_config["states"]:
            aggregated[idx] = [eval[idx] for eval in evaluations]
        return aggregated

    def _validate_shapes(self, y_true: np.ndarray, y_pred: np.ndarray, feature_name: str) -> None:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch for feature {feature_name}: true {y_true.shape}, pred {y_pred.shape}"
            )

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, data_type: str, metric: str) -> float:
        if data_type == "numerical":
            if metric == "rmse":
                return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
            elif metric == "mae":
                return metrics.mean_absolute_error(y_true, y_pred)
            elif metric == "r2":
                return metrics.r2_score(y_true, y_pred)
        elif data_type in ["binary", "categorical"]:
            y_pred_classes = np.round(y_pred).astype(int)
            if metric == "accuracy":
                return metrics.accuracy_score(y_true, y_pred_classes)
            elif metric == "f1":
                return metrics.f1_score(y_true, y_pred_classes, average='binary' if data_type == "binary" else 'macro')
            elif metric == "precision":
                return metrics.precision_score(y_true, y_pred_classes, average='binary' if data_type == "binary" else 'macro')
            elif metric == "recall":
                return metrics.recall_score(y_true, y_pred_classes, average='binary' if data_type == "binary" else 'macro')
        
        raise ValueError(f"Unsupported metric '{metric}' for data type '{data_type}'")