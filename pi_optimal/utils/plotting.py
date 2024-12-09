import matplotlib.pyplot as plt

from typing import Dict, Any
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def plot_n_step_episode_rollout(next_states: np.ndarray, next_states_hat: np.ndarray, dataset: Any) -> None:
    """
    Plot n-step episode rollout of a specific episode.

    This function creates a separate plot for each state in the dataset,
    showing the n-step episode rollout of the predicted states.

    Args:
        next_states (np.ndarray): A numpy array containing the actual next states.
            Shape: (1, n_steps, 1, n_features)
        next_states_hat (np.ndarray): A numpy array containing the predicted next states.
            Shape: (1, n_steps, 1, n_features)
        dataset (Any): An object containing dataset configuration.
            Must have a 'dataset_config' attribute with a 'states' key.

    Returns:
        None: This function doesn't return any value but displays the plots.

    Notes:
        - Each plot shows the n-step episode rollout for a specific state.
        - X-axis represents timesteps, Y-axis represents the state value.
        - The plot includes two line graphs: one for the actual states and one for the predicted states.
        - The actual states are shown in blue, and the predicted states are shown in orange.
        - The actual states are represented by solid lines, and the predicted states are represented by dashed lines.
        - The plot includes a legend to differentiate between the actual and predicted states.
    """
    assert next_states.shape == next_states_hat.shape, "Shape mismatch between actual and predicted states"

    n_steps, n_features = next_states.shape[1], next_states.shape[3]
    timesteps = range(n_steps)

    for i in range(n_features):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(timesteps, next_states[0, :, 0, i], label=f"Actual State {i}", color="blue")
        ax.plot(timesteps, next_states_hat[0, :, 0, i], label=f"Predicted State {i}", color="orange", linestyle="--")
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("State Value")
        ax.set_title(f"State Index {i}")
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    return

def plot_n_step_evaluation(evaluation_results: Dict[str, Dict[str, Dict[str, Any]]], dataset: Any) -> None:
    """
    Plot n-step evaluation results for each state in the dataset.

    This function creates a separate plot for each state in the dataset,
    showing the evaluation metric over different timesteps.

    Args:
        evaluation_results (Dict[str, Dict[str, Dict[str, Any]]]): A nested dictionary containing evaluation results.
            Structure: {timestep: {feature: {metric: str, value: float, name: str, type: str}}}
        dataset (Any): An object containing dataset configuration.
            Must have a 'dataset_config' attribute with a 'states' key.

    Returns:
        None: This function doesn't return any value but displays the plots.

    Notes:
        - Each plot shows the evaluation metric for a specific state across all timesteps.
        - X-axis represents timesteps, Y-axis represents the evaluation metric.
        - The plot includes a line graph with markers for each data point.
        - Grid lines are added for better readability.
        - X-axis labels are rotated 45 degrees if there are many timesteps.
    """

    states: list = dataset.dataset_config["states"]
    timesteps: list = list(evaluation_results.keys())

    for feature_key in states:
        data: list = [evaluation_results[timestep][feature_key]["value"] for timestep in timesteps]
        metric: str = evaluation_results[timesteps[0]][feature_key]["metric"]
        name: str = evaluation_results[timesteps[0]][feature_key]["name"]
        data_type: str = evaluation_results[timesteps[0]][feature_key]["type"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timesteps, data, label=metric, color='b', marker='o')
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel(metric)
        ax.set_title(f'State Name: {name} ({data_type})')
        ax.legend()
        ax.grid(True)
        
        ax.set_xticks(timesteps)
        
        plt.tight_layout()
        plt.show()

    return