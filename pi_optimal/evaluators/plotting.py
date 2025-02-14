import numpy as np
from typing import Any, Dict
import plotly.graph_objects as go


def plot_n_step_episode_rollout(next_states: np.ndarray, 
                                next_states_hat: np.ndarray, 
                                dataset: Any) -> None:
    """
    Plot n-step episode rollout with Plotly and a dropdown menu.

    Args:
        next_states (np.ndarray): Actual next states.
            Shape: (1, n_steps, 1, n_features)
        next_states_hat (np.ndarray): Predicted next states.
            Shape: (1, n_steps, 1, n_features)
        dataset (Any): An object containing dataset configuration.
            Must have a 'dataset_config' attribute with a 'states' key
            (a list of state/variable configurations, each having a "name" key).

    Returns:
        None
    """
    # Ensure the shapes match.
    assert next_states.shape == next_states_hat.shape, "Shape mismatch between actual and predicted states"

    n_steps = next_states.shape[1]
    n_features = next_states.shape[3]
    timesteps = list(range(n_steps))
    
    # Get variable names from dataset config.
    # Assumes dataset_config["states"] is a list of dicts with at least a "name" key.
    state_configs = dataset.dataset_config.get("states")
    state_names = [state_configs[i]["name"] for i in range(len(state_configs))]

    # Create a Plotly figure.
    fig = go.Figure()

    # Add two traces per state: one for actual and one for predicted.
    # Only the traces for the first state (index 0) are visible initially.
    for i in range(n_features):
        visible_flag = True if i == 0 else False
        
        # Actual state trace.
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=next_states[0, :, 0, i],
            mode='lines',
            visible=visible_flag,
            name=f"Actual: {state_names[i]}",
            line=dict(color="blue", dash="solid")
        ))
        # Predicted state trace.
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=next_states_hat[0, :, 0, i],
            mode='lines',
            visible=visible_flag,
            name=f"Predicted: {state_names[i]}",
            line=dict(color="orange", dash="dash")
        ))

    # Create dropdown buttons. Each button updates the 'visible'
    # property so that only the two traces for the selected state are shown.
    buttons = []
    total_traces = 2 * n_features
    for i in range(n_features):
        # Create a visibility list for all traces: only traces for state i are visible.
        visibility = [False] * total_traces
        visibility[2 * i] = True       # Actual trace.
        visibility[2 * i + 1] = True   # Predicted trace.
        
        button = dict(
            label=f"{state_names[i]}",
            method="update",
            args=[{"visible": visibility},
                  {"title": f"n-step Episode Rollout for {state_names[i]}",
                   "xaxis": {"title": "Timestep"},
                   "yaxis": {"title": "State Value"}}]
        )
        buttons.append(button)

    # Update layout to include the dropdown menu.
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        title=f"{n_steps}-step Episode Rollout for {state_names[0]}",
        xaxis_title="Timestep",
        yaxis_title="State Value",
        template="plotly_white"
    )

    fig.show()


def plot_n_step_evaluation(evaluation_results: Dict[str, Dict[str, Dict[str, Any]]],
                           dataset: Any) -> None:
    """
    Plot n-step evaluation results with Plotly and a dropdown menu.

    Args:
        evaluation_results (Dict[str, Dict[str, Dict[str, Any]]]): 
            A nested dictionary containing evaluation results.
            Structure: {timestep: {feature: {metric: str, value: float, name: str, type: str}}}
        dataset (Any): An object containing dataset configuration.
            Must have a 'dataset_config' attribute with a 'states' key.
            The 'states' key should be a list of state configurations. If each state
            configuration is a dict, it is assumed to contain a "name" key that will
            be used to index evaluation_results.
    
    Returns:
        None
    """
    # Extract state keys for evaluation.
    # If dataset.dataset_config["states"] is a list of dictionaries, extract their "name" value;
    # otherwise, assume it's already a list of hashable keys.
    states_config = dataset.dataset_config.get("states")
    if states_config is not None:
        state_keys = [state["name"] if isinstance(state, dict) else state for state in states_config]
    else:
        # Fallback to using the keys from the evaluation_results at the first timestep.
        state_keys = list(evaluation_results[next(iter(evaluation_results))].keys())
    
    # Get and sort timesteps. (Assumes keys can be converted to float.)
    timesteps = list(evaluation_results.keys())
    timesteps = sorted(timesteps, key=lambda t: float(t))
    
    # Create a Plotly figure.
    fig = go.Figure()
    
    # Add one trace per state.
    # Only the first state's trace is visible initially.
    for i, state_key in enumerate(state_keys):
        # Extract evaluation data for this state across timesteps.
        y_data = [evaluation_results[t][state_key]["value"] for t in timesteps]
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=y_data,
            mode='lines+markers',
            visible=True if i == 0 else False,
            name=state_key
        ))
    
    # Create dropdown buttons so that only one state's trace is visible at a time.
    buttons = []
    for i, state_key in enumerate(state_keys):
        # Create visibility: only one trace visible per state.
        visibility = [False] * len(state_keys)
        visibility[i] = True
        
        # Extract metric information from the first timestep for the current state.
        metric = evaluation_results[timesteps[0]][state_key]["metric"]
        name_val = evaluation_results[timesteps[0]][state_key]["name"]
        data_type = evaluation_results[timesteps[0]][state_key]["type"]
        
        button = dict(
            label=state_key,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{name_val} ({data_type}) - {metric}",
                   "xaxis": {"title": "Timestep"},
                   "yaxis": {"title": metric}}]
        )
        buttons.append(button)
    
    # Update layout to include the dropdown menu.
    init_state = state_keys[0]
    init_metric = evaluation_results[timesteps[0]][init_state]["metric"]
    init_name = evaluation_results[timesteps[0]][init_state]["name"]
    init_type = evaluation_results[timesteps[0]][init_state]["type"]

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        title=f"{init_name} ({init_type}) - {init_metric}",
        xaxis_title="Timestep",
        yaxis_title=init_metric,
        template="plotly_white"
    )
    
    fig.show()
