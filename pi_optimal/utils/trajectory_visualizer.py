# visualization.py
import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display
import ipysheet
from ..agents.agent import Agent
from ..datasets.base_dataset import BaseDataset

class TrajectoryVisualizer:
    def __init__(self, agent: Agent, current_dataset: BaseDataset, best_actions: np.array = None):
        self.agent = agent
        self.current_dataset = current_dataset
        self.best_actions = best_actions

        # Initialize variables
        self.predicted_trajectories = None
        self.current_actions = None
        self.saved_trajectories = []

        # Prepare action and state information
        self._prepare_action_info()
        self._prepare_state_info()

        # Create interactive widgets
        self.action_sheet = self._create_action_sheet()
        self.update_button = self._create_update_button()
        self.save_button = self._create_save_button()
        self.state_dropdown = self._create_state_dropdown()
        self.fig = go.FigureWidget()

        # Set up UI
        self.ui = widgets.VBox([
            self.action_sheet,
            widgets.HBox([self.update_button, self.save_button]),
            self.state_dropdown,
            self.fig
        ])

    def display(self):
        display(self.ui)

    def _prepare_action_info(self):
        # Extract action names
        self.action_names = [
            self.current_dataset.dataset_config["actions"][key]["name"]
            for key in self.current_dataset.dataset_config["actions"]
        ]
        if self.best_actions is None:
            horizon = self.agent.config["horizon"]
            self.initial_actions = self.backtransform(self.current_dataset.actions, 
                                                   self.current_dataset.dataset_config["actions"])[:horizon]
        else:
            self.initial_actions = self.best_actions

    def _prepare_state_info(self):
        # Extract state information
        states = self.current_dataset.dataset_config["states"]
        self.state_names = [states[state_idx]["name"] for state_idx in states]
        state_types = [states[state_idx]["type"] for state_idx in states]
        feature_begin_idx = [states[state_idx]["feature_begin_idx"] for state_idx in states]
        feature_end_idx = [states[state_idx]["feature_end_idx"] for state_idx in states]

        # Create a mapping from state names to their indices and types
        self.state_info = {
            name: {
                "index": idx,
                "type": state_types[idx],
                "feature_begin": feature_begin_idx[idx],
                "feature_end": feature_end_idx[idx]
            }
            for idx, name in enumerate(self.state_names)
        }

    def _create_action_sheet(self):
        action_array = self.initial_actions
        action_names = self.action_names
        time_steps, action_dim = action_array.shape
        sheet = ipysheet.sheet(rows=time_steps, columns=action_dim, column_headers=action_names)
        sheet.layout.width = '600px'
        sheet.layout.height = '300px'

        # Create cells for actions
        for t in range(time_steps):
            for d in range(action_dim):
                ipysheet.cell(sheet=sheet, row=t, column=d, value=action_array[t, d], numeric_format='0.00')
        return sheet

    def _create_update_button(self):
        button = widgets.Button(
            description='Update Trajectory',
            button_style='success',
            tooltip='Click to resimulate with new actions',
        )
        button.on_click(self._on_update_button_clicked)
        return button

    def _create_save_button(self):
        button = widgets.Button(
            description='Save Trajectory',
            button_style='info',
            tooltip='Save the current trajectory for comparison',
        )
        button.on_click(self._on_save_button_clicked)
        return button

    def _create_state_dropdown(self):
        dropdown = widgets.Dropdown(
            options=self.state_names,
            value=self.state_names[0],
            description='Select State:',
            disabled=False,
        )
        dropdown.observe(self._on_state_change, names='value')
        return dropdown

    def _resimulate_trajectories(self):
        # Read the action array from the sheet
        action_values = []
        for cell in self.action_sheet.cells:
            action_values.append(cell.value)
        action_array_new = np.array(action_values).reshape(self.initial_actions.shape)
        self.current_actions = action_array_new.copy()
        action_array_new = self.transform(action_array_new, self.current_dataset.dataset_config["actions"])
        

        # Resimulate the trajectories
        state, action_history, _, _ = self.current_dataset[len(self.current_dataset) - 1]
        state = state[np.newaxis, :]
        action_history = action_history[np.newaxis, :]
        simulation_actions = action_array_new[np.newaxis, :]
        self.predicted_trajectories = self.agent.policy.simulate_trajectories(
            models=self.agent.models,
            states=state,
            action_history=action_history,
            actions=simulation_actions
        )
        for i in range(len(self.predicted_trajectories)):
            self.predicted_trajectories[i][0] = self.backtransform(
                self.predicted_trajectories[i][0],
                self.current_dataset.dataset_config["states"]
            )

    def _on_update_button_clicked(self, b):
        self._resimulate_trajectories()
        self._plot_trajectory(self.state_dropdown.value)

    def _on_save_button_clicked(self, b):
        if self.predicted_trajectories is not None:
            # Calculate uncertainty
            state_idx = self.state_info[self.state_dropdown.value]["index"]
            state_uncertainty = np.array(self.predicted_trajectories).std(axis=0)[0][:, state_idx]
            # Save the current trajectory
            self.saved_trajectories.append({
                'actions': self.current_actions.copy(),
                'trajectory': self.predicted_trajectories.copy(),
                'uncertainty': state_uncertainty.copy(),
                'label': f'Trajectory {len(self.saved_trajectories)+1}'
            })
            print(f"Trajectory {len(self.saved_trajectories)} saved.")
            self._plot_trajectory(self.state_dropdown.value)

    def _plot_trajectory(self, selected_state):
        if self.predicted_trajectories is None:
            print("Please resimulate the trajectories by clicking 'Update Trajectory' button.")
            return
        # Retrieve state information
        state_idx = self.state_info[selected_state]["index"]
        state_type = self.state_info[selected_state]["type"]
        if state_type == "numerical":
            # Prepare the data for plotting
            with self.fig.batch_update():
                self.fig.data = []  # Clear existing data
                # Plot saved trajectories
                for saved in self.saved_trajectories:
                    traj = saved['trajectory']
                    actions = saved['actions']
                    uncertainty = saved['uncertainty']
                    state_estimates = np.array(traj).mean(axis=0)[0][:, state_idx]
                    time_steps = np.arange(len(state_estimates))
                    # Prepare customdata for hover
                    customdata = actions.tolist()
                    hovertemplate = (
                        'Time: %{x}<br>'
                        'Value: %{y}<br>'
                        + ''.join([f'{name}: %{{customdata[{i}]}}<br>' for i, name in enumerate(self.action_names)])
                        + '<extra></extra>'
                    )
                    # Plot the trajectory line
                    self.fig.add_trace(go.Scatter(
                        x=time_steps,
                        y=state_estimates,
                        mode='lines+markers',
                        name=saved['label'],
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        line=dict(width=2)
                    ))
                    # Plot the uncertainty band
                    self.fig.add_trace(go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([state_estimates + uncertainty,
                                         (state_estimates - uncertainty)[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{saved["label"]} Uncertainty',
                        showlegend=False
                    ))
                # Plot the current trajectory
                state_estimates = np.array(self.predicted_trajectories).mean(axis=0)[0][:, state_idx]
                state_uncertainty = np.array(self.predicted_trajectories).std(axis=0)[0][:, state_idx]
                time_steps = np.arange(len(state_estimates))
                actions = self.current_actions
                customdata = actions.tolist()
                hovertemplate = (
                    'Time: %{x}<br>'
                    'Value: %{y}<br>'
                    + ''.join([f'{name}: %{{customdata[{i}]}}<br>' for i, name in enumerate(self.action_names)])
                    + '<extra></extra>'
                )
                # Plot the current trajectory line
                self.fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=state_estimates,
                    mode='lines+markers',
                    name='Current Trajectory',
                    line=dict(color='blue', width=3, dash='dash'),
                    customdata=customdata,
                    hovertemplate=hovertemplate
                ))
                # Plot the uncertainty band for the current trajectory
                self.fig.add_trace(go.Scatter(
                    x=np.concatenate([time_steps, time_steps[::-1]]),
                    y=np.concatenate([state_estimates + state_uncertainty,
                                     (state_estimates - state_uncertainty)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,180,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Uncertainty',
                    showlegend=False
                ))
                self.fig.update_layout(
                    title=f"Predicted {selected_state} Trajectory",
                    xaxis_title="Time Step",
                    yaxis_title=selected_state,
                    hovermode='x unified',
                    template='plotly_white'
                )
        else:
            print(f"State '{selected_state}' is not numerical and cannot be plotted.")

    def _on_state_change(self, change):
        self._plot_trajectory(change['new'])

    def backtransform(self, array: np.array, config_list: list) -> np.array:
        """
        Backtransforms the array using the config list.
        """
        retransformed_array = array.copy()
        for i in range(len(config_list)):
            if config_list[i]["processor"] is not None:
                if config_list[i]["type"] == "numerical":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    retransformed_array[:,feature_begin_idx] = config_list[i]["processor"].inverse_transform(retransformed_array[:,feature_begin_idx].reshape(-1,1), copy=False).reshape(-1)
        return retransformed_array

    def transform(self, array: np.array, config_list: list) -> np.array:
        """
        Transforms the array using the config list.
        """
        transformed_array = array.copy()
        for i in range(len(config_list)):
            if config_list[i]["processor"] is not None:
                if config_list[i]["type"] == "numerical":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    transformed_array[:,feature_begin_idx] = config_list[i]["processor"].transform(transformed_array[:,feature_begin_idx].reshape(-1,1), copy=False).reshape(-1)
                elif config_list[i]["type"] == "categorial":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    transformed_array[:,feature_begin_idx] = config_list[i]["processor"].transform(transformed_array[:,feature_begin_idx].reshape(-1,1)).reshape(-1)
                
        return transformed_array