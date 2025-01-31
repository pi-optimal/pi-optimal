import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display
import ipysheet
from pi_optimal.agents.agent import Agent
from pi_optimal.datasets.base_dataset import BaseDataset
from plotly.colors import sequential, hex_to_rgb
import logging

class TrajectoryVisualizer:
    """
    Tool for visualizing and comparing trajectories of different action sequences.

    This tool allows you to visualize the predicted trajectories of a state variable under different action sequences.
    You can simulate the trajectory of a state variable by providing a sequence of actions and clicking the 'Simulate Trajectory' button.
    You can save the simulated trajectory for comparison with other trajectories by clicking the 'Save Trajectory' button.
    You can compare the saved trajectories with the current trajectory by selecting the state variable from the dropdown menu.
    
    Attributes:
        agent (Agent): The agent object that contains the policy and models.
        current_dataset (BaseDataset): The current dataset object.
        best_actions (np.array): The best actions to visualize. If None, the initial actions from the dataset are used.
        lookback_timesteps (int): The number of historical timesteps to show in the plot.
        round_digits (int): The number of digits to round the numerical state values to.
        color_palette (list): List of colors used for plotting.
        color_index (int): Index to keep track of the current color in the palette.
        color_map (dict): Mapping of labels to colors.
        historical_states_transformed (np.array): Transformed historical states.
        historical_states (np.array): Original historical states after inverse transformation.
        historical_actions (np.array): Original historical actions after inverse transformation.
        predicted_trajectories (list): List of predicted trajectories.
        current_actions (np.array): Array of current actions.
        saved_trajectories (list): List of saved trajectories for comparison.
        action_sheet (ipysheet.sheet): Interactive sheet for inputting actions.
        update_button (widgets.Button): Button to simulate the trajectory with new actions.
        save_button (widgets.Button): Button to save the current trajectory.
        state_dropdown (widgets.Dropdown): Dropdown to select the state variable for visualization.
        fig (go.FigureWidget): Plotly figure widget for visualization.
        ui (widgets.VBox): VBox widget containing the UI elements.
    """

    def __init__(
        self,
        agent: Agent,
        current_dataset: BaseDataset,
        best_actions: np.array = None,
        lookback_timesteps: int = 8,
        round_digits: int = 2
    ):
        """ 
        Initializes the TrajectoryVisualizer. 
        
        Args:
            agent (Agent): The agent object that contains the policy and models.
            current_dataset (BaseDataset): The current dataset object.
            best_actions (np.array): The best actions to visualize. If None, the initial actions from the dataset are used.
            lookback_timesteps (int): The number of historical timesteps to show in the plot.
            round_digits (int): The number of digits to round the numerical state values to.
        """
        self.agent = agent
        self.current_dataset = current_dataset
        self.best_actions = best_actions
        self.round_digits = round_digits
        self.hash_id = np.random.randint(0, 100000)
        self.logger = logging.getLogger(f"TrajectoryVisualizer-{self.hash_id}")
        
        # Check if lookback timesteps is greater than the dataset length
        if lookback_timesteps > len(self.current_dataset) - 2:
            logging.warning(f"Lookback timesteps {lookback_timesteps} is greater than the dataset length. "
                        f"Setting lookback timesteps to {len(self.current_dataset) - 2}.")
            self.lookback_timesteps = len(self.current_dataset) - 2
        else:
            self.lookback_timesteps = lookback_timesteps + 2
        
        # Color palette for plotting
        self.color_palette = [
            'rgb(133,68,205)',
            'rgb(255, 51, 194)',
            'rgb(88,77,173)',
            'rgb(255, 221, 51)',
            'rgb(255, 177, 51)',
            'rgb(120,153,221)'
        ]
        self.color_index = 0
        self.color_map = {}

        # Backtransform historical data
        self.historical_states_transformed = self.current_dataset.states[-self.lookback_timesteps:, ]
        self.historical_states = self.current_dataset.inverse_transform_features(
            "states", self.historical_states_transformed
        )
        self.historical_actions = self.current_dataset.inverse_transform_features(
            "actions", self.current_dataset.actions[-self.lookback_timesteps:, ]
        )

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
        
        # Add a placeholder annotation
        self.fig.add_annotation(
            x=0.5,
            y=0.5,
            text="Please click 'Simulate Trajectory' button to visualize an action sequeince trajectory.",
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper"
        )
        self.fig.update_layout(template='plotly_white')

        
        # Set up UI
        self.ui = widgets.VBox([
            self.action_sheet,
            widgets.HBox([self.update_button, self.save_button]),
            self.state_dropdown,
            self.fig
        ])

    def _get_color(self, label):
        """ Returns a color from the palette based on the label. """
        if label not in self.color_map:
            color = self.color_palette[self.color_index % len(self.color_palette)]
            self.color_map[label] = color
            self.color_index += 1
        return self.color_map[label]

    def _hex_to_rgba(self, hex_color, alpha=0.2):
        """ Converts a hex color to rgba with the specified alpha value. """
        rgba_color = hex_color.replace('rgb', 'rgba').replace(')', f',{alpha})')
        return rgba_color

    def display(self):
        """ Displays the UI elements. """
        display(self.ui)

    def _prepare_action_info(self):
        """ Extracts action names and initial actions from the dataset. """
        
        # Extract action names
        self.action_names = [
            self.current_dataset.dataset_config["actions"][key]["name"]
            for key in self.current_dataset.dataset_config["actions"]
        ]

        # Extract initial actions
        if self.best_actions is None:
            horizon = self.agent.config["horizon"]
            self.initial_actions = self.backtransform(
                self.current_dataset.actions,
                self.current_dataset.dataset_config["actions"]
            )[:horizon]
        else:
            self.initial_actions = self.best_actions

    def _prepare_state_info(self):
        """ Extracts state names and information from the dataset. """
        
        # Extract state information
        states = self.current_dataset.dataset_config["states"]
        self.state_names = [states[state_idx]["name"] for state_idx in states]
        state_types = [states[state_idx]["type"] for state_idx in states]
        feature_begin_idx = [states[state_idx]["feature_begin_idx"] for state_idx in states]
        feature_end_idx = [states[state_idx]["feature_end_idx"] for state_idx in states]
        processors = [states[state_idx]["processor"] for state_idx in states]

        # Create a mapping from state names to their indices and types
        self.state_info = {
            name: {
            "index": idx,
            "type": state_types[idx],
            "feature_begin": feature_begin_idx[idx],
            "feature_end": feature_end_idx[idx],
            "categories": processors[idx].transform(np.array(processors[idx].categories_[0]).reshape(-1, 1)).T[0].tolist() if state_types[idx] != "numerical" else None,
            "processor": processors[idx]
            }
            for idx, name in enumerate(self.state_names)
        }

    def _get_numeric_format(self):
        """ Returns a numeric format string based on the number of round digits. """
        return f'0.{self.round_digits * "0"}'
    
    def _create_action_sheet(self):
        """ Creates an interactive sheet for inputting actions. """
        action_array = self.initial_actions
        action_names = self.action_names
        time_steps, action_dim = action_array.shape
        sheet = ipysheet.sheet(rows=time_steps, columns=action_dim, column_headers=action_names)
        sheet.layout.width = '600px'
        sheet.layout.height = '300px'

        # Create cells for actions
        for t in range(time_steps):
            for d in range(action_dim):
                ipysheet.cell(
                    sheet=sheet,
                    row=t,
                    column=d,
                    value=action_array[t, d],
                    numeric_format=self._get_numeric_format()
                )
        return sheet

    def _create_update_button(self):
        """ Creates a button to simulate the trajectory with new actions. """
        button = widgets.Button(
            description='Simulate Trajectory',
            button_style='success',
            tooltip='Click to resimulate with new actions',
        )
        button.style.button_color = "MediumPurple"
        button.on_click(self._on_update_button_clicked)
        return button

    def _create_save_button(self):
        """ Creates a button to save the current trajectory for comparison. """
        button = widgets.Button(
            description='Save Trajectory',
            button_style='info',
            tooltip='Save the current trajectory for comparison',
        )
        button.style.button_color = "Thistle"
        button.on_click(self._on_save_button_clicked)
        return button
    

    def _create_state_dropdown(self):
        """ Creates a dropdown to select the state variable for visualization. """
        dropdown = widgets.Dropdown(
            options=self.state_names,
            value=self.state_names[0],
            description='Select State:',
            disabled=False,
        )
        dropdown.observe(self._on_state_change, names='value')
        return dropdown

    def _resimulate_trajectories(self):
        """ Resimulates the trajectories with the new actions. """
        self.logger.info("Resimulating trajectories with new actions...", "PROCESS")
        # Read the action array from the sheet
        action_values = []
        for cell in self.action_sheet.cells:
            action_values.append(cell.value)
        action_array_new = np.array(action_values).reshape(self.initial_actions.shape)
        self.current_actions = action_array_new.copy()
        action_array_new = self.transform(
            action_array_new,
            self.current_dataset.dataset_config["actions"]
        )

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
        # Backtransform the predicted states
        for i in range(len(self.predicted_trajectories)):
            self.predicted_trajectories[i][0] = self.backtransform(
                self.predicted_trajectories[i][0],
                self.current_dataset.dataset_config["states"]
            )
        self.logger.info("Trajectories resimulated successfully.", "SUCCESS")

    def _on_update_button_clicked(self, b):
        """ Callback function for the 'Simulate Trajectory' button. """
        self._resimulate_trajectories()
        self._update_plot(self.state_dropdown.value)

    def _on_save_button_clicked(self, b):
        """ Callback function for the 'Save Trajectory' button. """
        if self.predicted_trajectories is not None:
            state_idx = self.state_info[self.state_dropdown.value]["index"]
            state_type = self.state_info[self.state_dropdown.value]["type"]
            # For numerical states, we store standard deviation as uncertainty
            if state_type == "numerical":
                state_uncertainty = np.array(self.predicted_trajectories).std(axis=0)[0][:, state_idx]
            else:
                # For discrete/binary states, we won't store "uncertainty" in the same sense.
                # But we keep an empty array or None. You can store something else if you want.
                state_uncertainty = None

            self.saved_trajectories.append({
                'actions': self.current_actions.copy(),
                'trajectory': self.predicted_trajectories.copy(),
                'uncertainty': state_uncertainty,
                'label': f'Trajectory {len(self.saved_trajectories)+1}'
            })
            logging.info(f"Trajectory {len(self.saved_trajectories)} saved.")
            self._update_plot(self.state_dropdown.value)


    def _update_plot(self, selected_state):
        """ Master function to handle plotting for both numerical and discrete states. """
        
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout = {}
            
        self._plot_historical_trajectories(selected_state)

        state_type = self.state_info[selected_state]["type"]
        if state_type == "numerical":
            self._plot_numerical_trajectory(selected_state)
        elif state_type in ["categorial", "binary"]:  # or however you name them
            self._plot_discrete_trajectory(selected_state)
        else:
            logging.warning(f"State '{selected_state}' has an unknown type: {state_type}.")
        y_max = max([max(d.y) for d in self.fig.data])
        y_min = min([min(d.y) for d in self.fig.data])

        if y_max == y_min:
            y_max += 1
            y_min -= 1
        self._plot_simulation_line(y_min, y_max)

    def _plot_simulation_line(self, y_min, y_max):
        """ Adds a vertical dashed line to separate historical and simulated data. """
        
        # Increase the y-axis range a bit
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Add a vertical dashed line to separate historical and simulated data
        self.fig.add_shape(
            dict(
                type="line",
                x0=self.lookback_timesteps - 2,
                y0=y_min,
                x1=self.lookback_timesteps - 2,
                y1=y_max,
                line=dict(color="black", width=2, dash="dash")
            )
        )
    
        self.fig.add_annotation(
            x=self.lookback_timesteps - 2 - 1,
            y=y_max,
            text="⬅ Historical",
            showarrow=False,
            font=dict(size=12)
        )
        self.fig.add_annotation(
            x=self.lookback_timesteps - 2 + 1,
            y=y_max,
            text="Simulation ➡",    
            showarrow=False,
            font=dict(size=12)
        )


    def _plot_historical_trajectories(self, selected_state):
        """
        Plots the historical data (the "lookback" portion) for the selected state.
        For numerical states, shows line + markers.
        For discrete states, shows the actual realized category for each timestep.
        """
        state_idx = self.state_info[selected_state]["index"]
        state_type = self.state_info[selected_state]["type"]

        if state_type == "numerical":
            label = 'Historical Trajectory'
            color = self._get_color(label)

            # Plot the current trajectory
            states = self.historical_states[:-1, state_idx]
            states = states.astype(float)
            states = states.round(self.round_digits)
            time_steps = np.arange(0, self.lookback_timesteps - 1)

            # We'll just attach the last used action to each time step's hover data
            # so that the user sees something consistent.
            actions = self.historical_actions
            for action in actions:
                for i in range(len(action)):
                    if isinstance(action[i], (int, float)):
                        action[i] = round(action[i], self.round_digits)

            customdata = actions.tolist()
            hovertemplate = (
                'Time: %{x}<br>'
                'Value: %{y}<br>'
                + ''.join([f'{name}: %{{customdata[{i}]}}<br>' for i, name in enumerate(self.action_names)])
                + '<extra></extra>'
            )
            self.fig.add_trace(go.Scatter(
                x=time_steps,
                y=states,
                mode='lines+markers',
                name='Historical Trajectory',
                line=dict(color=color, width=3),
                hovertemplate=hovertemplate,
                customdata=customdata,
                showlegend=True,
            ))
            self.fig.update_layout(
                title=f"Predicted {selected_state} Trajectory",
                xaxis_title="Time Step",
                hovermode='x unified',
                template='plotly_white',
                yaxis=dict(
                    title=selected_state,
                    tickmode='auto',
                    tickvals=None,
                    ticktext=None
                )
            )

        elif state_type in ["categorial", "binary"]:
            
            if "categories" not in self.state_info[selected_state] or \
               self.state_info[selected_state]["categories"] is None:
                logging.warning(f"No categories found in dataset config for state '{selected_state}'.")
                return

            categories = self.state_info[selected_state]["categories"]
            processor = self.state_info[selected_state]["processor"]
            # Build a map from category to integer index
            cat_to_idx = {cat: i for i, cat in enumerate(categories)}

            # The historical states are the "raw" categories if they've been backtransformed properly.
            # However, sometimes you get numeric codes. Adjust as needed.
            hist_vals = self.historical_states[:-1, state_idx]  # shape (T,)
            hist_vals = processor.transform(hist_vals.reshape(-1, 1)).reshape(-1)
            time_steps = np.arange(0, self.lookback_timesteps - 1)

            # Convert categories to integer indices
            y_indices = []
            for val in hist_vals:
                # If already a string category, use cat_to_idx
                # If numeric or something else, handle accordingly
                if val in cat_to_idx:
                    y_indices.append(cat_to_idx[val])
                else:
                    # fallback if it's numeric or doesn't match
                    # just cast to int if possible
                    try:
                        idx = int(val)
                        if idx < len(categories):
                            y_indices.append(idx)
                        else:
                            y_indices.append(None)
                    except:
                        y_indices.append(None)

            # Plot markers at the respective category index
            label = "Historical Categories"
            color = self._get_color(label)

            z_data = np.zeros((len(categories), len(time_steps))) 
            hovertext = []
            for cat_i, cat_name in enumerate(categories):
                hovertext.append([
                    f"Time={t}, Category={cat_name}<br>Prob={z_data[cat_i, t_i]:.2f}"
                    for t_i, t in enumerate(time_steps)
                ])
            
            # Add a one a Z_data where there is a category
            for t, y in zip(time_steps, y_indices):
                if y is not None:
                    z_data[y, t] = 1.0


            self.fig.add_trace(go.Heatmap(
                    x=time_steps,
                    y=categories,
                    z=z_data,
                    text=hovertext,
                    hoverinfo="text",  # uses text matrix
                    colorscale="purples",  # pick any color scale you like
                    zmin=-0.1,
                    zmax=1.0,
                    xgap=3,  
                    ygap=3,
                    name="Predicted Distribution"
                ))

            
            # Tweak layout to show category labels
            self.fig.update_layout(
                title=f"Predicted {selected_state} Trajectory (Discrete)",
                xaxis_title="Time Step",
                hovermode='closest',
                template='plotly_white',
                yaxis=dict(
                    tickmode='array',
                    tickvals=categories, 
                    ticktext=processor.inverse_transform(np.array(categories).reshape(-1, 1)).T[0]
                )
            )

        else:
            logging.warning(f"State '{selected_state}' is not supported for historical plotting.")

    def _plot_numerical_trajectory(self, selected_state):
        """
        Plot the predicted (and saved) numerical trajectories plus uncertainty bands.
        """
        if self.predicted_trajectories is None:
            logging.warning("Please resimulate the trajectories by clicking 'Update Trajectory' button.")
            return

        # Retrieve state information
        state_idx = self.state_info[selected_state]["index"]

        with self.fig.batch_update():
            # Plot saved trajectories
            for saved in self.saved_trajectories:
                traj = saved['trajectory']
                actions = saved['actions']
                # round only numerical actions
                actions = np.array([round(action[0], self.round_digits) for action in actions if isinstance(action[0], (int, float))]).reshape(-1, 1)
                # Add a row of "nan" just to keep shape consistent with the new appended item
                actions = np.concatenate([actions, np.array([["nan"] * actions.shape[1]])])
                uncertainty = saved['uncertainty']
                label = saved['label']
                color = self._get_color(label)

                state_estimates = np.array(traj).mean(axis=0)[0][:, state_idx]
                # Insert historical last state in front so that line connects smoothly
                state_estimates = np.concatenate([[self.historical_states[-2, state_idx]], state_estimates])
                state_estimates = state_estimates.round(self.round_digits)

                if uncertainty is not None and len(uncertainty) == len(state_estimates) - 1:
                    # Insert 0 in front for the historical anchor
                    uncertainty = np.concatenate([np.zeros(1), uncertainty])
                else:
                    # fallback if no uncertainty or shape mismatch
                    uncertainty = np.zeros(len(state_estimates))

                time_steps = np.arange(
                    self.lookback_timesteps - 2,
                    self.lookback_timesteps + len(state_estimates) - 2
                )
            
                customdata = actions.tolist()
                hovertemplate = (
                    'Time: %{x}<br>'
                    'Value: %{y}<br>'
                    + ''.join([f'{name}: %{{customdata[{i}]}}<br>' for i, name in enumerate(self.action_names)])
                    + '<extra></extra>'
                )


                # Plot trajectory line
                self.fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=state_estimates,
                    mode='lines+markers',
                    name=saved['label'],
                    line=dict(color=color, width=2),
                    hovertemplate=hovertemplate,
                    customdata=customdata,
                    showlegend=True,
                    legendgroup=saved['label'],
                ))
                # Plot uncertainty band
                self.fig.add_trace(go.Scatter(
                    x=np.concatenate([time_steps, time_steps[::-1]]),
                    y=np.concatenate([
                        state_estimates + uncertainty,
                        (state_estimates - uncertainty)[::-1]
                    ]),
                    fill='toself',
                    fillcolor=self._hex_to_rgba(color, alpha=0.2),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name=f'{saved["label"]} Uncertainty',
                    showlegend=False,
                    legendgroup=saved['label'],
                ))

            # Plot the current (unsaved) trajectory
            label = 'Current Trajectory'
            color = self._get_color(label)
            state_estimates = np.array(self.predicted_trajectories).mean(axis=0)[0][:, state_idx].copy()
            state_uncertainty = np.array(self.predicted_trajectories).std(axis=0)[0][:, state_idx].copy()

            state_estimates = np.concatenate([[self.historical_states[-2, state_idx]], state_estimates])
            state_estimates = state_estimates.round(self.round_digits)
            state_uncertainty = np.concatenate([np.zeros(1), state_uncertainty])

            time_steps = np.arange(
                self.lookback_timesteps - 2,
                self.lookback_timesteps + len(state_estimates) - 2
            )
            actions = self.current_actions 
            customdata = actions.tolist()
            hovertemplate = (
                'Time: %{x}<br>'
                'Value: %{y}<br>'
                + ''.join([f'{name}: %{{customdata[{i}]}}<br>' for i, name in enumerate(self.action_names)])
                + '<extra></extra>'
            )
            self.fig.add_trace(go.Scatter(
                x=time_steps,
                y=state_estimates,
                mode='lines+markers',
                name='Current Trajectory',
                line=dict(color=color, width=3, dash='dash'),
                hovertemplate=hovertemplate,
                customdata=customdata,
                showlegend=True,
                legendgroup='current',
            ))

            self.fig.add_trace(go.Scatter(
                x=np.concatenate([time_steps, time_steps[::-1]]),
                y=np.concatenate([
                    state_estimates + state_uncertainty,
                    (state_estimates - state_uncertainty)[::-1]
                ]),
                fill='toself',
                fillcolor=self._hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='Uncertainty',
                showlegend=False,
                legendgroup='current',
            ))

            self.fig.update_layout(
                title=f"Predicted {selected_state} Trajectory",
                xaxis_title="Time Step",
                yaxis_title=selected_state,
                hovermode='x unified',
                template='plotly_white',
                yaxis=dict(
                    title=selected_state,
                    tickmode='auto',
                    tickvals=None,
                    ticktext=None
                )
            )

  
    def _plot_discrete_trajectory(self, selected_state):
        """
        Plot the predicted (and saved) discrete trajectories as a 2D heatmap:
          x-axis = timesteps (relative to the lookback horizon),
          y-axis = possible categories,
          color  = Probability that the ensemble picks that category at each timestep.
        """
        if self.predicted_trajectories is None:
            logging.warning("Please resimulate the trajectories by clicking 'Update Trajectory' button.")
            return

        # Retrieve state information
        state_idx = self.state_info[selected_state]["index"]
        categories = self.state_info[selected_state].get("categories", None)

        if not categories:
            logging.warning(f"No category list found for state '{selected_state}'. Cannot plot discrete heatmap.")
            return

        # Get the predicted distribution for each category at each time step
        num_models = len(self.predicted_trajectories)
        max_time = self.predicted_trajectories[0][0].shape[0]

        # Count the number of times each category is predicted at each time step
        category_counts = np.zeros((max_time, len(categories)))

        for m_i in range(num_models):
            # predicted_trajectories[m_i][0] => shape (T, state_dim)
            model_predictions = self.predicted_trajectories[m_i][0][:, state_idx]  # shape (T, )
            # For each time step, find the category index
            for t, val in enumerate(model_predictions):
                # If val is a string and is exactly one of the categories
                if val in categories:
                    cat_idx = categories.index(val)
                    category_counts[t, cat_idx] += 1
                else:
                    # If it is numeric, we interpret it as an index
                    try:
                        cat_idx = int(val)
                        if 0 <= cat_idx < len(categories):
                            category_counts[t, cat_idx] += 1
                    except:
                        pass

        # Convert counts to probabilities
        # shape => (T, #categories)
        category_probs = category_counts / num_models
        offset = self.lookback_timesteps - 1
        time_axis = np.arange(offset, offset + max_time)
        z_data = category_probs.T  # shape (#categories, T)

        # Build the heatmap
        hovertext = []
        for cat_i, cat_name in enumerate(categories):
            hovertext.append([
                f"Time={t}, Category={cat_name}<br>Prob={z_data[cat_i, t_i]:.2f}"
                for t_i, t in enumerate(time_axis)
            ])


        # Add one Heatmap trace
        self.fig.add_trace(go.Heatmap(
            x=time_axis,
            y=categories,
            z=z_data,
            text=hovertext,
            hoverinfo="text",  # uses text matrix
            colorscale="purples",  # pick any color scale you like
            zmin=-0.1,
            zmax=1.0,
            xgap=3,  
            ygap=3,
            name="Predicted Distribution",
            showscale=False
        ))

        processor = self.state_info[selected_state]["processor"]
        self.fig.update_layout(
            title=f"Predicted {selected_state} Distribution",
            hovermode='closest',
            template='plotly_white',
            xaxis_title="Time Step",
            yaxis=dict(
                    title=selected_state,
                    tickmode='array',
                    tickvals=categories,
                    ticktext=processor.inverse_transform(np.array(categories).reshape(-1, 1)).T[0]
            )
        )

    def _on_state_change(self, change):
        """ Callback function for the state dropdown. """
        self._update_plot(change['new'])

    def backtransform(self, array: np.array, config_list: list) -> np.array:
        """ Backtransforms the array using the config list. """
        retransformed_array = array.copy()
        for i in range(len(config_list)):
            if config_list[i]["processor"] is not None:
                if config_list[i]["type"] == "numerical":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    retransformed_array[:, feature_begin_idx] = config_list[i]["processor"].inverse_transform(
                        retransformed_array[:, feature_begin_idx].reshape(-1, 1),
                        copy=False
                    ).reshape(-1)
                    
        return retransformed_array

    def transform(self, array: np.array, config_list: list) -> np.array:
        """ Transforms the array using the config list. """
        transformed_array = array.copy()
        for i in range(len(config_list)):
            if config_list[i]["processor"] is not None:
                if config_list[i]["type"] == "numerical":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    transformed_array[:, feature_begin_idx] = config_list[i]["processor"].transform(
                        transformed_array[:, feature_begin_idx].reshape(-1, 1),
                        copy=False
                    ).reshape(-1)
                elif config_list[i]["type"] == "categorial":
                    feature_begin_idx = config_list[i]["feature_begin_idx"]
                    transformed_array[:, feature_begin_idx] = config_list[i]["processor"].transform(
                        transformed_array[:, feature_begin_idx].reshape(-1, 1)
                    ).reshape(-1)
        return transformed_array