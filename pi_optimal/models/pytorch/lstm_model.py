# pi_optimal/models/pytorch/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pi_optimal.models.pytorch.pytorch_base_model import PytorchBaseModel
import numpy as np

class LSTMModel(PytorchBaseModel):
    """
    A PyTorch model with an LSTM shared trunk using one-hot encoded categorical features.
    
    This model processes a sequence of past states and actions using an LSTM.
    At each prediction step the last LSTM output is passed through a fully connected layer,
    and then through the output heads.
    
    For categorical features:
      - During training, the output heads produce raw logits.
        These are later converted (via softmax) and compared against one-hot encoded targets.
      - For autoregressive feedback (and during inference), the logits are converted by applying softmax,
        then taking argmax and re-encoding as a one-hot vector, so that the next input uses a discrete class.
    """
    def __init__(self, dataset_config, model_params=None, probabilistic=False,
                 epochs=10, lr=1e-3, batch_size=32, device='cpu'):
        super().__init__(dataset_config, model_params=model_params, probabilistic=probabilistic,
                         epochs=epochs, lr=lr, batch_size=batch_size, device=device)
        
        if "actions_size" not in self.dataset_config:
            self.dataset_config["actions_size"] = sum(
                feature["feature_end_idx"] - feature["feature_begin_idx"]
                for feature in self.dataset_config["actions"].values()
            )
        # Each timestep is the concatenation of state and action.
        self.input_size_seq = self.dataset_config["states_size"] + self.dataset_config["actions_size"]
        model_params = model_params if model_params is not None else {}
        self.hidden_dim = model_params.get("hidden_dim", 128)
        self.num_layers = model_params.get("num_layers", 1)

        self.lstm = nn.LSTM(
            input_size=self.input_size_seq, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.lstm_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm_fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.shared_output_dim = self.hidden_dim

    def _prepare_input_data(self, past_states, past_actions):
        if not isinstance(past_states, torch.Tensor):
            past_states = torch.tensor(past_states, dtype=torch.float32)
        if not isinstance(past_actions, torch.Tensor):
            past_actions = torch.tensor(past_actions, dtype=torch.float32)
        # If past_actions has one extra dimension (i.e. 4D instead of 3D) and that extra dimension is size 1, squeeze it.
        if past_actions.ndim == 4 and past_actions.shape[1] == 1:
            past_actions = past_actions.squeeze(dim=1)
        return torch.cat([past_states, past_actions], dim=-1).float()


    def forward(self, past_states, past_actions, future_actions=None,
                horizon=1, inference=False, temperature=1.0):
        """
        Forward pass using the LSTM-based shared network.
        
        Args:
            past_states (Tensor): shape (batch, lookback, state_dim)
            past_actions (Tensor): shape (batch, lookback, action_dim)
            future_actions (Tensor, optional): shape (batch, horizon, action_dim)
            horizon (int): Number of time steps to predict.
            inference (bool): If True, use discrete cleaning for categorical outputs (i.e. argmax -> one-hot).
                            During training, this flag should be False.
            temperature (float): Temperature for softmax (if needed).
            
        Returns:
            - If future_actions is None: returns a one-step prediction (detached)
            where categorical features are converted into one-hot vectors.
            - Otherwise returns predictions of shape (batch, horizon, total_state_dim).
            For categorical features:
                • If inference is False (training), raw logits are returned for loss calculation.
                • If inference is True (autoregressive inference), the outputs are "cleaned"
                by converting softmax probabilities to discrete one-hot vectors before feedback.
        """
        # If no future actions are provided, assume one-step prediction for inference.
        if future_actions is None:
            inference = True
            horizon = 1

        # One-step inference branch.
        if future_actions is None and inference:
            current_sequence = self._prepare_input_data(past_states, past_actions)
            lstm_out, _ = self.lstm(current_sequence)   # (batch, lookback, hidden_dim)
            x = lstm_out[:, -1, :]                 # (batch, hidden_dim)
            x = F.relu(self.lstm_fc1(x))           # (batch, hidden_dim)
            x = F.relu(self.lstm_fc2(x))
            shared_out = self.lstm_fc3(x)
            outputs = []
            for idx in sorted(self.dataset_config["states"].keys(), key=lambda k: int(k)):
                feature = self.dataset_config["states"][idx]
                head = self.output_heads[str(idx)]
                logits = head(shared_out)
                if feature["type"] in ["categorial", "binary"]:
                    # Compute softmax and then discrete prediction.
                    dist = F.softmax(logits, dim=-1)
                    argmax_idx = torch.argmax(dist, dim=-1)
                    # Convert class index to one-hot vector.
                    one_hot = F.one_hot(argmax_idx, num_classes=logits.shape[-1]).float()
                    outputs.append(one_hot)
                else:
                    outputs.append(logits)
            next_state = torch.cat(outputs, dim=1)
            return next_state.detach().cpu().numpy()
        
        # Autoregressive multi-step prediction branch.
        predictions = []    # Store raw logits for loss computation.
        current_sequence = self._prepare_input_data(past_states, past_actions)  # (batch, lookback, state+action)
        for t in range(horizon):
            lstm_out, _ = self.lstm(current_sequence)   # (batch, lookback, hidden_dim)
            last_out = lstm_out[:, -1, :]                 # (batch, hidden_dim)                # (batch, hidden_dim)
            x = F.relu (self.lstm_fc1(last_out))           # (batch, hidden_dim)
            x = F.relu(self.lstm_fc2(x))
            shared_out = self.lstm_fc3(x)          # (batch, hidden_dim)
            
            raw_logits = []       # To be used for loss.
            cleaned_outputs = []  # To be fed back autoregressively.
            for idx in sorted(self.dataset_config["states"].keys(), key=lambda k: int(k)):
                feature = self.dataset_config["states"][idx]
                head = self.output_heads[str(idx)]
                logits = head(shared_out)
                raw_logits.append(logits)
                if feature["type"] in ["categorial", "binary"]:
                    if inference:
                        with torch.no_grad():
                            dist = F.softmax(logits, dim=-1)
                            argmax_idx = torch.argmax(dist, dim=-1)
                            one_hot = F.one_hot(argmax_idx, num_classes=logits.shape[-1]).float()
                        cleaned_outputs.append(one_hot)
                    else:
                        with torch.no_grad():
                            dist = F.softmax(logits, dim=-1)
                            argmax_idx = torch.argmax(dist, dim=-1)
                            one_hot = F.one_hot(argmax_idx, num_classes=logits.shape[-1]).float()
                        cleaned_outputs.append(one_hot)
                else:
                    cleaned_outputs.append(logits)
            next_state_raw = torch.cat(raw_logits, dim=1)      # For loss computation.
            next_state_clean = torch.cat(cleaned_outputs, dim=1) # For feeding back.
            predictions.append(next_state_raw.unsqueeze(1))
            
            current_future_action = future_actions[:, t, :]     # (batch, action_dim)
            new_timestep = torch.cat([next_state_clean, current_future_action], dim=1)
            current_sequence = torch.cat([current_sequence[:, 1:, :], new_timestep.unsqueeze(1)], dim=1)
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


    def _prepare_target_data(self, future_states):
        """
        Prepare target data for loss computation.
        Assumes future_states is either a numpy array or a torch tensor.
        For one-step prediction, it reshapes the data accordingly.
        """
        if isinstance(future_states, torch.Tensor):
            future_states = future_states.detach().cpu().numpy()
        assert future_states.shape[1] == 1, "Only one-step ahead prediction is supported."
        return future_states.reshape(-1, future_states.shape[-1])

    def forward_n_steps(self, inital_state, actions, n_steps, backtransform=True):
        assert n_steps > 0 
        assert inital_state.shape[0] == actions.shape[0]
        assert actions.shape[1] == n_steps
         

        state = inital_state
        next_states = []
        for i in range(n_steps):
            action = actions[:, i]
            next_state = self.forward(state, action)
            next_states.append([next_state])
            state = np.roll(state, -1, axis=1)
            state[:,-1] = next_state
        next_states = np.array(next_states)
        next_states = np.transpose(next_states, (2, 0, 1, 3))
        return next_states
