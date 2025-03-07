# pi_optimal/models/pytorch/pytorch_base_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pi_optimal.models.base_model import BaseModel

class PytorchBaseModel(BaseModel, nn.Module):
    def __init__(self, dataset_config, model_params=None, probabilistic=False, epochs=10, lr=1e-3, batch_size=32, device='cpu'):
        """
        Args:
            dataset_config (dict): Configuration including 'states', 'actions', and 'lookback_timesteps'.
            model_params (dict, optional): Parameters for the shared trunk network.
            probabilistic (bool): If True, numerical outputs will be modeled probabilistically 
                                  (i.e. predicting both mean and log-variance).
        """
        super().__init__()
        nn.Module.__init__(self)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device


        self.dataset_config = dataset_config
        self.lookback_timesteps = dataset_config.get("lookback_timesteps", 1)
        self.probabilistic = probabilistic
        
        # Expect these sizes to be set in the config
        self.state_size = dataset_config.get("states_size", None)
        self.action_size = dataset_config.get("actions_size", None)
        if self.state_size is None or self.action_size is None:
            raise ValueError("Dataset config must include 'states_size' and 'actions_size'.")

        # Input dimension: flattening past states and actions over the lookback period.
        self.input_dim = self.lookback_timesteps * (self.state_size + self.action_size)
        
        # Build the shared trunk (default is an MLP; subclasses can override)
        self.shared_network = self.build_shared_network(model_params)
        if not hasattr(self, 'shared_output_dim'):
            raise ValueError("build_shared_network must set self.shared_output_dim")
        
        # Create output heads – one per state feature.
        # For numerical features: if probabilistic, head outputs 2 values per feature (mean & log-variance).
        self.output_heads = nn.ModuleDict()
        for idx, feature in self.dataset_config["states"].items():
            head_dim = feature["feature_end_idx"] - feature["feature_begin_idx"]
            if feature["type"] == "numerical":
                if self.probabilistic:
                    # Output both mean and log variance.
                    self.output_heads[str(idx)] = nn.Linear(self.shared_output_dim, head_dim * 2)
                else:
                    self.output_heads[str(idx)] = nn.Linear(self.shared_output_dim, head_dim)
            elif feature["type"] in ["categorial", "binary"]:
                # Classification head: number of classes should be provided (e.g., via 'n_classes').
                
                if feature["type"] == "binary":
                    n_classes = 2
                else:
                    processor = feature["processor"]
                    n_classes = len(processor.categories_[0])
                if n_classes is None:
                    raise ValueError(f"'n_classes' must be provided for categorial feature {feature['name']}")
                self.output_heads[str(idx)] = nn.Linear(self.shared_output_dim, n_classes)
            else:
                raise ValueError(f"Unknown feature type: {feature['type']}")

    def build_shared_network(self, params):
        """
        Build the shared trunk network.
        Subclasses may override this method to implement custom architectures (e.g., LSTM).
        A default MLP is provided here.
        
        Args:
            params (dict, optional): Parameters for the MLP (e.g., hidden dimensions).
        
        Returns:
            nn.Module: The shared network.
        """
        hidden_dim = params.get("hidden_dim", 128) if params else 128
        network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shared_output_dim = hidden_dim  # Expose the output dimension.
        return network

    def _prepare_input_data(self, past_states, past_actions):
        """
        Flatten the past states and actions and concatenate them.
        Args:
            past_states (Tensor): Shape (batch, lookback, state_dim)
            past_actions (Tensor): Shape (batch, lookback, action_dim)
        Returns:
            Tensor: Shape (batch, lookback*(state_dim+action_dim))
        """
        if not isinstance(past_states, torch.Tensor):
            past_states = torch.tensor(past_states, dtype=torch.float32)
        if not isinstance(past_actions, torch.Tensor):
            past_actions = torch.tensor(past_actions, dtype=torch.float32)
            
        batch_size = past_states.size(0)
        x = torch.cat([past_states.view(batch_size, -1),
                    past_actions.view(batch_size, -1)], dim=1)
        return x.float()  


    def forward_one_step(self, past_states, past_actions):
        """Performs a single-step prediction given past states and actions."""
        x = self._prepare_input_data(past_states, past_actions)
        shared_out = self.shared_network(x)
        head_outputs = []
        for idx in sorted(self.dataset_config["states"].keys(), key=lambda k: int(k)):
            head = self.output_heads[str(idx)]
            out = head(shared_out)
            head_outputs.append(out)
        next_state = torch.cat(head_outputs, dim=1)
        return next_state


    def forward(self, past_states, past_actions, future_actions=None, horizon=1):
        """
        When future_actions is provided, perform autoregressive multi-step prediction.
        When future_actions is None, perform a one-step prediction.
        
        Args:
            past_states (Tensor): shape (batch, lookback, state_dim)
            past_actions (Tensor): shape (batch, lookback, action_dim)
            future_actions (Tensor, optional): shape (batch, horizon, action_dim)
            horizon (int): Number of steps to predict.
        
        Returns:
            Tensor: If future_actions is None, returns a one-step prediction.
                    Otherwise returns predictions of shape (batch, horizon, total_state_dim)
        """
        if future_actions is None:
            return self.forward_one_step(past_states, past_actions).detach().cpu().numpy()
        
        predictions = []
        state_seq = past_states  # (batch, lookback, state_dim)
        action_seq = past_actions  # (batch, lookback, action_dim)
        for t in range(horizon):
            current_future_action = future_actions[:, t, :]  # (batch, action_dim)
            x = self._prepare_input_data(state_seq, action_seq)
            shared_out = self.shared_network(x)
            head_outputs = []
            for idx in sorted(self.dataset_config["states"].keys(), key=lambda k: int(k)):
                head = self.output_heads[str(idx)]
                out = head(shared_out)
                head_outputs.append(out)
            next_state = torch.cat(head_outputs, dim=1)
            predictions.append(next_state.unsqueeze(1))
            state_seq = torch.cat([state_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)
            action_seq = torch.cat([action_seq[:, 1:, :], current_future_action.unsqueeze(1)], dim=1)
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def compute_loss(self, predictions, future_states):
        """
        Compute a combined loss over all predicted time steps and output heads.
        For numerical outputs, if probabilistic, use the negative log-likelihood of a Gaussian;
        otherwise, use MSE loss. For categorial outputs, use cross entropy.
        
        Args:
            predictions (Tensor): (batch, horizon, total_state_dim)
            future_states (Tensor): (batch, horizon, total_state_dim) – ground truth.
        
        Returns:
            Tensor: The aggregated loss.
        """
        total_loss = 0.0
        for idx, feature in self.dataset_config["states"].items():
            begin = feature["feature_begin_idx"]
            end = feature["feature_end_idx"]
            pred_feature = predictions[:, :, begin:end]
            true_feature = future_states[:, :, begin:end]
            if feature["type"] == "numerical":
                if self.probabilistic:
                    # For probabilistic outputs, split predictions into mean and log variance.
                    head_dim = end - begin
                    pred_mean = pred_feature[..., :head_dim]
                    pred_log_var = pred_feature[..., head_dim:]
                    # Negative log-likelihood for Gaussian
                    # Constant terms are omitted since they don't affect gradients.
                    nll = 0.5 * (pred_log_var + ((true_feature - pred_mean) ** 2) / torch.exp(pred_log_var))
                    loss = nll.mean()
                else:
                    loss = nn.MSELoss()(pred_feature, true_feature)
            elif feature["type"] in ["categorial", "binary"]:
                loss = nn.CrossEntropyLoss()(
                    pred_feature[:,0,:],
                    true_feature[:,0,:],
                )
            else:
                raise ValueError(f"Unknown feature type: {feature['type']}")
            total_loss += loss
        return total_loss

    def fit(self, dataset):
        """
        A simple training loop for the model.
        Args:
            dataset: A PyTorch dataset that returns a tuple 
                     (past_states, past_actions, future_states, future_actions).
            epochs (int): Number of epochs.
            lr (float): Learning rate.
            batch_size (int): Batch size.
            device (str): 'cpu' or 'cuda'.
        Returns:
            self
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)
        self.train()
        
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0          
            for batch in dataloader:
                past_states, past_actions, future_states, future_actions = batch
                past_states = past_states.to(self.device).float()
                past_actions = past_actions.to(self.device).float()
                future_states = future_states.to(self.device).float()
                future_actions = future_actions.to(self.device).float()
                
                horizon = future_states.size(1)
                optimizer.zero_grad()
                preds = self.forward(past_states, past_actions, future_actions, horizon=horizon)
                loss = self.compute_loss(preds, future_states)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader)}")
        return self

    def save(self, filepath):
        """
        Save model state, dataset configuration, and metadata.
        """
        save_data = {
            "state_dict": self.state_dict(),
            "dataset_config": self.dataset_config,
            "model_type": self.__class__.__name__,
            "probabilistic": self.probabilistic,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath, device='cpu', **kwargs):
        """
        Load a saved model.
        Args:
            filepath (str): Path to the saved file.
            device (str): Device to load the model onto.
            kwargs: Additional parameters for the model constructor.
        Returns:
            An instance of cls.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if data.get("model_type") != cls.__name__:
            raise ValueError(f"Model type mismatch: Expected {cls.__name__}, got {data.get('model_type')}")
        dataset_config = data["dataset_config"]
        # Use the saved probabilistic flag if not overridden.
        probabilistic = data.get("probabilistic", False)
        model = cls(dataset_config, probabilistic=probabilistic, **kwargs)
        model.load_state_dict(data["state_dict"])
        model.to(device)
        return model
    
    