# pi_optimal/models/base_sklearn_model.py
import numpy as np
from tqdm.auto import tqdm
import pickle
from torch.utils.data import DataLoader
from pi_optimal.models.base_model import BaseModel

class BaseSklearnModel(BaseModel):
    def __init__(self, **params):
        """
        Initialize the model.

        Parameters:
            use_past_states_for_reward (bool): Determines the input used for reward prediction.
                - False: Use only the predicted next state (non-reward part).
                - True:  Use a concatenation of the past states and actions 
                         and the predicted next state to predict the reward.
                Defaults to True.
            Other parameters may be passed via **params.
        """
        super().__init__(**params)
        self.params = params
      
    def fit(self, dataset):
        """Fits the model to the dataset."""
        raise NotImplementedError

    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        # Predict all features except the reward
        next_state_pred = []
        for i, model in enumerate(self.models):
            if i != self.dataset_config["reward_feature_idx"]:
                feature_next_state = model.predict(X)
                next_state_pred.append(feature_next_state)
        
        # Convert predictions into a (n_samples, n_features_without_reward) array
        next_state_pred = np.array(next_state_pred).T

        if self.use_past_states_for_reward:
            # Concatenate the past input with the predicted next state to predict the reward.   
            reward_input = np.concatenate([X, next_state_pred], axis=1)
        else:
            # Use only the predicted next state to predict the reward.
            reward_input = next_state_pred

        reward_idx = self.dataset_config["reward_vector_idx"]
        reward = self.models[self.dataset_config["reward_feature_idx"]].predict(reward_input)
        
        # Insert the predicted reward back into the next state array at the configured position
        next_state = np.insert(next_state_pred, reward_idx, reward, axis=1)
        return next_state

    def forward(self, state, action):
        X = self._prepare_input_data(state, action)
        return self.predict(X)
    
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
            state[:, -1] = next_state
        next_states = np.array(next_states)
        next_states = np.transpose(next_states, (2, 0, 1, 3))
        return next_states
        
    def save(self, filepath):
        """Secure model saving with metadata and pickle"""
        save_data = {
            "models": self.models,
            "dataset_config": self.dataset_config,
            "params": self.params,
            "model_type": self.__class__.__name__,
            "model_config": getattr(self, 'model_config', None)
        }
            
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        """Safe model loading with version checking"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if data.get('model_type') != cls.__name__:
            raise ValueError(f"Model type mismatch: Expected {cls.__name__}, got {data.get('model_type')}")
            
        instance = cls(params=data["params"])
        instance.models = data["models"]
        instance.dataset_config = data["dataset_config"]
        if 'model_config' in data:
            instance.model_config = data["model_config"]
        return instance

    def _prepare_input_data(self, past_states, past_actions):
        flatten_past_states = past_states.reshape(past_states.shape[0], -1)
        flatten_past_actions = past_actions.reshape(past_actions.shape[0], -1)
        return np.concatenate([flatten_past_states, flatten_past_actions], axis=1)

    def _prepare_target_data(self, future_states):
        assert future_states.shape[1] == 1  # only support one step ahead prediction
        future_states = np.array(future_states)
        return future_states.reshape(-1, future_states.shape[-1])

    def _get_target_for_feature(self, y, feature_index):
        feature = self.dataset_config["states"][feature_index]
        feature_begin_idx = feature["feature_begin_idx"]
        feature_end_idx = feature["feature_end_idx"]
        return y[:, feature_begin_idx:feature_end_idx].ravel()

    def fit(self, dataset):
        self.dataset_config = dataset.dataset_config

        dataloader = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False, num_workers=0
        )
        past_states, past_actions, future_states, _ = next(iter(dataloader))
        past_states_actions = self._prepare_input_data(past_states, past_actions)
        next_states = self._prepare_target_data(future_states)

        self.dataset_config = dataloader.dataset.dataset_config

        self.models = [
            self._create_estimator(self.dataset_config["states"], state_idx)
            for state_idx in self.dataset_config["states"]
        ]

        # Fit the models
        for i, model in enumerate(tqdm(self.models, desc="Training models...")):
            if i != self.dataset_config["reward_feature_idx"]:
                next_state_target = self._get_target_for_feature(next_states, i)
                model.fit(past_states_actions, next_state_target)
            else:
                reward_idx = self.dataset_config["reward_vector_idx"]
                target_reward = self._get_target_for_feature(next_states, i)
                # Remove the reward from the next state to predict the reward.
                next_state_wo_reward = np.delete(next_states, reward_idx, axis=1)

                if self.use_past_states_for_reward:
                    # Concatenate the past input with the next state without reward to predict the reward.
                    states_w_next_states_wo_reward = np.concatenate([past_states_actions, next_state_wo_reward], axis=1)
                    model.fit(states_w_next_states_wo_reward, target_reward)
                else:
                    model.fit(next_state_wo_reward, target_reward)