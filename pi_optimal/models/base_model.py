import numpy as np
from tqdm.auto import tqdm
import pickle
from torch.utils.data import DataLoader
import joblib

class BaseModel:
    
    def fit(self, dataset):
        """Fits the model to the dataset."""
        raise NotImplementedError

    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        # Predict all features except the reward
        next_state = []
        for i, model in enumerate(self.models):
            if i != self.dataset_config["reward_feature_idx"]:
                feature_next_state = model.predict(X)
                next_state.append(feature_next_state)
        
        # Predict the reward from the next state
        next_state = np.array(next_state).T
        reward_idx = self.dataset_config["reward_vector_idx"]
        reward = self.models[self.dataset_config["reward_feature_idx"]].predict(next_state)
        next_state = np.insert(next_state, reward_idx, reward, axis=1)

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
            state[:,-1] = next_state
        next_states = np.array(next_states)
        next_states = np.transpose(next_states, (2, 0, 1, 3))
        return next_states
        
    def save(self, filepath):
        with open(filepath, "wb") as f:
            joblib.dump(
                {
                    "models": self.models,
                    "dataset_config": self.dataset_config,
                    "params": self.params,
                },
                f,
                compress=3
            )

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = joblib.load(f)

        instance = cls(**data["params"])
        instance.models = data["models"]
        instance.dataset_config = data["dataset_config"]
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
        X = self._prepare_input_data(past_states, past_actions)
        y = self._prepare_target_data(future_states)

        self.dataset_config = dataloader.dataset.dataset_config

        self.models = [
            self._create_estimator(self.dataset_config["states"][state_idx]["type"])
            for state_idx in self.dataset_config["states"]
        ]

        # Fit all models except the reward model
        for i, model in enumerate(tqdm(self.models, desc="Training models...")):
            if i != self.dataset_config["reward_feature_idx"]:
                y_target = self._get_target_for_feature(y, i)
                model.fit(X, y_target)
            else:
                reward_idx = self.dataset_config["reward_vector_idx"]
                target_reward = self._get_target_for_feature(y, i)
                next_state = y
                next_state_wo_reward = np.delete(next_state, reward_idx, axis=1)
                model.fit(next_state_wo_reward, target_reward)
