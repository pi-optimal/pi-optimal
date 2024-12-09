import numpy as np
import pickle


class BaseModel:
    
    def fit(self, dataset):
        """Fits the model to the dataset."""
        raise NotImplementedError

    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        X_hat = np.array([model.predict(X) for model in self.models], dtype=np.float32).T
        return X_hat

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
            pickle.dump(
                {
                    "models": self.models,
                    "dataset_config": self.dataset_config,
                    "params": self.params,
                },
                f,
            )

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

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

