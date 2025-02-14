# test_base_model.py
import os
import pickle
import tempfile

import numpy as np
import pytest
from torch.utils.data import Dataset, DataLoader

from pi_optimal.models.base_model import BaseModel

# --- Dummy estimator class used by our dummy model ---
class DummyEstimator:
    """
    A dummy estimator that simply returns a constant value on prediction.
    It also stores the training data passed to its fit method.
    """
    def __init__(self, constant):
        self.constant = constant
        self.fitted_X = None
        self.fitted_y = None

    def fit(self, X, y):
        self.fitted_X = X
        self.fitted_y = y

    def predict(self, X):
        # Return an array of shape (n_samples,) filled with self.constant.
        return np.full((X.shape[0],), self.constant, dtype=np.float32)


# --- Dummy model subclass ---
class DummyModel(BaseModel):
    def __init__(self, **kwargs):
        self.params = kwargs

    def _create_estimator(self, feature_type):
        if feature_type == "reward":
            return DummyEstimator(20)
        else:
            return DummyEstimator(10)


# --- Dummy dataset for testing the fit method ---
class DummyDataset(Dataset):
    """
    A dummy dataset that returns a fixed batch of data.
    
    It returns a 4-tuple: (past_states, past_actions, future_states, extra)
    where:
      - past_states: shape (n_samples, 2)
      - past_actions: shape (n_samples, 1)
      - future_states: shape (n_samples, 1, 2)  
          (Note: _prepare_target_data asserts that future_states.shape[1]==1)
      - extra: dummy data (ignored)
    """
    def __init__(self, dataset_config, n_samples=5):
        self.dataset_config = dataset_config
        self.n_samples = n_samples
        self.past_states = np.arange(n_samples * 2).reshape(n_samples, 2).astype(np.float32)
        self.past_actions = np.ones((n_samples, 1), dtype=np.float32)
        # Make future_states a constant array (e.g. all 5’s); shape (n_samples, 1, 2)
        self.future_states = np.full((n_samples, 1, 2), 5, dtype=np.float32)
        self.extra = np.zeros((n_samples,), dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (self.past_states[index],
                self.past_actions[index],
                self.future_states[index],
                self.extra[index])


# --- A fixed dataset configuration used by many tests ---
dataset_config = {
    "states": {
        0: {"name": "state", "type": "numerical", "feature_begin_idx": 0, "feature_end_idx": 1},
        1: {"name": "reward", "type": "reward", "feature_begin_idx": 1, "feature_end_idx": 2},
    },
    "reward_feature_idx": 1,
    "reward_vector_idx": 0,
}


# ========= Tests for helper methods =========

def test_prepare_input_data():
    dummy_model = DummyModel(params={})
    # Create simple past_states and past_actions arrays.
    past_states = np.array([[1, 2], [3, 4]], dtype=np.float32)
    past_actions = np.array([[5], [6]], dtype=np.float32)
    result = dummy_model._prepare_input_data(past_states, past_actions)
    expected = np.concatenate([past_states, past_actions], axis=1)
    np.testing.assert_array_equal(result, expected)


def test_prepare_target_data():
    dummy_model = DummyModel(params={})
    # Create a future_states array of shape (2, 1, 2)
    future_states = np.array([[[7, 8]], [[9, 10]]], dtype=np.float32)
    result = dummy_model._prepare_target_data(future_states)
    expected = future_states.reshape(2, 2)
    np.testing.assert_array_equal(result, expected)


def test_get_target_for_feature():
    dummy_model = DummyModel(params={})
    dummy_model.dataset_config = dataset_config
    # Create a dummy target y with shape (4, 2)
    y = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]], dtype=np.float32)
    # For feature 0, target should be column 0.
    target0 = dummy_model._get_target_for_feature(y, 0)
    np.testing.assert_array_equal(target0, np.array([1, 3, 5, 7], dtype=np.float32))
    # For feature 1, target should be column 1.
    target1 = dummy_model._get_target_for_feature(y, 1)
    np.testing.assert_array_equal(target1, np.array([2, 4, 6, 8], dtype=np.float32))


# ========= Tests for fit, predict, and forward =========

def test_fit_method():
    # Create a dummy dataset with 5 samples.
    ds = DummyDataset(dataset_config, n_samples=5)
    dummy_model = DummyModel(params={"dummy": "value"})
    dummy_model.dataset_config = dataset_config
    # When fit is called, the DataLoader (with batch_size == len(dataset)) returns a batch.
    dummy_model.fit(ds)
    # Expect two estimators (one for state and one for reward)
    assert len(dummy_model.models) == 2
    # Check that both estimators have been "fitted" (i.e. their fit methods were called)
    est_state = dummy_model.models[0]
    est_reward = dummy_model.models[1]
    assert est_state.fitted_X is not None
    assert est_state.fitted_y is not None
    assert est_reward.fitted_X is not None
    assert est_reward.fitted_y is not None


def test_predict():
    dummy_model = DummyModel(params={"dummy": "value"})
    dummy_model.dataset_config = dataset_config
    # Manually set up two dummy estimators.
    # For non-reward (index 0) we use constant 10; for reward (index 1) constant 20.
    dummy_model.models = [DummyEstimator(10), DummyEstimator(20)]
    # Create an input X with arbitrary shape; here, we use 3 samples.
    X = np.zeros((3, 4), dtype=np.float32)
    result = dummy_model.predict(X)
    # The predict method works as follows:
    # 1. For index 0 (non-reward), model.predict(X) returns [10, 10, 10].
    #    So next_state becomes an array of shape (3,1) filled with 10.
    # 2. Then the reward model (index 1) is called with that next_state and returns [20, 20, 20].
    # 3. The reward (20) is inserted at column index dataset_config["reward_vector_idx"] (i.e. 0).
    # So the final predicted output should be an array of shape (3,2) where the first column is 20 and the second is 10.
    expected = np.full((3, 2), 10, dtype=np.float32)
    expected[:, 0] = 20
    np.testing.assert_array_equal(result, expected)


def test_forward():
    dummy_model = DummyModel(params={"dummy": "value"})
    dummy_model.dataset_config = dataset_config
    dummy_model.models = [DummyEstimator(10), DummyEstimator(20)]
    # Define state and action arrays.
    state = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=np.float32)
    action = np.array([[7], [8], [9]], dtype=np.float32)
    # forward calls _prepare_input_data then predict.
    result = dummy_model.forward(state, action)
    expected = np.full((3, 2), 10, dtype=np.float32)
    expected[:, 0] = 20
    np.testing.assert_array_equal(result, expected)


def test_forward_n_steps():
    dummy_model = DummyModel(params={"dummy": "value"})
    dummy_model.dataset_config = dataset_config
    dummy_model.models = [DummyEstimator(10), DummyEstimator(20)]
    # Define an initial state with batch_size 1, lookback_steps 1 and state dimension 2.
    initial_state = np.array([[[0, 0]]], dtype=np.float32)
    # Define actions with shape (batch_size, lookback_steps, action_dim). Let n_steps = 24.
    actions = np.ones((1, 24, 2), dtype=np.float32)
    result = dummy_model.forward_n_steps(initial_state, actions, n_steps=24)
    # The method returns an array of shape (batch_size, n_steps, 1, output_dim).
    # Here, output_dim is 2 (reward and non-reward).
    assert result.shape == (1, 24, 1, 2)
    # Each prediction (from forward) should be the same as in test_predict: [20, 10]
    expected_pred = np.full((1, 24, 1, 2), 10, dtype=np.float32)
    expected_pred[:, :, :, 0] = 20
    # The predictions are wrapped in a singleton dimension along axis 2.
    np.testing.assert_array_equal(result, expected_pred)


# ========= Tests for assertion errors in forward_n_steps =========

def test_forward_n_steps_invalid_n_steps():
    dummy_model = DummyModel(params={"dummy": "value"})
    with pytest.raises(AssertionError):
        dummy_model.forward_n_steps(np.zeros((2, 2)), np.zeros((2, 3)), 0)


def test_forward_n_steps_batch_mismatch():
    dummy_model = DummyModel(params={"dummy": "value"})
    # initial_state has batch size 2, actions has batch size 3.
    with pytest.raises(AssertionError):
        dummy_model.forward_n_steps(np.zeros((2, 2)), np.zeros((3, 3)), 3)


def test_forward_n_steps_action_timestep_mismatch():
    dummy_model = DummyModel(params={"dummy": "value"})
    # actions.shape[1] must equal n_steps; here we use n_steps=3 but actions have only 2 timesteps.
    with pytest.raises(AssertionError):
        dummy_model.forward_n_steps(np.zeros((2, 2)), np.zeros((2, 2)), 3)


# ========= Tests for save and load =========

def test_save_and_load(tmp_path):
    dummy_model = DummyModel(params={"dummy": "value"})
    dummy_model.dataset_config = dataset_config
    dummy_model.models = [DummyEstimator(10), DummyEstimator(20)]
    dummy_model.params = {"dummy": "value"}
    dummy_model.model_config = {"config": "test"}
    filepath = str(tmp_path / "model.pkl")
    dummy_model.save(filepath)
    
    loaded_model = DummyModel.load(filepath)
    # Check that parameters, dataset configuration, and model configuration are preserved.
    assert loaded_model.params == dummy_model.params
    assert loaded_model.dataset_config == dummy_model.dataset_config
    assert loaded_model.model_config == dummy_model.model_config
    # Check that the loaded models are functional (e.g. predict returns the same constant).
    pred0 = loaded_model.models[0].predict(np.zeros((3, 4)))
    np.testing.assert_array_equal(pred0, np.full((3,), 10, dtype=np.float32))


def test_load_model_type_mismatch(tmp_path):
    # Create a pickle file with an incorrect model type.
    filepath = str(tmp_path / "bad_model.pkl")
    bad_data = {
        "models": [],
        "dataset_config": dataset_config,
        "params": {"dummy": "value"},
        "model_type": "OtherModel",
        "model_config": None
    }
    with open(filepath, "wb") as f:
        pickle.dump(bad_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="Model type mismatch: Expected DummyModel, got OtherModel"):
        DummyModel.load(filepath)
