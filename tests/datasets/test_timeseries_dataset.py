# test_timeseries_dataset.py
import numpy as np
import pandas as pd
import pytest

from pi_optimal.datasets.timeseries_dataset import TimeseriesDataset
from pi_optimal.datasets.utils.processors import ProcessorRegistry

# --- Dummy processor and monkeypatch fixture ---

class DummyProcessor:
    """
    A dummy processor that performs an identity transform.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X):
        return self

    def transform(self, X):
        return X  # identity

    def inverse_transform(self, X):
        return X  # identity

@pytest.fixture(autouse=True)
def dummy_processor(monkeypatch):
    """
    For all tests in this file, override ProcessorRegistry.get to return DummyProcessor.
    """
    monkeypatch.setattr(ProcessorRegistry, 'get', lambda name, type, **params: DummyProcessor(**params))

# --- Helper function to create a dummy DataFrame ---

def create_dummy_df(num_episodes=2, timesteps_per_episode=3):
    """
    Create a DataFrame with a specified number of episodes and timesteps per episode.
    The DataFrame includes columns for episode, time, one state column, one reward column,
    and one action column.
    """
    data = {
        'episode': np.repeat(np.arange(num_episodes), timesteps_per_episode),
        'time': np.tile(np.arange(timesteps_per_episode), num_episodes),
        'state1': np.arange(num_episodes * timesteps_per_episode) + 10,  # distinct values
        'reward': np.ones(num_episodes * timesteps_per_episode),
        'action1': np.arange(num_episodes * timesteps_per_episode) + 100,
    }
    return pd.DataFrame(data)

# --- Tests for initialization and boundary calculations ---

def test_timeseries_dataset_initialization():
    df = create_dummy_df(num_episodes=2, timesteps_per_episode=3)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # Check episode boundaries
    expected_episode_end_index = np.array([3, 6])  # cumulative sizes: 3 and 6
    expected_episode_start_index = np.array([0, 3])
    np.testing.assert_array_equal(ts_dataset.episode_end_index, expected_episode_end_index)
    np.testing.assert_array_equal(ts_dataset.episode_start_index, expected_episode_start_index)

    # Check timestep boundaries.
    # For 2 episodes with 3 timesteps each, we expect:
    #   timestep_start_index: [0, 0, 0, 3, 3, 3]
    #   timestep_end_index:   [2, 2, 2, 5, 5, 5]
    expected_timestep_start_index = np.array([0, 0, 0, 3, 3, 3])
    expected_timestep_end_index = np.array([2, 2, 2, 5, 5, 5])
    np.testing.assert_array_equal(ts_dataset.timestep_start_index, expected_timestep_start_index)
    np.testing.assert_array_equal(ts_dataset.timestep_end_index, expected_timestep_end_index)

    # The transformed states should come from hstacking two features (state1 and reward).
    # With 6 rows, the shape should be (6, 2).
    assert ts_dataset.states.shape == (6, 2)
    # For actions, we have one column so shape should be (6, 1)
    assert ts_dataset.actions.shape == (6, 1)

    # Check episode length statistics (each episode has 3 timesteps).
    assert ts_dataset.min_episode_length == 3
    assert ts_dataset.max_episode_length == 3
    assert ts_dataset.median_episode_length == 3

# --- Tests for episode and timestep retrieval methods ---

def test_get_episode():
    df = create_dummy_df(num_episodes=2, timesteps_per_episode=3)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # Retrieve episode 0.
    past_states, past_actions, future_states, future_actions = ts_dataset.get_episode(0)
    # Since episode 0 has 3 timesteps, each list should have 3 elements.
    assert past_states.shape[0] == 3
    assert past_actions.shape[0] == 3
    assert future_states.shape[0] == 3
    assert future_actions.shape[0] == 3

    # Each past sample should be padded to (lookback_timesteps, state_feature_dim).
    for ps in past_states:
        assert ps.shape == (2, ts_dataset.states.shape[1])
    # Each future sample should be padded to (forecast_timesteps, state_feature_dim).
    for fs in future_states:
        assert fs.shape == (2, ts_dataset.states.shape[1])

def test_get_existing_episodes_at_timestep():
    df = create_dummy_df(num_episodes=3, timesteps_per_episode=4)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=1,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # For timestep index 0, all episodes should have a 0th timestep.
    episodes_at_0 = ts_dataset.get_existing_episodes_at_timestep(0)
    np.testing.assert_array_equal(episodes_at_0, np.array([0, 1, 2]))

    # For timestep index 4, in 4-timestep episodes, the condition (episode_start_index + 4 < episode_end_index)
    # is not met (since it would be equal), so expect an empty array.
    episodes_at_3 = ts_dataset.get_existing_episodes_at_timestep(4)
    np.testing.assert_array_equal(episodes_at_3, np.array([]))

def test_get_timestep_from_all_episodes():
    df = create_dummy_df(num_episodes=2, timesteps_per_episode=3)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(1)
    # With 2 episodes, we expect arrays of length 2.
    assert past_states.shape[0] == 2
    assert past_actions.shape[0] == 2
    assert future_states.shape[0] == 2
    assert future_actions.shape[0] == 2

def test_get_all_episodes():
    df = create_dummy_df(num_episodes=2, timesteps_per_episode=3)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    all_states, all_actions, all_future_states, all_future_actions = ts_dataset.get_all_episodes()
    # There should be 2 episodes.
    assert len(all_states) == 2
    # In each episode, the number of samples equals the number of timesteps in the episode (3 in this case).
    for ep in all_states:
        assert ep.shape[0] == 3

# --- Tests for feature transformation and inverse transformation ---

def test_transform_and_inverse_transform_features():
    df = create_dummy_df(num_episodes=1, timesteps_per_episode=5)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    transformed_states = ts_dataset.states
    # Inverse-transform the states.
    recovered_states = ts_dataset.inverse_transform_features("states", transformed_states)
    # The original “states” in BaseDataset are built by hstacking the state column and the reward column.
    original_state1 = df['state1'].values.reshape(-1, 1)
    original_reward = df['reward'].values.reshape(-1, 1)
    expected_states = np.hstack([original_state1, original_reward])
    np.testing.assert_array_equal(recovered_states, expected_states)

# --- Tests for __getitem__ (padding, shapes, and noise addition) ---

def test_getitem_padding_and_shapes():
    df = create_dummy_df(num_episodes=1, timesteps_per_episode=4)
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=3,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # Choose a middle index (e.g. index=2) to get a sample.
    past_states, past_actions, future_states, future_actions = ts_dataset.__getitem__(2)
    # past_states should be padded to have shape (3, state_feature_dim) 
    assert past_states.shape == (3, ts_dataset.states.shape[1])
    # future_states should be padded to have shape (2, state_feature_dim)
    assert future_states.shape == (2, ts_dataset.states.shape[1])

def test_noise_intensity_on_past_states():
    df = create_dummy_df(num_episodes=1, timesteps_per_episode=4)
    ts_dataset_no_noise = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    ts_dataset_noise = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=5.0,  # nonzero noise
    )
    # Compare the past states returned at a given index.
    ps_no_noise, _, _, _ = ts_dataset_no_noise.__getitem__(1)
    ps_noise, _, _, _ = ts_dataset_noise.__getitem__(1)
    # They should differ because of the added noise.
    assert not np.array_equal(ps_no_noise, ps_noise)

# --- Test for state value calculation ---

def test_calculate_state_values():
    # Create a DataFrame with one episode and known rewards.
    df = pd.DataFrame({
        'episode': [0, 0, 0],
        'time': [0, 1, 2],
        'state1': [10, 20, 30],
        'reward': [1, 2, 3],
        'action1': [100, 200, 300]
    })
    # Instantiate a TimeseriesDataset (its timeseries parameters are irrelevant here).
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=1,
        forecast_timesteps=1,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # Calculate discounted cumulative rewards with gamma = 0.9.
    df_with_values = ts_dataset._calculate_state_values(df.copy(), gamma=0.9)
    # For rewards [1, 2, 3]:
    #   discounted[2] = 3
    #   discounted[1] = 2 + 0.9*3 = 4.7
    #   discounted[0] = 1 + 0.9*4.7 = 5.23
    expected = np.array([5.23, 4.7, 3.0])
    np.testing.assert_allclose(df_with_values['state_value'].values, expected, rtol=1e-2)

# --- Test for future data including the current state ---

def test_future_data_includes_current_state_intended():
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1],
        'state1': [10, 20],
        'reward': [1, 1],
        'action1': [100, 200]
    })
    ts_dataset = TimeseriesDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=1,
        forecast_timesteps=1,
        train_processors=True,
        is_inference=False,
        noise_intensity_on_past_states=0.0,
    )
    # Retrieve the sample at index 0.
    _, _, future_states, _ = ts_dataset.__getitem__(0)
    # According to your intended behavior, the forecast for index 0 should be the state at index 0.
    expected_future = ts_dataset.states[0:1]  # expected slice (state at index 0)
    np.testing.assert_array_equal(
        future_states,
        expected_future,
        err_msg="Future data should include the current state as per design."
    )

