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

@pytest.fixture
def base_dataset_config():
    # This is a minimal configuration required by the TimeseriesDataset.
    # (Note: in your actual project you may have a more elaborate config.)
    return {
        "episode_column": "episode",
        "timestep_column": "time",
        "states": {
            0: {"name": "state1", "type": "numerical", "processor": None}
        },
        "actions": {
            0: {"name": "action1", "type": "numerical", "processor": None}
        },
        "reward": "reward"
    }


# --- Helper function to create a dummy DataFrame ---
def create_dummy_df(num_episodes=2, timesteps_per_episode=3, start_state=10, start_action=100):
    """
    Create a DataFrame with a specified number of episodes and timesteps per episode.
    The DataFrame includes columns for episode, time, one state column, one reward column,
    and one action column.
    """
    data = {
        'episode': np.repeat(np.arange(num_episodes), timesteps_per_episode),
        'time': np.tile(np.arange(timesteps_per_episode), num_episodes),
        'state1': np.arange(num_episodes * timesteps_per_episode) + start_state,  # distinct values
        'reward': np.ones(num_episodes * timesteps_per_episode),
        'action1': np.arange(num_episodes * timesteps_per_episode) + start_action,
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
    # For timestep index 1, all episodes should have a 1th timestep.
    episodes_at_1 = ts_dataset.get_existing_episodes_at_timestep(1)
    np.testing.assert_array_equal(episodes_at_1, np.array([0, 1, 2]))

    # For timestep index 4, in 4-timestep episodes, the condition (episode_start_index + 4 < episode_end_index)
    # is not met (since it would be equal), so expect an empty array.
    episodes_at_3 = ts_dataset.get_existing_episodes_at_timestep(4)
    np.testing.assert_array_equal(episodes_at_3, np.array([]))


    # Remove the last 3 timesteps from the dataset.
    df = df.iloc[:-3]
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
    # For timestep index 1, the last one should not have a time step one (since it was removed).
    episodes_at_1 = ts_dataset.get_existing_episodes_at_timestep(1)
    np.testing.assert_array_equal(episodes_at_1, np.array([0, 1]))



def test_get_timestep_from_all_episodes():
    df = create_dummy_df(num_episodes=5, timesteps_per_episode=10)
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

    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(0)
    # Expect that the past state and actions are zeros
    assert np.all(past_states == 0)
    assert np.all(past_actions == 0)

    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(1)
    # With 2 episodes, we expect arrays of length 2.
    assert past_states.shape[0] == 5
    assert past_actions.shape[0] == 5
    assert future_states.shape[0] == 5
    assert future_actions.shape[0] == 5

    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(9)

def test_get_timestep_from_all_episodes_consistent_episodes(base_dataset_config):
    num_episodes = 5
    timesteps_per_episode = 10
    # Create a dummy dataframe with uniform episode lengths.
    df = create_dummy_df(num_episodes=num_episodes, timesteps_per_episode=timesteps_per_episode)
    
    # Create the dataset. Note: we pass train_processors=False because our processors are None.
    ts_dataset = TimeseriesDataset(
        df,
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
    
    # --- Test 1: timestep index 0 ---
    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(0)
    
    # For the very first timestep of every episode, no historical data is available.
    # The _pad_past_data method should add a full 2-row padding (of zeros).
    assert past_states.shape == (num_episodes, 2, ts_dataset.states.shape[1])
    assert np.all(past_states == 0), "At timestep 0, past states must be padded with zeros."
    assert past_actions.shape == (num_episodes, 2, ts_dataset.actions.shape[1])
    assert np.all(past_actions == 0), "At timestep 0, past actions must be padded with zeros."
    
    # For future data, since forecast_timesteps == 2, each episode should provide the first two actual rows.
    for ep in range(num_episodes):
        global_start = ts_dataset.episode_start_index[ep]
        expected_first = ts_dataset.states[global_start]
        expected_second = ts_dataset.states[global_start + 1]
        np.testing.assert_array_equal(future_states[ep, 0], expected_first)
        np.testing.assert_array_equal(future_states[ep, 1], expected_second)
    assert future_states.shape == (num_episodes, 2, ts_dataset.states.shape[1])
    assert future_actions.shape == (num_episodes, 2, ts_dataset.actions.shape[1])
    
    # --- Test 2: timestep index 1 ---
    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(1)
    
    # For a nonzero timestep the _get_past_data method will select the available rows
    # (which may be fewer than the lookback window) and then pad at the beginning.
    for ep in range(num_episodes):
        # For each episode, index = episode_start_index + 1.
        current_idx = ts_dataset.episode_start_index[ep] + 1
        # Without padding, the past data would be only one row (the one at global index 0).
        # After padding to 2 rows, we expect:
        #   Row 0: zeros (padding)
        #   Row 1: state at global index = episode_start_index (i.e. first actual row)
        expected_past = np.vstack([
            np.zeros(ts_dataset.states.shape[1]),
            ts_dataset.states[ts_dataset.episode_start_index[ep]]
        ])
        np.testing.assert_array_equal(past_states[ep], expected_past)
        
        # Future data should be the rows at indices 1 and 2.
        expected_future = np.vstack([
            ts_dataset.states[current_idx],
            ts_dataset.states[current_idx + 1]
        ])
        np.testing.assert_array_equal(future_states[ep], expected_future)
    
    # --- Test 3: timestep index at the end of the episode ---
    # When the index is at the very end, the future data won’t be complete and padding will occur.
    last_timestep = timesteps_per_episode - 1  # For example, 9 when there are 10 timesteps.
    past_states, past_actions, future_states, future_actions = ts_dataset.get_timestep_from_all_episodes(last_timestep)
    
    for ep in range(num_episodes):
        # For each episode, current global index = episode_start_index + last_timestep.
        current_idx = ts_dataset.episode_start_index[ep] + last_timestep
        # Past data: available rows are from current_idx - lookback_timesteps to current_idx.
        # In our uniform episodes this should be exactly two rows (no padding needed).
        expected_past = ts_dataset.states[current_idx - 2: current_idx]
        np.testing.assert_array_equal(past_states[ep], expected_past)
        
        # Future data: only one row is available (the row at current_idx),
        # so the _pad_future_data function should add one row of zeros.
        expected_future = ts_dataset.states[current_idx: current_idx + 1]
        np.testing.assert_array_equal(future_states[ep, 0], expected_future[0])
        np.testing.assert_array_equal(future_states[ep, 1], np.zeros(ts_dataset.states.shape[1]))
    
    # --- Test 4: Episodes with varying lengths ---
    # Create a new dataframe where the episodes have different numbers of timesteps.
    data1 = pd.DataFrame({
        "episode": [0] * 5,
        "time": np.arange(5),
        "state1": np.arange(5),
        "reward": np.ones(5),
        "action1": np.arange(5, 10)
    })
    data2 = pd.DataFrame({
        "episode": [1] * 8,
        "time": np.arange(8),
        "state1": np.arange(10, 18),
        "reward": np.ones(8),
        "action1": np.arange(18, 26)
    })
    data3 = pd.DataFrame({
        "episode": [2] * 3,
        "time": np.arange(3),
        "state1": np.arange(26, 29),
        "reward": np.ones(3),
        "action1": np.arange(29, 32)
    })
    df_mixed = pd.concat([data1, data2, data3], ignore_index=True)
    
    ts_dataset_mixed = TimeseriesDataset(
        df_mixed,
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
    
    # For timestep index 0, every episode has a valid timestep.
    past_states, past_actions, future_states, future_actions = ts_dataset_mixed.get_timestep_from_all_episodes(0)
    assert past_states.shape[0] == 3, "All three episodes should be returned at timestep 0."
    
    # For timestep index 4, only episodes with at least 5 timesteps are valid.
    past_states, past_actions, future_states, future_actions = ts_dataset_mixed.get_timestep_from_all_episodes(4)
    # In our mixed dataset, only episode 0 (length 5) and episode 1 (length 8) have a timestep 4.
    assert past_states.shape[0] == 2, "Only episodes with at least 5 timesteps should be returned for timestep 4."
    
    # --- Test 5: Testing the inference branch ---
    # When is_inference == True, the past-data is handled slightly differently.
    ts_dataset_inf = TimeseriesDataset(
        df,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1'],
        lookback_timesteps=2,
        forecast_timesteps=2,
        train_processors=True,
        is_inference=True,
        noise_intensity_on_past_states=0.0,
    )
    
    # For a timestep of 1 in inference mode, the _get_past_data method adds 1 to the past indices.
    # For each episode, for index = episode_start_index + 1:
    #   - Without inference, past data would have been the row at episode_start_index (padded).
    #   - With inference, past data becomes self.states[episode_start_index + 1 : 1+1] (i.e. one row),
    #     but because of the adjustment it is still padded to a 2-row array.
    past_states_inf, past_actions_inf, future_states_inf, future_actions_inf = ts_dataset_inf.get_timestep_from_all_episodes(1)
    for ep in range(num_episodes):
        # For each episode, the effective index is episode_start_index + 1.
        expected = ts_dataset.states[ts_dataset.episode_start_index[ep] + 1]
        # In inference mode, the padded past data is shifted so that the single available row is at the bottom.
        np.testing.assert_array_equal(past_states_inf[ep, 1], expected)
        np.testing.assert_array_equal(past_states_inf[ep, 0], np.zeros(ts_dataset.states.shape[1]))


def test_get_timestep_from_all_episodes_dataframe_directly(base_dataset_config):
    """
    This test verifies that get_timestep_from_all_episodes returns the expected past and future arrays,
    based directly on the DataFrame’s values. We compute, for each valid timestep offset in an episode,
    the expected arrays using the same lookback and forecast logic (including zero padding), and then compare
    against the dataset output.
    """
    # Settings
    lookback_timesteps = 2
    forecast_timesteps = 2
    timesteps_per_episode = 8
    num_episodes=3
    data1 = pd.DataFrame({
        "episode": [0] * 5,
        "time": np.arange(5),
        "state1": np.arange(5),
        "reward": np.ones(5),
        "action1": np.arange(5, 10)
    })
    data2 = pd.DataFrame({
        "episode": [1] * 8,
        "time": np.arange(8),
        "state1": np.arange(10, 18),
        "reward": np.ones(8),
        "action1": np.arange(18, 26)
    })
    data3 = pd.DataFrame({
        "episode": [2] * 3,
        "time": np.arange(3),
        "state1": np.arange(26, 29),
        "reward": np.ones(3),
        "action1": np.arange(29, 32)
    })

    # Combine the dataframes.
    df_mixed = pd.concat([data1, data2, data3], ignore_index=True)
    
    # Instantiate the dataset with mixed episodes.
    ts_dataset = TimeseriesDataset(
        df_mixed,
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


    # Loop over several timestep offsets (within an episode) and verify the output.
    # get_timestep_from_all_episodes uses an offset relative to the start of the episode.
    for offset in range(timesteps_per_episode):
        # For each episode, only those with enough timesteps (episode_start + offset < episode_end)
        # will be considered. For each valid episode, we calculate the expected past and future arrays.
        expected_past_states_list = []
        expected_past_actions_list = []
        expected_future_states_list = []
        expected_future_actions_list = []

        # Process each episode individually.
        for ep in range(num_episodes):
            ep_start = ts_dataset.episode_start_index[ep]
            ep_end = ts_dataset.episode_end_index[ep]
            if ep_start + offset < ep_end:
                global_idx = ep_start + offset

                # ---- Compute expected past values ----
                # The dataset uses _get_past_data:
                #   past_start = max(episode_start, global_idx - lookback_timesteps)
                #   past_data = self.states[past_start:global_idx]
                past_start = max(ep_start, global_idx - lookback_timesteps)
                raw_past_states = ts_dataset.states[past_start:global_idx]
                raw_past_actions = ts_dataset.actions[past_start:global_idx]

                # If there are fewer than lookback_timesteps rows, pad on the top with zeros.
                num_missing = lookback_timesteps - raw_past_states.shape[0]
                if num_missing > 0:
                    pad_states = np.zeros((num_missing, ts_dataset.states.shape[1]))
                    pad_actions = np.zeros((num_missing, ts_dataset.actions.shape[1]))
                    exp_past_states = np.vstack([pad_states, raw_past_states])
                    exp_past_actions = np.vstack([pad_actions, raw_past_actions])
                else:
                    exp_past_states = raw_past_states
                    exp_past_actions = raw_past_actions

                # ---- Compute expected future values ----
                # The dataset uses _get_future_data:
                #   future_end = min(ep_end + 1, global_idx + forecast_timesteps)
                #   future_data = self.states[global_idx:future_end]
                effective_end = ts_dataset.timestep_end_index[global_idx]  # This equals ep_end - 1
                future_end = min(effective_end + 1, global_idx + forecast_timesteps)

                raw_future_states = ts_dataset.states[global_idx:future_end]
                raw_future_actions = ts_dataset.actions[global_idx:future_end]
                num_future_missing = forecast_timesteps - raw_future_states.shape[0]
                if num_future_missing > 0:
                    pad_future_states = np.zeros((num_future_missing, ts_dataset.states.shape[1]))
                    pad_future_actions = np.zeros((num_future_missing, ts_dataset.actions.shape[1]))
                    exp_future_states = np.vstack([raw_future_states, pad_future_states])
                    exp_future_actions = np.vstack([raw_future_actions, pad_future_actions])
                else:
                    exp_future_states = raw_future_states
                    exp_future_actions = raw_future_actions

                # Append these expected values.
                expected_past_states_list.append(exp_past_states)
                expected_past_actions_list.append(exp_past_actions)
                expected_future_states_list.append(exp_future_states)
                expected_future_actions_list.append(exp_future_actions)

        # Convert lists to numpy arrays.
        expected_past_states = np.array(expected_past_states_list)
        expected_past_actions = np.array(expected_past_actions_list)
        expected_future_states = np.array(expected_future_states_list)
        expected_future_actions = np.array(expected_future_actions_list)

        # Call the dataset method for this offset.
        ds_past_states, ds_past_actions, ds_future_states, ds_future_actions = ts_dataset.get_timestep_from_all_episodes(offset)

        # Check that shapes match.
        assert ds_past_states.shape == expected_past_states.shape, \
            f"Past states shape mismatch at offset {offset}"
        assert ds_past_actions.shape == expected_past_actions.shape, \
            f"Past actions shape mismatch at offset {offset}"
        assert ds_future_states.shape == expected_future_states.shape, \
            f"Future states shape mismatch at offset {offset}"
        assert ds_future_actions.shape == expected_future_actions.shape, \
            f"Future actions shape mismatch at offset {offset}"

        # Verify that the actual arrays match the expected ones.
        np.testing.assert_array_equal(ds_past_states, expected_past_states,
                                        err_msg=f"Mismatch in past states for offset {offset}")
        np.testing.assert_array_equal(ds_past_actions, expected_past_actions,
                                        err_msg=f"Mismatch in past actions for offset {offset}")
        np.testing.assert_array_equal(ds_future_states, expected_future_states,
                                        err_msg=f"Mismatch in future states for offset {offset}")
        np.testing.assert_array_equal(ds_future_actions, expected_future_actions,
                                        err_msg=f"Mismatch in future actions for offset {offset}")

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

