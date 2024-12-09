import pytest
import pandas as pd
import numpy as np
from pi_optimal.datasets.timeseries_dataset import (
    TimeseriesDataset,
)  # Replace 'your_module' with the actual module name


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "episode": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            "timestep": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2],
            "state_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "state_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "action_1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "action_2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def simple_config():
    return {
        "episode_column": "episode",
        "timestep_column": "timestep",
        "states": {
            0: {"name": "state_1", "processor": None},
        },
        "actions": {
            0: {"name": "action_1", "processor": None},
        },
    }


@pytest.fixture
def complex_config():
    return {
        "episode_column": "episode",
        "timestep_column": "timestep",
        "states": {
            0: {"name": "state_1", "processor": None},
            1: {"name": "state_2", "processor": None},
        },
        "actions": {
            0: {"name": "action_1", "processor": None},
            1: {"name": "action_2", "processor": None},
        },
    }


class TestTimeseriesDatasets:

    def test_simple_dataset(self, test_df, simple_config):
        dataset = TimeseriesDataset(
            test_df, simple_config, lookback_timesteps=2, forecast_timesteps=1
        )

        assert dataset.states.shape == (10, 1)
        assert dataset.actions.shape == (10, 1)

        past_states, past_actions, future_states, future_actions = dataset[3]

        assert past_states.shape == (2, 1)
        assert past_actions.shape == (2, 1)
        assert future_states.shape == (1, 1)
        assert future_actions.shape == (1, 1)

    def test_complex_dataset(self, test_df, complex_config):
        dataset = TimeseriesDataset(
            test_df, complex_config, lookback_timesteps=3, forecast_timesteps=2
        )

        assert dataset.states.shape == (10, 2)
        assert dataset.actions.shape == (10, 2)

        past_states, past_actions, future_states, future_actions = dataset[5]

        assert past_states.shape == (3, 2)
        assert past_actions.shape == (3, 2)
        assert future_states.shape == (2, 2)
        assert future_actions.shape == (2, 2)

    def test_episode_lengths(self, test_df, complex_config):
        dataset = TimeseriesDataset(test_df, complex_config)

        assert dataset.min_episode_length == 3
        assert dataset.max_episode_length == 4
        assert dataset.median_episode_length == 3

    def test_get_episode_complex(self, test_df, complex_config):
        dataset = TimeseriesDataset(
            test_df, complex_config, lookback_timesteps=2, forecast_timesteps=1
        )
        past_states, past_actions, future_states, future_actions = dataset.get_episode(
            1
        )

        assert len(past_states) == 4
        assert len(past_actions) == 4
        assert len(future_states) == 4
        assert len(future_actions) == 4

        assert past_states[0].shape == (2, 2)
        assert past_actions[0].shape == (2, 2)
        assert future_states[0].shape == (1, 2)
        assert future_actions[0].shape == (1, 2)

    def test_get_all_episodes_complex(self, test_df, complex_config):
        dataset = TimeseriesDataset(
            test_df, complex_config, lookback_timesteps=2, forecast_timesteps=1
        )
        episodes = dataset.get_all_episodes()
        all_states, all_actions, all_next_states, all_next_actions = episodes

        assert len(all_states) == 3
        assert len(all_states[0]) == 3
        assert len(all_states[1]) == 4
        assert len(all_states[2]) == 3

        assert all_states[0][0].shape == (2, 2)
        assert all_states[1][0].shape == (2, 2)
        assert all_states[2][0].shape == (2, 2)

        assert len(all_actions) == 3
        assert len(all_actions[0]) == 3
        assert len(all_actions[1]) == 4
        assert len(all_actions[2]) == 3

        assert all_actions[0][0].shape == (2, 2)
        assert all_actions[1][0].shape == (2, 2)
        assert all_actions[2][0].shape == (2, 2)

        assert len(all_next_states) == 3
        assert len(all_next_states[0]) == 3
        assert len(all_next_states[1]) == 4
        assert len(all_next_states[2]) == 3

        assert all_next_states[0][0].shape == (1, 2)
        assert all_next_states[1][0].shape == (1, 2)
        assert all_next_states[2][0].shape == (1, 2)

        assert len(all_next_actions) == 3
        assert len(all_next_actions[0]) == 3
        assert len(all_next_actions[1]) == 4
        assert len(all_next_actions[2]) == 3

        assert all_next_actions[0][0].shape == (1, 2)
        assert all_next_actions[1][0].shape == (1, 2)
        assert all_next_actions[2][0].shape == (1, 2)


    @pytest.mark.parametrize("lookback,forecast", [(1, 1), (2, 2), (3, 1), (1, 3)])
    def test_various_configs(self, test_df, complex_config, lookback, forecast):
        dataset = TimeseriesDataset(
            test_df,
            complex_config,
            lookback_timesteps=lookback,
            forecast_timesteps=forecast,
        )
        past_states, past_actions, future_states, future_actions = dataset[5]

        assert past_states.shape == (lookback, 2)
        assert past_actions.shape == (lookback, 2)
        assert future_states.shape == (forecast, 2)
        assert future_actions.shape == (forecast, 2)

    def test_padding_complex(self, test_df, complex_config):
        dataset = TimeseriesDataset(
            test_df, complex_config, lookback_timesteps=5, forecast_timesteps=3
        )
        past_states, past_actions, future_states, future_actions = dataset[0]

        assert past_states.shape == (5, 2)
        assert past_actions.shape == (5, 2)
        assert future_states.shape == (3, 2)
        assert future_actions.shape == (3, 2)

        # Check if padding is applied correctly (zeros for padded values)
        assert np.all(past_states[0:5] == 0)
        assert np.all(past_actions[0:5] == 0)

        np.testing.assert_array_equal(
            future_states, np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        )

        np.testing.assert_array_equal(
            future_actions, np.array([[0, 1], [1, 0], [0, 1]])
        )

    def test_different_episode_lengths(self, test_df, complex_config):
        dataset = TimeseriesDataset(
            test_df, complex_config, lookback_timesteps=2, forecast_timesteps=2
        )

        # Test for episode 0 (length 3)
        past_states_0, past_actions_0, future_states_0, future_actions_0 = dataset[2]
        assert past_states_0.shape == (2, 2)
        assert future_states_0.shape == (
            2,
            2,
        )  # Only 1 future step available, but with padding lead to two
        assert past_actions_0.shape == (2, 2)
        assert future_actions_0.shape == (
            2,
            2,
        )  # Only 1 future step available, but with padding lead to two

        np.testing.assert_array_equal(past_states_0, np.array([[1, 0.1], [2, 0.2]]))

        np.testing.assert_array_equal(past_actions_0, np.array([[0, 1], [1, 0]]))

        np.testing.assert_array_equal(future_states_0, np.array([[3, 0.3], [0, 0]]))

        np.testing.assert_array_equal(future_actions_0, np.array([[0, 1], [0, 0]]))

        # Test for episode 1 (length 4)
        past_states_1, past_actions_1, future_states_1, future_actions_1 = dataset[8]
        assert past_states_1.shape == (2, 2)
        assert future_states_1.shape == (2, 2)
        assert past_actions_1.shape == (2, 2)
        assert future_actions_1.shape == (2, 2)

        np.testing.assert_array_equal(past_states_1, np.array([[0, 0], [8, 0.8]]))

        np.testing.assert_array_equal(past_actions_1, np.array([[0, 0], [1, 0]]))

        np.testing.assert_array_equal(future_states_1, np.array([[9, 0.9], [10, 1]]))

        np.testing.assert_array_equal(future_actions_1, np.array([[0, 1], [1, 0]]))
