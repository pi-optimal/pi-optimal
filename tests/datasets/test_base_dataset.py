# test_base_dataset.py
import numpy as np
import pandas as pd
import pytest
from pi_optimal.datasets.base_dataset import BaseDataset

# --- Tests for _infer_column_properties ---

def _dummy_instance():
    # Create a minimal valid DataFrame and config so that we can instantiate BaseDataset.
    df = pd.DataFrame({
        'episode': [0],
        'timestep': [0],
        'reward': [0]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'timestep',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    return BaseDataset(df, dataset_config=config)

def test_infer_column_properties_numeric():
    ds = _dummy_instance()
    col_type, processor, eval_metric = ds._infer_column_properties(np.dtype('float64'))
    assert col_type == "numerical"
    assert processor == {"name": "StandardScaler", "params": {}}
    assert eval_metric == "mae"

def test_infer_column_properties_categorical_object():
    ds = _dummy_instance()
    # Using object dtype
    col_type, processor, eval_metric = ds._infer_column_properties(np.dtype('O'))
    assert col_type == "categorial"
    assert processor == {"name": "OrdinalEncoder"}
    assert eval_metric == "accuracy"

def test_infer_column_properties_categorical_category():
    ds = _dummy_instance()
    # Create a pandas CategoricalDtype for testing
    cat_dtype = pd.CategoricalDtype()
    col_type, processor, eval_metric = ds._infer_column_properties(cat_dtype)
    assert col_type == "categorial"
    assert processor == {"name": "OrdinalEncoder"}
    assert eval_metric == "accuracy"

def test_infer_column_properties_boolean():
    ds = _dummy_instance()
    col_type, processor, eval_metric = ds._infer_column_properties(np.dtype('bool'))
    assert col_type == "binary"
    assert processor is None
    assert eval_metric == "f1_binary"

def test_infer_column_properties_unknown():
    ds = _dummy_instance()
    # datetime64 is not caught by any of the above conditions in _infer_column_properties.
    col_type, processor, eval_metric = ds._infer_column_properties(np.dtype('datetime64[ns]'))
    assert col_type == "unknown"
    assert processor is None
    assert eval_metric is None

# --- Test _create_dataset_config via __init__ with dataset_config=None ---

def test_create_dataset_config():
    data = {
        'unit': [0, 0, 1, 1],
        'time': [0, 1, 0, 1],
        'state1': [1.0, 2.0, 3.0, 4.0],
        'reward': [10, 20, 30, 40],
        'action1': ['a', 'b', 'a', 'b']
    }
    df = pd.DataFrame(data)
    # Pass dataset_config as None so that __init__ calls _create_dataset_config.
    ds = BaseDataset(
        df,
        dataset_config=None,
        unit_index='unit',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=['action1']
    )
    config = ds.dataset_config
    assert config['episode_column'] == 'unit'
    assert config['timestep_column'] == 'time'
    # In states, the first state is from state_columns and the next from the reward column.
    assert 0 in config['states'] and 1 in config['states']
    assert config['states'][0]['name'] == 'state1'
    assert config['states'][1]['name'] == 'reward'
    # Check actions
    assert 0 in config['actions']
    assert config['actions'][0]['name'] == 'action1'
    # Reward column is set correctly.
    assert config['reward_column'] == 'reward'

# --- Tests for _validate_input and __init__ error conditions ---

def test_validate_input_empty_dataframe():
    df = pd.DataFrame()
    config = {
        'episode_column': 'episode',
        'timestep_column': 'timestep',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    with pytest.raises(ValueError, match="Input dataframe must not be empty."):
        BaseDataset(df, dataset_config=config)

def test_validate_input_missing_episode():
    df = pd.DataFrame({'time': [0, 1]})
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    with pytest.raises(ValueError, match="Episode column 'episode' not found in dataframe."):
        BaseDataset(df, dataset_config=config)

def test_validate_input_missing_timestep():
    df = pd.DataFrame({'episode': [0, 0]})
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    with pytest.raises(ValueError, match="Timestep column 'time' not found in dataframe."):
        BaseDataset(df, dataset_config=config)

def test_validate_input_convert_episode_to_int():
    # Episode column is float but convertible.
    df = pd.DataFrame({
        'episode': [0.0, 0.0, 1.0, 1.0],
        'time': [0, 1, 0, 1],
        'reward': [10, 20, 30, 40]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    # Verify that the episode column has been converted to integer type.
    assert pd.api.types.is_integer_dtype(ds.df['episode'])

def test_validate_input_convert_episode_to_int_fail():
    # Episode column contains non-numeric values.
    df = pd.DataFrame({
        'episode': ['a', 'b'],
        'time': [0, 1],
        'reward': [10, 20]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    with pytest.raises(ValueError, match="Episode column 'episode' must be of integer type."):
        BaseDataset(df, dataset_config=config)

def test_adjust_episode_numbers():
    # Episodes are not continuous: they are [1, 1, 3, 3] and should be remapped to [0, 0, 1, 1]
    df = pd.DataFrame({
        'episode': [1, 1, 3, 3],
        'time': [0, 1, 0, 1],
        'reward': [10, 20, 30, 40]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    unique_episodes = np.sort(ds.df['episode'].unique())
    np.testing.assert_array_equal(unique_episodes, np.array([0, 1]))

def test_invalid_timestep_dtype():
    # Timestep column is float (and not integer or datetime) so should raise ValueError.
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0.1, 1.2],
        'reward': [10, 20]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    with pytest.raises(ValueError, match="Timestep column 'time' must be of integer or datetime type."):
        BaseDataset(df, dataset_config=config)

# --- Tests for timestep adjustment ---

def test_adjust_datetime_timesteps():
    # Timestep column is datetime and should be converted to sequential integers.
    times = pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00'])
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': times,
        'reward': [10, 20]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    # The timesteps should have been replaced by 0 and 1.
    assert (ds.df['time'].values == np.array([0, 1])).all()

def test_adjust_noncontinuous_integer_timesteps():
    # Timestep column is integer but does not start at 0; should be adjusted to [0, 1]
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [5, 6],
        'reward': [10, 20]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    np.testing.assert_array_equal(ds.df['time'].values, np.array([0, 1]))

def test_adjust_noncontinuous_integer_timesteps_multiple_episodes():
    # Each episode's timesteps should be adjusted independently.
    df = pd.DataFrame({
        'episode': [0, 0, 1, 1],
        'time': [5, 6, 10, 12],
        'reward': [10, 20, 30, 40]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    # For both episodes the timesteps should become [0, 1].
    np.testing.assert_array_equal(ds.df['time'].values, np.array([0, 1, 0, 1]))

# --- Test for _setup_dataset (sorting, index reset, num_episodes) ---

def test_setup_dataset_sorting_and_index_reset():
    # Create a DataFrame with unsorted rows.
    df = pd.DataFrame({
        'episode': [1, 0, 1, 0],
        'time': [1, 1, 0, 0],
        'reward': [10, 20, 30, 40]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    # The DataFrame should be sorted by episode and time, and the index reset.
    sorted_df = ds.df
    expected_df = sorted_df.sort_values(by=['episode', 'time']).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_df, expected_df)
    # The number of episodes should equal the number of unique episode numbers.
    assert ds.num_episodes == len(sorted_df['episode'].unique())

# --- Tests for _infer_action_type ---

def test_infer_action_type_numerical():
    # Create a DataFrame with a numerical action column.
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1],
        'reward': [10, 20],
        'action1': [0.1, 0.2]
    })
    # Using dataset_config=None so that _create_dataset_config builds the config from the provided column names.
    ds = BaseDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=[],
        action_columns=['action1']
    )
    # All actions are numerical so the type should be "mpc-continuous".
    assert ds.action_type == "mpc-continuous"

def test_infer_action_type_categorial():
    # Create a DataFrame with a categorial action column.
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1],
        'reward': [10, 20],
        'action1': ['a', 'b']
    })
    ds = BaseDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=[],
        action_columns=['action1']
    )
    # All actions are categorial so the type should be "mpc-discrete".
    assert ds.action_type == "mpc-discrete"

def test_infer_action_type_mixed():
    # Create a DataFrame with one numerical and one categorial action.
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1],
        'reward': [10, 20],
        'action1': [0.1, 0.2],
        'action2': ['a', 'b']
    })
    ds = BaseDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=[],
        action_columns=['action1', 'action2']
    )
    # When action types are mixed the default is "mpc-continuous".
    assert ds.action_type == "mpc-continuous"

def test_infer_action_type_no_actions():
    # If no action columns are provided, the action_types set is empty.
    # Note: all([]) returns True in Python, so the default becomes "mpc-continuous".
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1],
        'reward': [10, 20],
        'state1': [1.0, 2.0]
    })
    ds = BaseDataset(
        df,
        dataset_config=None,
        unit_index='episode',
        timestep_column='time',
        reward_column='reward',
        state_columns=['state1'],
        action_columns=[]
    )
    assert ds.action_type == "mpc-continuous"

# --- Test __len__ ---

def test_len():
    df = pd.DataFrame({
        'episode': [0, 0, 1, 1],
        'time': [0, 1, 0, 1],
        'reward': [10, 20, 30, 40]
    })
    config = {
        'episode_column': 'episode',
        'timestep_column': 'time',
        'states': {},
        'actions': {},
        'reward_column': 'reward'
    }
    ds = BaseDataset(df, dataset_config=config)
    assert len(ds) == 4

# --- Test missing reward column in _create_dataset_config (KeyError) ---

def test_missing_reward_column_in_create_config():
    # If the reward column is missing from the DataFrame, _create_dataset_config will raise a KeyError.
    df = pd.DataFrame({
        'episode': [0, 0],
        'time': [0, 1]
    })
    with pytest.raises(KeyError):
        BaseDataset(
            df,
            dataset_config=None,
            unit_index='episode',
            timestep_column='time',
            reward_column='reward',
            state_columns=[],
            action_columns=[]
        )
