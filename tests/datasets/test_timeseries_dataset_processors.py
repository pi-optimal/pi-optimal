import pytest
import pandas as pd
import numpy as np
from pi_optimal.datasets.timeseries_dataset import (
    TimeseriesDataset,
)  # Replace 'your_module' with the actual module name
from pi_optimal.datasets.utils.processors import (
    ProcessorRegistry,
)  # Adjust import path as needed


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "episode": [0, 0, 0, 1, 1, 1, 2, 2],
            "step": [0, 1, 2, 0, 1, 2, 0, 1],
            "numerical_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "categorical_feature": ["A", "B", "A", "C", "B", "A", "C", "B"],
            "binary_feature": [0, 1, 0, 1, 1, 0, 1, 0],
            "action": [0, 1, 2, 1, 0, 2, 1, 0],
        }
    )


@pytest.fixture
def dataset_config():
    return {
        "episode_column": "episode",
        "timestep_column": "step",
        "states": {
            0: {
                "name": "numerical_feature",
                "type": "numerical",
                "processor": {"name": "StandardScaler"},
            },
            1: {
                "name": "categorical_feature",
                "type": "categorial",
                "processor": {"name": "OneHotEncoder"},
            },
            2: {"name": "binary_feature", "type": "binary", "processor": None},
        },
        "actions": {
            0: {
                "name": "action",
                "type": "categorial",
                "processor": {"name": "OrdinalEncoder"},
            },
        },
    }


def test_processor_initialization(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    assert dataset.states.shape[1] == 5  # 1 (numerical) + 3 (one-hot) + 1 (binary)
    assert dataset.actions.shape[1] == 1


def test_standard_scaler(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    numerical_feature = dataset.states[:, 0]
    assert np.isclose(numerical_feature.mean(), 0, atol=1e-7)
    assert np.isclose(numerical_feature.std(), 1, atol=1e-7)


def test_one_hot_encoder(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    one_hot_feature = dataset.states[:, 1:4]
    assert one_hot_feature.shape == (8, 3)
    assert np.all(one_hot_feature.sum(axis=1) == 1)


def test_binary_feature(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    binary_feature = dataset.states[:, 4]
    assert np.array_equal(binary_feature, sample_df["binary_feature"].values)


def test_ordinal_encoder(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    assert np.all(dataset.actions >= 0)
    assert np.all(dataset.actions < 3)


def test_get_item(sample_df, dataset_config):
    dataset = TimeseriesDataset(
        sample_df, dataset_config, lookback_timesteps=2, forecast_timesteps=1
    )
    past_states, past_actions, future_states, future_actions = dataset[3]

    assert past_states.shape == (2, 5)
    assert past_actions.shape == (2, 1)
    assert future_states.shape == (1, 5)
    assert future_actions.shape == (1, 1)


def test_get_episode(sample_df, dataset_config):
    dataset = TimeseriesDataset(
        sample_df, dataset_config, lookback_timesteps=2, forecast_timesteps=1
    )
    past_states, past_actions, future_states, future_actions = dataset.get_episode(1)

    assert len(past_states) == 3
    assert len(past_actions) == 3
    assert len(future_states) == 3
    assert len(future_actions) == 3

    assert past_states[0].shape == (2, 5)
    assert past_actions[0].shape == (2, 1)
    assert future_states[0].shape == (1, 5)
    assert future_actions[0].shape == (1, 1)


@pytest.mark.parametrize(
    "processor_name, feature_type",
    [
        ("StandardScaler", "categorical"),
        ("OneHotEncoder", "numerical"),
    ],
)
def test_incompatible_processor(
    sample_df, dataset_config, processor_name, feature_type
):
    dataset_config["states"][0]["processor"]["name"] = processor_name
    dataset_config["states"][0]["type"] = feature_type

    with pytest.raises(ValueError):
        TimeseriesDataset(sample_df, dataset_config)


def test_custom_processor(sample_df, dataset_config):
    class CustomProcessor:
        def fit(self, X):
            pass

        def transform(self, X):
            return X * 2

    ProcessorRegistry.add_processor("CustomProcessor", CustomProcessor, ["numerical"])

    dataset_config["states"][0]["processor"]["name"] = "CustomProcessor"
    dataset = TimeseriesDataset(sample_df, dataset_config)

    assert np.all(dataset.states[:, 0] == sample_df["numerical_feature"].values * 2)

    ProcessorRegistry.remove_processor("CustomProcessor")


def test_binarizer_transformation(sample_df, dataset_config):
    dataset_config["states"][0] = {
        "name": "numerical_feature",
        "type": "numerical",
        "processor": {"name": "Binarizer", "params": {"threshold": 4.5}},
    }
    dataset = TimeseriesDataset(sample_df, dataset_config)

    original_values = sample_df["numerical_feature"].values
    transformed_values = dataset.states[:, 0]

    expected_values = (original_values > 4.5).astype(int)

    assert np.array_equal(transformed_values, expected_values)
    assert set(np.unique(transformed_values)) == {0, 1}


def test_minmax_scaler_transformation(sample_df, dataset_config):
    dataset_config["states"][0]["processor"] = {
        "name": "MinMaxScaler",
        "params": {"feature_range": (0, 1)},
    }
    dataset = TimeseriesDataset(sample_df, dataset_config)

    original_values = sample_df["numerical_feature"].values
    transformed_values = dataset.states[:, 0]

    expected_min = 0
    expected_max = 1

    assert np.isclose(transformed_values.min(), expected_min, atol=1e-7)
    assert np.isclose(transformed_values.max(), expected_max, atol=1e-7)
    assert np.allclose(
        transformed_values,
        (original_values - original_values.min())
        / (original_values.max() - original_values.min()),
        atol=1e-7,
    )


def test_label_encoder_transformation(sample_df, dataset_config):
    dataset_config["states"][1]["processor"] = {"name": "LabelEncoder"}
    dataset = TimeseriesDataset(sample_df, dataset_config)

    original_values = sample_df["categorical_feature"].values
    transformed_values = dataset.states[:, 1]

    unique_original = np.unique(original_values)
    unique_transformed = np.unique(transformed_values)

    assert len(unique_original) == len(unique_transformed)
    assert np.all(transformed_values >= 0)
    assert np.all(transformed_values < len(unique_original))

    # Check if the same original values are consistently encoded
    for orig_val in unique_original:
        encoded_val = transformed_values[original_values == orig_val]
        assert np.all(encoded_val == encoded_val[0])

def test_inverse_transform_features(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    # Transform the features
    transformed_states = dataset.states
    transformed_actions = dataset.actions
    
    # Inverse transform the features
    inverse_states = dataset.inverse_transform_features("states", transformed_states)
    inverse_actions = dataset.inverse_transform_features("actions", transformed_actions)
    
    numerical_feature_invert = inverse_states[:, 0].astype(np.float32)
    numerical_feature_invert_target = sample_df["numerical_feature"].values
    # Check if the inverse transformed data matches the original data
    assert np.allclose(numerical_feature_invert, numerical_feature_invert_target, atol=1e-7)
    assert np.array_equal(inverse_states[:, -1], sample_df["binary_feature"].values)
    assert np.array_equal(inverse_actions[:, 0], sample_df["action"].values)

def test_inverse_transform_standard_scaler(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    original_values = sample_df["numerical_feature"].values
    transformed_values = dataset.states[:, 0]
    inverse_transformed = dataset.inverse_transform_features("states", dataset.states)[:, 0]
    
    original_values = np.array(original_values, dtype=np.float32)
    inverse_transformed = np.array(inverse_transformed, dtype=np.float32)
    assert np.allclose(original_values, inverse_transformed, atol=1e-7)

def test_inverse_transform_one_hot_encoder(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    original_values = sample_df["categorical_feature"].values
    transformed_values = dataset.states[:, 1:4]
    inverse_transformed = dataset.inverse_transform_features("states", dataset.states)[:, 1]
    
    assert np.array_equal(original_values, inverse_transformed)

def test_inverse_transform_ordinal_encoder(sample_df, dataset_config):
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    original_values = sample_df["action"].values
    transformed_values = dataset.actions
    inverse_transformed = dataset.inverse_transform_features("actions", transformed_values)
    
    assert np.array_equal(original_values, inverse_transformed.flatten())

def test_inverse_transform_minmax_scaler(sample_df, dataset_config):
    dataset_config["states"][0]["processor"] = {
        "name": "MinMaxScaler",
        "params": {"feature_range": (0, 1)},
    }
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    original_values = sample_df["numerical_feature"].values
    transformed_values = dataset.states[:, 0]
    inverse_transformed = dataset.inverse_transform_features("states", dataset.states)[:, 0]
    
    original_values = np.array(original_values, dtype=np.float32)
    inverse_transformed = np.array(inverse_transformed, dtype=np.float32)
    assert np.allclose(original_values, inverse_transformed, atol=1e-7)

def test_inverse_transform_label_encoder(sample_df, dataset_config):
    dataset_config["states"][1]["processor"] = {"name": "LabelEncoder"}
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    original_values = sample_df["categorical_feature"].values
    transformed_values = dataset.states[:, 1]
    inverse_transformed = dataset.inverse_transform_features("states", dataset.states)[:, 1]
    
    assert np.array_equal(original_values, inverse_transformed)

def test_inverse_transform_binarizer(sample_df, dataset_config):
    dataset_config["states"][0] = {
        "name": "numerical_feature",
        "type": "numerical",
        "processor": {"name": "Binarizer", "params": {"threshold": 4.5}},
    }
    dataset = TimeseriesDataset(sample_df, dataset_config)
    
    transformed_values = dataset.states[:, 0]
    inverse_transformed = dataset.inverse_transform_features("states", dataset.states)[:, 0]
    
    assert np.array_equal(transformed_values, inverse_transformed)