import pytest
import pandas as pd
from pi_optimal.datasets.base_dataset import BaseDataset  # Adjust import path as needed


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "episode": [0, 0, 0, 1, 1, 1, 2, 2],
            "timestep": [0, 1, 2, 0, 1, 2, 0, 1],
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


@pytest.fixture
def valid_config():
    return {"episode_column": "episode", "timestep_column": "timestep"}


class TestBaseDataset:

    def test_valid_initialization(self, valid_df, valid_config):
        dataset = BaseDataset(valid_df, valid_config)
        assert isinstance(dataset, BaseDataset)
        assert len(dataset) == 8
        assert dataset.num_episodes == 3

    def test_input_validation(self, valid_df, valid_config):
        # Test episode column type
        invalid_df = valid_df.copy()
        invalid_df["episode"] = invalid_df["episode"].astype(float)
        with pytest.raises(AssertionError, match="Episode column must be of type int"):
            BaseDataset(invalid_df, valid_config)

        # Test episode starting from 0
        invalid_df = valid_df.copy()
        invalid_df["episode"] += 1
        with pytest.raises(AssertionError, match="Episode column must start from 0"):
            BaseDataset(invalid_df, valid_config)

        # Test number of episodes
        invalid_df = valid_df.copy()
        invalid_df.loc[len(invalid_df)] = [4, 0, 6]  # Add a gap in episode numbers
        with pytest.raises(
            AssertionError,
            match="Number of episodes must be equal to the maximum episode number",
        ):
            BaseDataset(invalid_df, valid_config)

        # Test timestep column type
        invalid_df = valid_df.copy()
        invalid_df["timestep"] = invalid_df["timestep"].astype(float)
        with pytest.raises(AssertionError, match="Timestep column must be of type int"):
            BaseDataset(invalid_df, valid_config)

    def test_dataset_setup(self, valid_df, valid_config):
        dataset = BaseDataset(valid_df, valid_config)
        assert dataset.episode_column == "episode"
        assert dataset.timestep_column == "timestep"
        assert dataset.df.index.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert dataset.df["episode"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
        assert dataset.df["timestep"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1]

    def test_len_method(self, valid_df, valid_config):
        dataset = BaseDataset(valid_df, valid_config)
        assert len(dataset) == 8

        # Test with empty dataframe
        empty_df = pd.DataFrame(columns=["episode", "timestep", "value"])
        with pytest.raises(AssertionError, match="Input dataframe must not be empty"):
            empty_dataset = BaseDataset(empty_df, valid_config)

    def test_timestep_continuity(self, valid_df, valid_config):
        # Test with valid data (should pass)
        dataset = BaseDataset(valid_df, valid_config)
        assert isinstance(dataset, BaseDataset)

        # Test with gap in timesteps (should fail)
        invalid_df = valid_df.copy()
        invalid_df.loc[1, "timestep"] = 3
        with pytest.raises(
            AssertionError,
            match="Timesteps in episode 0 must start from 0 and increment by 1",
        ):
            BaseDataset(invalid_df, valid_config)

        # Test with timesteps not starting from 0 (should fail)
        invalid_df = valid_df.copy()
        invalid_df.loc[invalid_df["episode"] == 1, "timestep"] += 1
        with pytest.raises(
            AssertionError,
            match="Timesteps in episode 1 must start from 0 and increment by 1",
        ):
            BaseDataset(invalid_df, valid_config)

        # Test with non-consecutive timesteps (should fail)
        invalid_df = valid_df.copy()
        invalid_df.loc[4, "timestep"] = 2
        with pytest.raises(
            AssertionError,
            match="Timesteps in episode 1 must start from 0 and increment by 1",
        ):
            BaseDataset(invalid_df, valid_config)

        # Test with non-consecutive timesteps (should fail)
        invalid_df = valid_df.copy()
        invalid_df.loc[4, "timestep"] = 3
        with pytest.raises(
            AssertionError,
            match="Timesteps in episode 1 must start from 0 and increment by 1",
        ):
            BaseDataset(invalid_df, valid_config)

        # Test with mixed valid and invalid episodes
        invalid_df = valid_df.copy()
        new_rows = pd.DataFrame(
            {"episode": [3, 3], "timestep": [1, 2], "value": [9, 10]}
        )
        invalid_df = pd.concat([invalid_df, new_rows], ignore_index=True)
        with pytest.raises(
            AssertionError,
            match="Timesteps in episode 3 must start from 0 and increment by 1",
        ):
            BaseDataset(invalid_df, valid_config)
