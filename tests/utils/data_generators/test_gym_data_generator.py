import pytest
import pandas as pd
import numpy as np
import gymnasium as gym
from pi_optimal.utils.data_generators.gym_data_generator import (
    GymDataGenerator,
)  # Replace with actual import path


@pytest.mark.parametrize(
    "env_name,expected_columns",
    [
        (
            "CartPole-v1",
            {
                "episode",
                "step",
                "reward",
                "done",
                "state_0",
                "state_1",
                "state_2",
                "state_3",
                "action_0",
            },
        ),
        (
            "LunarLander-v2",
            {
                "episode",
                "step",
                "reward",
                "done",
                "state_0",
                "state_1",
                "state_2",
                "state_3",
                "state_4",
                "state_5",
                "state_6",
                "state_7",
                "action_0",
            },
        ),
        (
            "MountainCar-v0",
            {"episode", "step", "reward", "done", "state_0", "state_1", "action_0"},
        ),
        (
            "Pendulum-v1",
            {
                "episode",
                "step",
                "reward",
                "done",
                "state_0",
                "state_1",
                "state_2",
                "action_0",
            },
        ),
        (
            "BipedalWalker-v3",
            {
                "episode",
                "step",
                "reward",
                "done",
                "state_0",
                "state_1",
                "state_2",
                "state_3",
                "state_4",
                "state_5",
                "state_6",
                "state_7",
                "state_8",
                "state_9",
                "state_10",
                "state_11",
                "state_12",
                "state_13",
                "state_14",
                "state_15",
                "state_16",
                "state_17",
                "state_18",
                "state_19",
                "state_20",
                "state_21",
                "state_22",
                "state_23",
                "action_0",
                "action_1",
                "action_2",
                "action_3",
            },
        ),
    ],
)
class TestGymDataGenerator:

    @pytest.fixture
    def generator(self, env_name, expected_columns):
        return GymDataGenerator(env_name)

    def test_initialization(self, env_name, expected_columns, generator):
        assert isinstance(generator.env, gym.Env)
        assert generator.env.spec.id == env_name

    def test_collect_data(self, env_name, expected_columns, generator):
        data = generator.collect(n_steps=42, max_steps_per_episode=10)
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data) == 42
        assert set(data.columns) == expected_columns

    def test_data_shape(self, generator):
        data = generator.collect(n_steps=100, max_steps_per_episode=2)
        assert data["episode"].nunique() == 50
        assert data["step"].max() < 2

    def test_reward(self, generator):
        data = generator.collect(n_steps=1000, max_steps_per_episode=100)
        assert all(data["reward"].notna())

    def test_first_reward_zero(self, generator):
        data = generator.collect(n_steps=1000, max_steps_per_episode=100)
        assert all(data.groupby("episode")["reward"].first() == 0)

    def test_done(self, generator):
        data = generator.collect(n_steps=1000, max_steps_per_episode=100)
        number_of_episodes = data["episode"].nunique()
        assert (number_of_episodes - data["done"].sum()) <= 1

    def test_same_seeds(self, generator):
        data1 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=1, action_seed=1
        )
        data2 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=1, action_seed=1
        )
        pd.testing.assert_frame_equal(data1, data2)

    def test_different_seeds(self, generator):
        data1 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=1, action_seed=3
        )
        data2 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=2, action_seed=4
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(data1, data2)

    def test_different_action_seeds(self, generator):
        data1 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=1, action_seed=3
        )
        data2 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=1, action_seed=4
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(data1, data2)

    def test_different_env_seeds(self, generator):
        data1 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=2, action_seed=1
        )
        data2 = generator.collect(
            n_steps=1000, max_steps_per_episode=100, env_seed=3, action_seed=1
        )
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(data1, data2)


@pytest.mark.parametrize("env_name", ["NonexistentEnv-v0"])
def test_error_handling(env_name):
    with pytest.raises(gym.error.Error):
        GymDataGenerator(env_name)


@pytest.mark.parametrize("n_steps,max_steps_per_episode", [(0, 10), (1, 0)])
def test_invalid_parameters(n_steps, max_steps_per_episode):
    generator = GymDataGenerator("CartPole-v1")
    with pytest.raises(ValueError):
        generator.collect(n_steps=n_steps, max_steps_per_episode=max_steps_per_episode)
