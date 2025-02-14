# test_planners.py
import numpy as np
import pytest

# Import the three planner classes.
from pi_optimal.planners.cem_planner import CEMPlanner  # Abstract (used indirectly)
from pi_optimal.planners.cem_discrete import CEMDiscretePlanner
from pi_optimal.planners.cem_continuous import CEMContinuousPlanner

# ========= Dummy Forward Model and Objective Function =========

class DummyForwardModel:
    """
    A dummy model with a forward() method that returns a constant array.
    This simulates the effect of a model predicting the next state.
    """
    def forward(self, states, action_history):
        # states: (population_size, history_length, state_dim)
        # Return a constant next state: here, an array of ones with shape (population_size, state_dim)
        pop_size = states.shape[0]
        state_dim = states.shape[2]
        return np.ones((pop_size, state_dim), dtype=np.float32)

def dummy_objective(traj):
    """
    A dummy objective function that simply sums all elements in the trajectory.
    (Traj may be a 2D array representing one trajectory.)
    """
    return np.sum(traj)

# ========= Tests for Parameter Validation and Evaluation =========

@pytest.mark.parametrize("n_iter, horizon, population_size, topk, uncertainty_weight, error_msg", [
    (10, 0, 1000, 100, 0.5, "horizon must be a positive integer."),
    (10, 4, 0, 100, 0.5, "population_size must be a positive integer."),
    (10, 4, 1000, 0, 0.5, "topk must be a positive integer"),
    (10, 4, 1000, 1001, 0.5, "topk must be a positive integer and less than or equal to population_size."),
    (10, 4, 1000, 100, -0.1, "uncertainty_weight must be a float between 0 and 1."),
    (10, 4, 1000, 100, 1.5, "uncertainty_weight must be a float between 0 and 1."),
    (0, 4, 1000, 100, 0.5, "n_iter must be a positive integer.")
])
def test_validate_planing_params_invalid(n_iter, horizon, population_size, topk, uncertainty_weight, error_msg):
    planner = CEMDiscretePlanner(action_dim=3)
    with pytest.raises(ValueError, match=error_msg):
        planner.validate_planing_params(n_iter, horizon, population_size, topk, uncertainty_weight)

def test_evaluate_trajectories_discrete():
    """
    Test the evaluation function (inherited from CEMPlanner) using fabricated trajectories.
    """
    planner = CEMDiscretePlanner(action_dim=3)
    planner.population_size = 10
    planner.uncertainty_weight = 0.5
    # Fabricate ensemble trajectories for two models.
    # Each trajectory array is of shape (population_size, horizon, state_dim).
    # For example, let horizon=4 and state_dim=2.
    ensemble_trajectories = [
        np.full((10, 4, 2), 5, dtype=np.float32),
        np.full((10, 4, 2), 7, dtype=np.float32)
    ]
    total_costs, cost_contribution, uncertainty_contribution = planner.evaluate_trajectories(ensemble_trajectories, dummy_objective)
    # All returned arrays should have shape (population_size,)
    assert total_costs.shape == (10,)
    assert cost_contribution.shape == (10,)
    assert uncertainty_contribution.shape == (10,)
    # Normalized costs and uncertainty should be between 0 and 1.
    assert np.all(total_costs >= 0) and np.all(total_costs <= 1)

# ========= Tests for CEMDiscretePlanner =========

def test_cem_discrete_planner_plan():
    """
    Test the discrete planner's plan() method end-to-end.
    The discrete planner uses argmax over the generated logits.
    """
    planner = CEMDiscretePlanner(action_dim=3)
    # Define planning parameters.
    n_iter = 2
    horizon = 2
    population_size = 5
    topk = 2
    uncertainty_weight = 0.5
    reset_planer = True
    # Define a dummy starting state.
    # For example, history_length = 3 and state_dim = 2.
    starting_state = np.zeros((3, 2), dtype=np.float32)
    # Define a dummy action history.
    # For discrete planner, action dimension equals the number of discrete actions (here, 3).
    action_history = np.zeros((3, 3), dtype=np.float32)
    # Create a list of two dummy models.
    dummy_model = DummyForwardModel()
    models = [dummy_model, dummy_model]
    # Call plan() with our dummy objective.
    action_sequence = planner.plan(models, starting_state, action_history, dummy_objective,
                                   n_iter=n_iter, allow_sigma=True, horizon=horizon,
                                   population_size=population_size, topk=topk,
                                   uncertainty_weight=uncertainty_weight, reset_planer=reset_planer)
    # In discrete planner, get_action_sequence returns np.argmax(self.mu, axis=1),
    # so the output should be a 1D array of length equal to horizon.
    assert action_sequence.shape == (horizon,)
    # Also, the internal counter current_iter should equal n_iter.
    assert planner.current_iter == n_iter

# ========= Tests for CEMContinuousPlanner =========

def test_cem_continuous_planner_generate_actions_clipping():
    """
    Test that continuous planner's generate_actions() returns actions clipped
    to the provided constraints.
    """
    horizon = 3
    action_dim = 2
    constraints = {"min": np.full((horizon, action_dim), -1, dtype=np.float32),
                   "max": np.full((horizon, action_dim), 1, dtype=np.float32)}
    planner = CEMContinuousPlanner(action_dim=action_dim, constraints=constraints)
    # Set planning parameters manually.
    planner.horizon = horizon
    planner.population_size = 10
    # Set mu and sigma so that many samples fall outside the constraint.
    planner.mu = np.zeros((horizon, action_dim))
    planner.sigma = np.ones((horizon, action_dim)) * 5
    actions, actions_samples = planner.generate_actions()
    # All generated actions should be clipped between -1 and 1.
    assert np.all(actions >= -1) and np.all(actions <= 1)
    # For continuous planner, generate_actions returns the same array for actions and samples.
    np.testing.assert_array_equal(actions, actions_samples)

def test_cem_continuous_planner_plan():
    """
    Test the continuous planner's plan() method end-to-end.
    For continuous actions, get_action_sequence returns self.mu.
    """
    horizon = 2
    action_dim = 2
    constraints = {"min": np.full((horizon, action_dim), -1, dtype=np.float32),
                   "max": np.full((horizon, action_dim), 1, dtype=np.float32)}
    planner = CEMContinuousPlanner(action_dim=action_dim, constraints=constraints)
    n_iter = 2
    population_size = 5
    topk = 2
    uncertainty_weight = 0.3
    reset_planer = True
    # Define starting_state: history_length = 3, state_dim = 2.
    starting_state = np.zeros((3, 2), dtype=np.float32)
    # Define action_history: shape (3, action_dim).
    action_history = np.zeros((3, action_dim), dtype=np.float32)
    # Create a list of two dummy models.
    dummy_model = DummyForwardModel()
    models = [dummy_model, dummy_model]
    # Run planning.
    action_sequence = planner.plan(models, starting_state, action_history, dummy_objective,
                                   n_iter=n_iter, allow_sigma=True, horizon=horizon,
                                   population_size=population_size, topk=topk,
                                   uncertainty_weight=uncertainty_weight, reset_planer=reset_planer)
    # For continuous planner, get_action_sequence returns self.mu,
    # so the output shape should be (horizon, action_dim).
    assert action_sequence.shape == (horizon, action_dim)
    assert planner.current_iter == n_iter
