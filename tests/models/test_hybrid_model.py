import pytest
from pi_optimal.models.sklearn.hybrid_model import HybridModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

# --- Tests for _validate_hybrid_params ---

def test_validate_hybrid_params_valid():
    valid_params = {
        0: {"type": "mlp", "params": {"hidden_layer_sizes": (32,), "learning_rate_init": 0.01}},
        1: {"type": "rf", "params": {"n_estimators": 20, "max_depth": 10}},
    }
    # Should return True without raising an error.
    assert HybridModel._validate_hybrid_params(valid_params) is True

def test_validate_hybrid_params_invalid_key():
    # Use a string as a key instead of an int.
    invalid_params = {
        "0": {"type": "mlp", "params": {"hidden_layer_sizes": (32,), "learning_rate_init": 0.01}},
    }
    with pytest.raises(ValueError, match="must be integers"):
        HybridModel._validate_hybrid_params(invalid_params)

def test_validate_hybrid_params_missing_type():
    # Missing "type" key for state index 0.
    invalid_params = {
        0: {"params": {"hidden_layer_sizes": (32,), "learning_rate_init": 0.01}},
    }
    with pytest.raises(ValueError, match="Missing 'type' key"):
        HybridModel._validate_hybrid_params(invalid_params)

def test_validate_hybrid_params_missing_params_key():
    # Missing "params" key for state index 0.
    invalid_params = {
        0: {"type": "mlp"},
    }
    with pytest.raises(ValueError, match="Missing 'params' key"):
        HybridModel._validate_hybrid_params(invalid_params)

# --- Tests for _create_estimator ---

def test_create_estimator_numerical_mlp():
    # Setup a configuration for a numerical feature with an MLP regressor.
    params = {
        0: {"type": "mlp", "params": {"hidden_layer_sizes": (32,), "learning_rate_init": 0.01}}
    }
    state_configs = {
        0: {"name": "state", "type": "numerical", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    hybrid_model = HybridModel(params)
    estimator = hybrid_model._create_estimator(state_configs, 0)
    assert isinstance(estimator, MLPRegressor)
    # Verify that estimator parameters are set as given.
    assert estimator.hidden_layer_sizes == (32,)
    assert estimator.learning_rate_init == 0.01

def test_create_estimator_numerical_rf():
    # Setup a configuration for a numerical feature with a Random Forest regressor.
    params = {
        0: {"type": "rf", "params": {"n_estimators": 20, "max_depth": 10}}
    }
    state_configs = {
        0: {"name": "state", "type": "numerical", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    hybrid_model = HybridModel(params)
    estimator = hybrid_model._create_estimator(state_configs, 0)
    assert isinstance(estimator, RandomForestRegressor)
    assert estimator.n_estimators == 20
    assert estimator.max_depth == 10

def test_create_estimator_categorial_rf():
    # Setup a configuration for a categorial feature with a Random Forest classifier.
    params = {
        0: {"type": "rf", "params": {"n_estimators": 30, "max_depth": 8}}
    }
    state_configs = {
        0: {"name": "state", "type": "categorial", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    hybrid_model = HybridModel(params)
    estimator = hybrid_model._create_estimator(state_configs, 0)
    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.n_estimators == 30
    assert estimator.max_depth == 8

def test_create_estimator_binary_mlp():
    # Setup a configuration for a binary feature with an MLP classifier.
    params = {
        0: {"type": "mlp", "params": {"hidden_layer_sizes": (64,), "learning_rate_init": 0.005}}
    }
    state_configs = {
        0: {"name": "binary_state", "type": "binary", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    hybrid_model = HybridModel(params)
    estimator = hybrid_model._create_estimator(state_configs, 0)
    assert isinstance(estimator, MLPClassifier)
    assert estimator.hidden_layer_sizes == (64,)
    assert estimator.learning_rate_init == 0.005

def test_create_estimator_unknown_feature_type():
    # Setup a configuration with an unsupported feature type.
    params = {
        0: {"type": "mlp", "params": {"hidden_layer_sizes": (32,), "learning_rate_init": 0.01}}
    }
    state_configs = {
        0: {"name": "state", "type": "unknown", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    hybrid_model = HybridModel(params)
    with pytest.raises(ValueError, match="Unknown feature type"):
        hybrid_model._create_estimator(state_configs, 0)

def test_create_estimator_unsupported_model_type():
    # Setup a configuration with a model type that is not supported for numerical features.
    params = {
        0: {"type": "unsupported", "params": {"param": 123}}
    }
    state_configs = {
        0: {"name": "state", "type": "numerical", "feature_begin_idx": 0, "feature_end_idx": 1}
    }
    
    with pytest.raises(ValueError):
        hybrid_model = HybridModel(params)
