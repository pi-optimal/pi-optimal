"""
Comprehensive verification script to test the static model fix across multiple scenarios
"""
import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pi_optimal.models.sklearn.static_model import StaticModel

def test_various_configurations():
    """Test static models with different configurations"""
    
    test_cases = [
        # (lookback_timesteps, states_size, test_state_idx)
        (3, 4, 1),  # 3 timesteps, 4 states, test state 1
        (5, 3, 2),  # 5 timesteps, 3 states, test state 2  
        (2, 6, 0),  # 2 timesteps, 6 states, test state 0
        (4, 5, 4),  # 4 timesteps, 5 states, test state 4 (last state)
    ]
    
    all_passed = True
    
    for lookback_timesteps, states_size, test_state_idx in test_cases:
        print(f"\n--- Testing config: {lookback_timesteps} timesteps, {states_size} states, testing state {test_state_idx} ---")
        
        # Create dataset configuration
        dataset_config = {
            'lookback_timesteps': lookback_timesteps,
            'states_size': states_size,
            'states': {}
        }
        
        for i in range(states_size):
            dataset_config['states'][i] = {
                'name': f'state_{i}',
                'feature_begin_idx': i,
                'feature_end_idx': i + 1,
                'type': 'numerical'
            }
        
        # Create mock data
        n_samples = 5
        n_actions = 2
        
        # Create past states with predictable pattern
        past_states = []
        for sample in range(n_samples):
            states_for_sample = []
            for t in range(lookback_timesteps):
                for f in range(states_size):
                    # Value = sample_id * 100 + timestep * 10 + feature_idx
                    value = sample * 100 + t * 10 + f
                    states_for_sample.append(value)
            past_states.append(states_for_sample)
        
        past_states = np.array(past_states)
        past_actions = np.random.rand(n_samples, lookback_timesteps * n_actions)
        X = np.hstack([past_states, past_actions])
        
        # Test static model
        static_model = StaticModel()
        static_model.dataset_config = dataset_config
        
        # Create estimator for the test state
        estimator = static_model._create_estimator(dataset_config['states'], test_state_idx)
        
        # Calculate expected position
        expected_position = (lookback_timesteps - 1) * states_size + test_state_idx
        
        print(f"Estimator feature_idx: {estimator.feature_idx}")
        print(f"Expected position: {expected_position}")
        
        # Fit and predict
        y = np.random.rand(n_samples)
        estimator.fit(X, y)
        predictions = estimator.predict(X)
        
        # Calculate expected predictions
        # For each sample, the value should be from the most recent timestep
        expected_predictions = []
        for sample in range(n_samples):
            expected_value = sample * 100 + (lookback_timesteps - 1) * 10 + test_state_idx
            expected_predictions.append(expected_value)
        expected_predictions = np.array(expected_predictions)
        
        print(f"Predictions: {predictions}")
        print(f"Expected: {expected_predictions}")
        
        if estimator.feature_idx == expected_position and np.allclose(predictions, expected_predictions):
            print("✅ PASS")
        else:
            print("❌ FAIL")
            all_passed = False
    
    return all_passed

def test_classifier_type():
    """Test that static classifiers also work correctly"""
    print("\n--- Testing StaticClassifier ---")
    
    dataset_config = {
        'lookback_timesteps': 3,
        'states_size': 2,
        'states': {
            0: {'name': 'state_0', 'feature_begin_idx': 0, 'feature_end_idx': 1, 'type': 'categorial'},
            1: {'name': 'state_1', 'feature_begin_idx': 1, 'feature_end_idx': 2, 'type': 'binary'}
        }
    }
    
    # Create mock data with categorical values
    n_samples = 5
    lookback_timesteps = 3
    states_size = 2
    
    past_states = []
    for sample in range(n_samples):
        states_for_sample = []
        for t in range(lookback_timesteps):
            for f in range(states_size):
                # Use discrete values for categorical/binary features
                value = (t + f) % 3  # Values 0, 1, 2
                states_for_sample.append(value)
        past_states.append(states_for_sample)
    
    past_states = np.array(past_states)
    past_actions = np.random.rand(n_samples, lookback_timesteps * 1)
    X = np.hstack([past_states, past_actions])
    
    # Test categorical classifier
    static_model = StaticModel()
    static_model.dataset_config = dataset_config
    
    # Test categorical state (state 0)
    classifier = static_model._create_estimator(dataset_config['states'], 0)
    y = np.array([0, 1, 2, 0, 1])  # Some categorical labels
    
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    
    print(f"Categorical classifier feature_idx: {classifier.feature_idx}")
    print(f"Expected position: {(lookback_timesteps - 1) * states_size + 0}")
    print(f"Predictions: {predictions}")
    
    # The predictions should be from position (3-1)*2 + 0 = 4
    # Which corresponds to values from timestep 2, feature 0
    expected_values = [(2 + 0) % 3 for _ in range(n_samples)]  # All should be 2
    expected_predictions = np.array(expected_values)
    
    print(f"Expected: {expected_predictions}")
    
    if np.array_equal(predictions, expected_predictions):
        print("✅ PASS - Categorical classifier works")
        return True
    else:
        print("❌ FAIL - Categorical classifier issue")
        return False

if __name__ == "__main__":
    print("Starting comprehensive static model verification...")
    
    try:
        # Test various numerical configurations
        config_success = test_various_configurations()
        
        # Test categorical classifier
        classifier_success = test_classifier_type()
        
        if config_success and classifier_success:
            print("\n🎉 All tests passed! Static model fix is working correctly across all scenarios!")
        else:
            print("\n💥 Some tests failed. Static model fix needs attention!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during comprehensive test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
