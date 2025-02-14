# test_processors.py
import copy
import pytest
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    KBinsDiscretizer,
    Binarizer,
)
from pi_optimal.datasets.utils.processors import ProcessorRegistry

# --- Fixture to reset the registry state after each test ---
@pytest.fixture(autouse=True)
def reset_registry():
    # Make a deep copy of the original registry
    original_registry = copy.deepcopy(ProcessorRegistry._registry)
    yield
    ProcessorRegistry._registry = original_registry

# --- Test available_processors ---
def test_available_processors():
    available = ProcessorRegistry.available_processors()
    # Check that some well-known processors are in the registry
    expected_keys = [
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "QuantileTransformer",
        "PowerTransformer",
        "OneHotEncoder",
        "OrdinalEncoder",
        "LabelEncoder",
        "KBinsDiscretizer",
        "Binarizer",
    ]
    for key in expected_keys:
        assert key in available

# --- Test valid processor instantiation ---
def test_get_valid_processor_numerical():
    scaler = ProcessorRegistry.get("StandardScaler", "numerical")
    assert isinstance(scaler, StandardScaler)

def test_get_valid_processor_categorial():
    encoder = ProcessorRegistry.get("OrdinalEncoder", "categorial")
    assert isinstance(encoder, OrdinalEncoder)

# --- Test that OneHotEncoder gets default parameter 'sparse_output=False' ---
def test_get_one_hot_encoder_default():
    encoder = ProcessorRegistry.get("OneHotEncoder", "categorial")
    assert isinstance(encoder, OneHotEncoder)
    # In recent versions of scikit-learn, OneHotEncoder uses "sparse_output"
    # to control whether the output is sparse. The registry forces it to be False.
    assert getattr(encoder, "sparse_output", None) is False

# --- Test unknown processor raises ValueError ---
def test_get_unknown_processor():
    with pytest.raises(ValueError, match="Unknown processor: FakeProcessor"):
        ProcessorRegistry.get("FakeProcessor", "numerical")

# --- Test incompatible feature type raises ValueError ---
def test_get_incompatible_feature_type():
    # StandardScaler is registered for "numerical" features only.
    with pytest.raises(ValueError, match="Processor 'StandardScaler' is not compatible with feature type 'categorial'"):
        ProcessorRegistry.get("StandardScaler", "categorial")

# --- Test that additional keyword arguments are passed to the constructor ---
def test_get_with_additional_kwargs():
    # For example, pass a parameter to MinMaxScaler (e.g., feature_range)
    scaler = ProcessorRegistry.get("MinMaxScaler", "numerical", feature_range=(0, 1))
    # MinMaxScaler stores feature_range as an attribute
    assert hasattr(scaler, "feature_range")
    assert scaler.feature_range == (0, 1)

# --- Define a dummy processor for testing add/remove ---
class DummyProcessor:
    def __init__(self, factor=1):
        self.factor = factor

def test_add_and_get_dummy_processor():
    # Add DummyProcessor with a compatible type "dummy"
    ProcessorRegistry.add_processor("DummyProcessor", DummyProcessor, ["dummy"])
    # Retrieve it with a custom keyword argument
    dummy = ProcessorRegistry.get("DummyProcessor", "dummy", factor=42)
    assert isinstance(dummy, DummyProcessor)
    assert dummy.factor == 42

def test_remove_processor():
    # First, add a dummy processor then remove it.
    ProcessorRegistry.add_processor("DummyProcessor", DummyProcessor, ["dummy"])
    available = ProcessorRegistry.available_processors()
    assert "DummyProcessor" in available
    ProcessorRegistry.remove_processor("DummyProcessor")
    available_after = ProcessorRegistry.available_processors()
    assert "DummyProcessor" not in available_after

def test_remove_nonexistent_processor():
    with pytest.raises(ValueError, match="Processor NonExistentProcessor not found in registry"):
        ProcessorRegistry.remove_processor("NonExistentProcessor")

# --- Test get_compatible_types ---
def test_get_compatible_types_valid():
    compatible = ProcessorRegistry.get_compatible_types("StandardScaler")
    assert isinstance(compatible, list)
    # StandardScaler is defined for numerical features.
    assert "numerical" in compatible

def test_get_compatible_types_unknown():
    with pytest.raises(ValueError, match="Unknown processor: UnknownProcessor"):
        ProcessorRegistry.get_compatible_types("UnknownProcessor")
