# processors.py
from sklearn.preprocessing import (  # type: ignore
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    KBinsDiscretizer,
    Binarizer,
    LabelEncoder,
)


class ProcessorRegistry:
    _registry = {
        "StandardScaler": (StandardScaler, ["numerical"]),
        "MinMaxScaler": (MinMaxScaler, ["numerical"]),
        "MaxAbsScaler": (MaxAbsScaler, ["numerical"]),
        "RobustScaler": (RobustScaler, ["numerical"]),
        "QuantileTransformer": (QuantileTransformer, ["numerical"]),
        "PowerTransformer": (PowerTransformer, ["numerical"]),
        "OneHotEncoder": (OneHotEncoder, ["categorial"]),
        "OrdinalEncoder": (OrdinalEncoder, ["categorial"]),
        "LabelEncoder": (LabelEncoder, ["categorial"]),
        "KBinsDiscretizer": (KBinsDiscretizer, ["numerical"]),
        "Binarizer": (Binarizer, ["numerical"]),
    }

    @classmethod
    def get(cls, name, feature_type, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown processor: {name}")

        processor_class, compatible_types = cls._registry[name]

        if feature_type not in compatible_types:
            raise ValueError(
                f"Processor '{name}' is not compatible with feature type '{feature_type}'. "
                f"Compatible types are: {compatible_types}"
            )

        if name == "OneHotEncoder" and "sparse_output" not in kwargs:
            kwargs["sparse_output"] = (
                False  # Ensure dense output by default for easier handling
            )

        return processor_class(**kwargs)

    @classmethod
    def available_processors(cls):
        return list(cls._registry.keys())

    @classmethod
    def add_processor(cls, name, processor_class, compatible_types):
        cls._registry[name] = (processor_class, compatible_types)

    @classmethod
    def remove_processor(cls, name):
        if name in cls._registry:
            del cls._registry[name]
        else:
            raise ValueError(f"Processor {name} not found in registry")

    @classmethod
    def get_compatible_types(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Unknown processor: {name}")
        return cls._registry[name][1]
