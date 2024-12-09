# base_data_generator.py
import pandas as pd


class BaseDataGenerator:
    """Base class for data generators.

    Data generators are used to collect datasets from various sources, such as
    Gymnasium environments, simulations, or real-world data sources. The collected
    datasets can be used for training offline RL Algorithms."""

    def collect(self) -> pd.DataFrame:
        """Collects a dataset and return as pandas dataframe."""
        raise NotImplementedError
