from abc import ABC, abstractmethod
from typing import Any

class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.

    All data loaders should inherit from this class and must implement the
    `load` and `get_data` methods.
    Training and eval loops should depend only on this interface, allowing 
    for flexibility in the choice of data loading mechanism.
    Every loader receives a config dictionary, which can contain whatever is
    needed for loading and preprocessing that data.
    """

    def __init__(self, config: dict):
        """Initialize the data loader with the provided configuration."""
        self.config = config
    
    @abstractmethod
    def load(self) -> None:
        """Load and preprocess the data."""
        ...

    @abstractmethod
    def get_data(self) -> Any:
        """Return the processed data."""
        ...