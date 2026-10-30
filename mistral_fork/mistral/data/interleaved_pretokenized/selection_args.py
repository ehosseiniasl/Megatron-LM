from abc import ABC, abstractmethod

import numpy as np


class SelectionStrategy(ABC):
    """Abstract base class for classes that have a select method."""

    @abstractmethod
    def select(
        self,
        documents_sizes: np.ndarray,
        documents: np.ndarray,
    ) -> np.ndarray:
        """Abstract select method to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def get_hash(self) -> str:
        """Abstract hash method to be implemented by subclasses."""
        raise NotImplementedError
