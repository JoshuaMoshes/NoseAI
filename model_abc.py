import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
