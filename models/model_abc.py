import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def load(cls, path: str) -> "Model":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        pass

    def predict_one(self, x: np.ndarray) -> str:
        return self.predict(x[np.newaxis, :])[0]

    def predict_proba(self, X: np.ndarray | list[np.ndarray]) -> list[dict | None]:
        """Return a list of {label: probability} dicts, or None when output is unavailable."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba")


def test_model(model: Model, X: np.ndarray, y: np.ndarray):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Model {model.__class__.__name__} accuracy: {accuracy:.2%}")
