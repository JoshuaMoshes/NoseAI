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


def test_model(model: Model, X: np.ndarray, y: np.ndarray):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Model {model.__class__.__name__} accuracy: {accuracy:.2%}")
