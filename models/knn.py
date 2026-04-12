import numpy as np
import pandas as pd
import pickle
from .model_abc import Model, test_model


class KNNModel(Model):
    def __init__(self, k: int = 5):
        self.k = k
        self.min_vals: np.ndarray | None = None
        self.ranges: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def _fit_min_max(self, X: np.ndarray):
        self.min_vals = X.min(axis=0)
        ranges = X.max(axis=0) - self.min_vals
        ranges[ranges == 0] = 1
        self.ranges = ranges

    def _transform_min_max(self, X: np.ndarray) -> np.ndarray:
        if self.min_vals is None or self.ranges is None:
            raise ValueError("Normalization parameters are not fitted yet.")
        return (X - self.min_vals) / self.ranges

    def _euclidean_distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        difference = point_a - point_b
        return np.sqrt(np.sum(difference * difference))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        self._fit_min_max(X)
        self.X_train = self._transform_min_max(X)
        self.y_train = y

        print("Training complete")
        print(f"K value: {self.k}")
        print(f"Training samples stored: {len(self.X_train)}")

    def _predict_one(self, x_test: np.ndarray) -> str:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")

        distances: list[tuple[float, str]] = []

        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(self.X_train[i], x_test)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda item: item[0])
        nearest_neighbors = distances[:self.k]

        label_counts: dict[str, int] = {}
        for _, label in nearest_neighbors:
            label_counts[label] = label_counts.get(label, 0) + 1

        best_label = None
        best_count = -1
        for label, count in label_counts.items():
            if count > best_count:
                best_count = count
                best_label = label

        if best_label is None:
            raise ValueError("Prediction failed. No nearest neighbors found.")

        return best_label

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[np.newaxis, :]

        X_norm = self._transform_min_max(X)
        predictions = [self._predict_one(x) for x in X_norm]
        return np.array(predictions)

    def _predict_proba_one(self, x_test: np.ndarray) -> dict:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet.")

        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(self.X_train[i], x_test)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda item: item[0])
        nearest = distances[:self.k]

        label_counts: dict[str, int] = {}
        for _, label in nearest:
            label_counts[label] = label_counts.get(label, 0) + 1
        return {label: count / self.k for label, count in label_counts.items()}

    def predict_proba(self, X: np.ndarray | list[np.ndarray]) -> list[dict]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = self._transform_min_max(X)
        return [self._predict_proba_one(x) for x in X_norm]

    def save(self, path: str):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet. Nothing to save.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "KNNModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    TARGET_COLUMN = "type"
    df = pd.read_csv("collected-data.csv")

    TARGET_COLUMN = "type"

    X = df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNNModel(k=5)

    model.fit(
        X_train,
        y_train
    )

    test_model(
        model,
        X_test,
        y_test
    )

    sample_values = [54, 94, 130, 765, 2, 12, 0, 33.59, 688.6, 100.0, 0.0, 3170.54, 0, 0]
    sample_values = np.array(sample_values, dtype=np.float32).reshape(1, -1)
    print("Predicted:", model.predict(sample_values))

    model.save("knn.pkl")
