import pandas as pd
import numpy as np
import pickle
import time

TARGET_COLUMN = "type"

LEARNING_RATE = 0.1
EPOCHS = 1000
PRINT_EVERY = 100

USE_VALIDATION_SETS = False

SELECTED_FEATURES = None


def get_feature_columns(dataframe, target_column, selected_features=None):
    if selected_features is not None:
        return selected_features
    return [column for column in dataframe.columns if column != target_column]


def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


class MLRModel:
    def __init__(self, learning_rate=0.1, epochs=1000, print_every=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_every = print_every
        self.feature_columns = None
        self.classes = None
        self.class_to_index = None
        self.min_vals = None
        self.ranges = None
        self.W = None
        self.b = None

    def min_max_fit(self, X):
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        return min_vals, ranges

    def min_max_transform(self, X):
        return (X - self.min_vals) / self.ranges

    def softmax(self, Z):
        # Subtract row max for numerical stability
        Z_shifted = Z - Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def cross_entropy_loss(self, P, Y_one_hot):
        n = len(Y_one_hot)
        log_probs = np.log(P + 1e-15)
        return -np.sum(Y_one_hot * log_probs) / n

    def one_hot_encode(self, y):
        n_classes = len(self.classes)
        Y = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            Y[i, self.class_to_index[label]] = 1
        return Y

    def fit(self, dataframe, target_column, selected_features=None):
        self.feature_columns = get_feature_columns(dataframe, target_column, selected_features)

        X_train = dataframe[self.feature_columns].to_numpy(dtype=float)
        y_train = dataframe[target_column].to_numpy()

        self.min_vals, self.ranges = self.min_max_fit(X_train)
        X_train = self.min_max_transform(X_train)

        self.classes = sorted(list(set(y_train)))
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}

        n_samples, n_features = X_train.shape
        n_classes = len(self.classes)

        np.random.seed(42)
        self.W = np.random.uniform(-0.1, 0.1, (n_features, n_classes))
        self.b = np.zeros(n_classes)

        Y_one_hot = self.one_hot_encode(y_train)

        print("Training multinomial logistic regression...")
        print("Features used:", self.feature_columns)
        print("Classes:", self.classes)
        print(f"Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
        print(f"Learning rate: {self.learning_rate} | Epochs: {self.epochs}")

        for epoch in range(1, self.epochs + 1):
            Z = X_train @ self.W + self.b
            P = self.softmax(Z)

            loss = self.cross_entropy_loss(P, Y_one_hot)

            dZ = (P - Y_one_hot) / n_samples
            dW = X_train.T @ dZ
            db = dZ.sum(axis=0)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            if epoch % self.print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} | Loss: {loss:.6f}")

        print("Training complete")

    def predict_one(self, x):
        Z = x @ self.W + self.b
        Z_shifted = Z - Z.max()
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / exp_Z.sum()
        class_index = int(np.argmax(P))
        return self.classes[class_index]

    def predict(self, X):
        if self.W is None:
            raise ValueError("Model must be fitted before calling predict().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = self.min_max_transform(X)
        return np.array([self.predict_one(x) for x in X_norm])

    def _predict_proba_one(self, x: np.ndarray) -> dict:
        Z = x @ self.W + self.b
        Z_shifted = Z - Z.max()
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / exp_Z.sum()
        return {cls: float(P[i]) for i, cls in enumerate(self.classes)}

    def predict_proba(self, X) -> list[dict]:
        if self.W is None:
            raise ValueError("Model must be fitted before calling predict_proba().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = self.min_max_transform(X)
        return [self._predict_proba_one(x) for x in X_norm]

    def _predict(self, dataframe, target_column, dataset_name="dataset"):
        X_test = dataframe[self.feature_columns].to_numpy(dtype=float)
        y_test = dataframe[target_column].to_numpy()
        X_test = self.min_max_transform(X_test)

        predictions = []
        start_time = time.time()
        total_items = len(X_test)

        for index, x_test in enumerate(X_test):
            prediction = self.predict_one(x_test)
            predictions.append(prediction)

            completed = index + 1
            elapsed_time = time.time() - start_time
            average_time_per_item = elapsed_time / completed
            remaining_items = total_items - completed
            estimated_time_left = average_time_per_item * remaining_items

            if completed % 100 == 0 or completed == total_items:
                print(
                    f"{dataset_name}: {completed}/{total_items} completed | "
                    f"elapsed: {elapsed_time:.2f}s | "
                    f"estimated time left: {estimated_time_left:.2f}s",
                    flush=True
                )

        return np.array(predictions), y_test

    def save(self, path: str):
        if self.W is None:
            raise ValueError("Model has not been trained yet. Nothing to save.")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MLRModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    def evaluate(self, dataframe, target_column, dataset_name="dataset"):
        start_time = time.time()
        predictions, y_true = self._predict(dataframe, target_column, dataset_name)
        total_time = time.time() - start_time
        accuracy = calculate_accuracy(y_true, predictions)

        print(f"{dataset_name} accuracy:", accuracy)
        print(f"{dataset_name} total time:", round(total_time, 2), "seconds")

        return accuracy


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    TARGET_COLUMN = "type"
    df = pd.read_csv("collected-data.csv")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN]
    )

    model = MLRModel(learning_rate=LEARNING_RATE, epochs=EPOCHS, print_every=PRINT_EVERY)
    model.fit(train_df, TARGET_COLUMN)
    model.evaluate(test_df, TARGET_COLUMN, "Test set")

    sample_values = [54, 94, 130, 765, 2, 12, 0, 33.59, 688.6, 100.0, 0.0, 3170.54, 0, 0]
    sample_values = np.array(sample_values, dtype=np.float32).reshape(1, -1)
    print("Predicted:", model.predict(sample_values[0]))

    model.save("mlr.pkl")
