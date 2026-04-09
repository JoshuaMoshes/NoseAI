import pandas as pd
import numpy as np
import time

trainingDataDataFrame = pd.read_csv("../training-data.csv")
testingDataDataFrame = pd.read_csv("../testing-data.csv")
onlineNutsValidation = pd.read_csv("../online_nuts_validation.csv")
onlineSpicesValidation = pd.read_csv("../online_spices_validation.csv")

TARGET_COLUMN = "type"

K_VALUE = 5

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


class KNNModel:
    def __init__(self, k=5):
        self.k = k
        self.feature_columns = None
        self.min_vals = None
        self.ranges = None
        self.X_train = None
        self.y_train = None

    def min_max_fit(self, X):
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        return min_vals, ranges

    def min_max_transform(self, X):
        return (X - self.min_vals) / self.ranges

    def euclidean_distance(self, point_a, point_b):
        difference = point_a - point_b
        return np.sqrt(np.sum(difference * difference))

    def fit(self, dataframe, target_column, selected_features=None):
        self.feature_columns = get_feature_columns(dataframe, target_column, selected_features)

        X_train = dataframe[self.feature_columns].to_numpy(dtype=float)
        y_train = dataframe[target_column].to_numpy()

        self.min_vals, self.ranges = self.min_max_fit(X_train)
        self.X_train = self.min_max_transform(X_train)
        self.y_train = y_train

        print("Training complete")
        print("K value:", self.k)
        print("Features used:", self.feature_columns)
        print("Training samples stored:", len(self.X_train))

    def predict_one(self, x_test):
        distances = []

        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(self.X_train[i], x_test)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda item: item[0])
        nearest_neighbors = distances[:self.k]

        label_counts = {}
        for _, label in nearest_neighbors:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        best_label = None
        best_count = -1
        for label, count in label_counts.items():
            if count > best_count:
                best_count = count
                best_label = label

        return best_label

    def predict_from_values(self, values):
        if self.feature_columns is None:
            raise ValueError("Model must be fitted before calling predict_from_values().")

        if len(values) != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} values, but got {len(values)}."
            )

        values_array = np.array(values, dtype=float).reshape(1, -1)
        normalized_values = self.min_max_transform(values_array)
        prediction = self.predict_one(normalized_values[0])

        return prediction

    def predict(self, dataframe, target_column, dataset_name="dataset"):
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

            if completed % 10 == 0 or completed == total_items:
                print(
                    f"{dataset_name}: {completed}/{total_items} completed | "
                    f"elapsed: {elapsed_time:.2f}s | "
                    f"estimated time left: {estimated_time_left:.2f}s",
                    flush=True
                )

        return np.array(predictions), y_test

    def evaluate(self, dataframe, target_column, dataset_name="dataset"):
        start_time = time.time()
        predictions, y_true = self.predict(dataframe, target_column, dataset_name)
        total_time = time.time() - start_time
        accuracy = calculate_accuracy(y_true, predictions)

        print(f"{dataset_name} accuracy:", accuracy)
        print(f"{dataset_name} total time:", round(total_time, 2), "seconds")

        return accuracy


model = KNNModel(k=K_VALUE)

model.fit(trainingDataDataFrame, TARGET_COLUMN, SELECTED_FEATURES)
# Uncomment this to get all the testing data:
#
# model.evaluate(testingDataDataFrame, TARGET_COLUMN, "Testing set")
#

# if USE_VALIDATION_SETS:
#     model.evaluate(onlineNutsValidation, TARGET_COLUMN, "Online nuts validation")
#     model.evaluate(onlineSpicesValidation, TARGET_COLUMN, "Online spices validation")

# This is one example you can plug in to see if it works for a specific element
sample_values = [54,94,130,765,2,12,0,33.59,688.6,100.0,0.0,3170.54]
predicted_target = model.predict_from_values(sample_values)

print("Predicted target:", predicted_target)