import serial
import json
import numpy as np
import pandas as pd
import time
from models.model_abc import Model
from models.transformer import TransformerModel, ScentTransformerNet
from models.lstm import LSTMModel, OdorLSTM
from models.knn import KNNModel
from normalizer import SensorNormalizer


# Keys matching SmellNET dataset
SMELLNET_FEATURE_KEYS = [
    "NO2", "C2H5OH", "VOC", "CO",
    "Alcohol", "LPG", "Benzene",
    "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude"
]

# Extra sensors not in SmellNET
EXTRA_FEATURE_KEYS = ["MQ5", "MP503"]

ALL_FEATURE_KEYS = SMELLNET_FEATURE_KEYS + EXTRA_FEATURE_KEYS

SERIAL_PORT = "/dev/cu.usbserial-5AA60782871"

TRAINING_CSV = "training-data.csv"
TARGET_COLUMN = "type"

BASELINE_WARMUP_SECONDS = 4

def load_training_stats(csv_path, target_column):
    """Compute per-feature mean and std from the training set."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    return means, stds

def reading_to_vector(reading: dict, keys: list) -> np.ndarray:
    return np.array([reading[k] for k in keys], dtype=np.float32)

def run_pipeline(models: list[Model], normalizer, port=SERIAL_PORT, baud=115200):
    with serial.Serial(port, baud, timeout=2) as ser:
        print("Waiting for sensor warmup...")

        baseline_time = time.time()
        while True:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue

            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "status" in packet or "error" in packet:
                print(f"[sensor] {packet}")
                continue

            # SmellNET-compatible feature vector for inference
            raw = reading_to_vector(packet, SMELLNET_FEATURE_KEYS)

            print(f"Feature vector: {raw}, Percent off baseline per feature: {
                (raw - normalizer.baseline) / normalizer.baseline * 100 if normalizer.baseline is not None else 'N/A'
                  }")

            if normalizer.baseline is None:
                print(f"Baseline warmup: {time.time() - baseline_time:.1f}s")
                normalizer.add_warmup_reading(raw)

                if time.time() - baseline_time >= BASELINE_WARMUP_SECONDS:
                    normalizer.compute_baseline()
                    print("Baseline computed:", normalizer.baseline)

                continue

            features = normalizer.normalize(raw)

            for model in models:
                name = model.__class__.__name__

                prediction = model.predict([features])

                label = prediction[0]

                if label is None:
                    continue

                # Full vector including extra sensors
                # full_features = reading_to_vector(packet, ALL_FEATURE_KEYS)

                print(f"idx={packet['idx']:05d}: {name}: {label}")

if __name__ == "__main__":
    train_means, train_stds = load_training_stats(TRAINING_CSV, TARGET_COLUMN)

    models = [
        TransformerModel.load("models/transformer.pt"),
        KNNModel.load("models/knn.pkl"),
        LSTMModel.load("LSTM-Model/best_model.pt")
    ]

    normalizer = SensorNormalizer(warmup_seconds=BASELINE_WARMUP_SECONDS, train_means=train_means, train_stds=train_stds)

    try:
        run_pipeline(models, normalizer)
    except KeyboardInterrupt:
        print("Exiting...")
