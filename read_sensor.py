import serial
import json
import numpy as np
import pandas as pd
import time
import csv
import os
from models.model_abc import Model
from models.transformer import TransformerModel, ScentTransformerNet
from models.lstm import LSTMModel, OdorLSTM
from models.knn import KNNModel
from models.mlr import MLRModel

ALL_FEATURE_KEYS = [
    "NO2", "C2H5OH", "VOC", "CO",
    "Alcohol", "LPG", "Benzene",
    "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude",
    "MQ5", "MP503"
]

SERIAL_PORT = "/dev/cu.usbserial-5AA60782871"
TRAINING_CSV = "training-data.csv"
TARGET_COLUMN = "type"
BASELINE_WARMUP_SECONDS = 4
COLLECT_CSV = "collected-data.csv"


def reading_to_vector(reading: dict, keys: list) -> np.ndarray:
    return np.array([reading[k] for k in keys], dtype=np.float32)


def read_packets(port=SERIAL_PORT, baud=115200):
    """Yield parsed sensor packets from serial."""
    with serial.Serial(port, baud, timeout=2) as ser:
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
            yield packet


def collect_data(port=SERIAL_PORT, baud=115200, output=COLLECT_CSV):
    """Record labeled sensor data to CSV. Prompts for label, then records
    until Ctrl+C. Run again with a new label for each scent."""
    label = input("Enter scent label: ").strip()
    if not label:
        print("No label provided, exiting.")
        return

    file_exists = os.path.exists(output)
    count = 0

    with open(output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(ALL_FEATURE_KEYS + [TARGET_COLUMN])

        print(f"Recording '{label}' — hold sample near sensor. Ctrl+C to stop.")
        try:
            for packet in read_packets(port, baud):
                raw = reading_to_vector(packet, ALL_FEATURE_KEYS)
                writer.writerow(list(raw) + [label])
                count += 1
                if count % 10 == 0:
                    f.flush()
                    print(f"  {count} readings collected for '{label}'")
        except KeyboardInterrupt:
            print(f"\nSaved {count} readings of '{label}' to {output}")


def run_pipeline(models: list[Model], port=SERIAL_PORT, baud=115200):
    print("Waiting for sensor warmup...")
    for packet in read_packets(port, baud):
        features = reading_to_vector(packet, ALL_FEATURE_KEYS)

        #print(f"idx={packet['idx']:05d}: features={features}")
        print("\n\n")
        for model in models:
            name = model.__class__.__name__
            prediction = model.predict([features])
            label = prediction[0]
            if label is None:
                continue
            print(f"idx={packet['idx']:05d}: {name}: {label}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect_data()
    else:
        models = [
            TransformerModel.load("models/transformer.pt"),
            KNNModel.load("models/knn.pkl"),
            LSTMModel.load("LSTM-Model/best_model.pt"),
            MLRModel.load("models/mlr.pkl")
        ]
        try:
            run_pipeline(models)
        except KeyboardInterrupt:
            print("Exiting...")
