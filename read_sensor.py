import serial
import json
import numpy as np

# Keys matching SmellNET dataset
SMELLNET_FEATURE_KEYS = [
    "NO2", "C2H5OH", "VOC", "CO",
    "Alcohol", "LPG", "Benzene",
    "Temperature", "Pressure", "Humidity", "Gas_Resistance"
]

# Extra sensors not in SmellNET
EXTRA_FEATURE_KEYS = ["MQ5", "MP503"]

ALL_FEATURE_KEYS = SMELLNET_FEATURE_KEYS + EXTRA_FEATURE_KEYS

SERIAL_PORT = "/dev/cu.usbserial-5AA60782871"

def reading_to_vector(reading: dict, keys: list) -> np.ndarray:
    return np.array([reading[k] for k in keys], dtype=np.float32)

def run_pipeline(model, port=SERIAL_PORT, baud=115200):
    with serial.Serial(port, baud, timeout=2) as ser:
        print("Waiting for sensor warmup...")
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
            features = reading_to_vector(packet, SMELLNET_FEATURE_KEYS)
            label = model.predict([features])[0]

            # Full vector including extra sensors
            full_features = reading_to_vector(packet, ALL_FEATURE_KEYS)

            print(f"idx={packet['idx']:05d} → {label}")

if __name__ == "__main__":
    class DummyModel:
        def predict(self, X):
            return ["apple"] * len(X)

    model = DummyModel()
    run_pipeline(model)
