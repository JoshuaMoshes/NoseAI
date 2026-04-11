import serial
import json
import numpy as np
from models.model_abc import Model
from models.transformer import TransformerModel, ScentTransformerNet


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

def reading_to_vector(reading: dict, keys: list) -> np.ndarray:
    return np.array([reading[k] for k in keys], dtype=np.float32)

def run_pipeline(models: list[Model], port=SERIAL_PORT, baud=115200):
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
            for model in models:
                name = model.__class__.__name__

                prediction = model.predict([features])

                print(f"Features: {features}")
                print(f"Model {name} predicted: {prediction}")

                label = prediction[0]

                if label is None:
                    continue

                # Full vector including extra sensors
                # full_features = reading_to_vector(packet, ALL_FEATURE_KEYS)

                print(f"idx={packet['idx']:05d}: {name}: {label}")

if __name__ == "__main__":
    model = TransformerModel.load("transformer.pt")

    try:
        run_pipeline([model])
    except KeyboardInterrupt:
        print("Exiting...")
