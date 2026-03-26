import serial
import json
import numpy as np

FEATURE_KEYS = [
    "no2", "c2h5oh", "voc", "co",
    "mq135", "mq9", "mq3",
    "temp", "pressure", "humidity", "gas_res"
]
SERIAL_PORT = "/dev/cu.usbserial-5AA60782871"

def reading_to_vector(reading: dict) -> np.ndarray:
    return np.array([reading[k] for k in FEATURE_KEYS], dtype=np.float32)

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
                continue  # skip malformed lines cleanly

            # Skip status messages
            if "status" in packet or "error" in packet:
                print(f"[sensor] {packet}")
                continue

            features = reading_to_vector(packet)
            label = model.predict([features])[0]

            print(f"idx={packet['idx']:05d} → {label}")

if __name__ == "__main__":
    class DummyModel:
        def predict(self, X):
            return ["apple"] * len(X)

    model = DummyModel()
    run_pipeline(model)
