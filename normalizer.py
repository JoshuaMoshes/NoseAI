import numpy as np

class SensorNormalizer:
    """Our hardware sensor readings are very different to SmellNET's training data, 
    so we need to do some normalization before inference. This class handles that."""

    def __init__(self, warmup_seconds, train_means, train_stds):
        self.train_means = train_means
        self.train_stds = train_stds
        self.baseline = None
        self._warmup_readings = []
        self.warmup_seconds = warmup_seconds

    def add_warmup_reading(self, raw_vector: np.ndarray):
        """Collect clean-air readings during warmup phase."""
        self._warmup_readings.append(raw_vector)
        print(f"Collected warmup reading {len(self._warmup_readings)}: {raw_vector}")

    def compute_baseline(self):
        """Compute per-sensor baseline from warmup readings."""
        self.baseline = np.mean(self._warmup_readings, axis=0)
        print(f"Baseline set from {len(self._warmup_readings)} readings")

    def normalize(self, raw_vector: np.ndarray) -> np.ndarray:
        """Baseline-subtract, then z-score against training stats."""
        if self.baseline is not None:
            centered = raw_vector - self.baseline
        else:
            centered = raw_vector
        return (centered - self.train_means) / self.train_stds
