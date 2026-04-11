import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import Dataset, DataLoader
from .model_abc import Model


WINDOW = 50
STRIDE = 10
HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
STEP_SIZE = 10
GAMMA = 0.5


class OdorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class _WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(Model):
    """
    LSTM odor classifier implementing the Model ABC.

    Two prediction modes:
      - Batch mode  (X.ndim == 3, shape (N, W, F)): pre-windowed data for offline eval.
      - Stream mode (X.ndim == 2, shape (N, F)):    raw frames fed one at a time from a
          live sensor.  Frames accumulate in a rolling buffer (size WINDOW+1); once full,
          FOTD is applied across the buffer to produce W delta frames, which are z-score
          normalised and fed to the LSTM.  Returns None for each frame until warm-up.
    """

    def __init__(self, window=WINDOW, stride=STRIDE, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT, lr=LR,
                 epochs=EPOCHS, batch_size=BATCH_SIZE,
                 step_size=STEP_SIZE, gamma=GAMMA):
        self.window = window
        self.stride = stride
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.gamma = gamma

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.net: OdorLSTM | None = None
        # z-score normalisation params (fitted on training deltas)
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None
        self.class_to_idx: dict | None = None
        self.idx_to_class: dict | None = None
        # Rolling buffer of raw frames for live streaming
        self._frame_buffer: deque = deque(maxlen=window + 1)

    def _fit_norm(self, X_flat: np.ndarray):
        self.norm_mean = X_flat.mean(axis=0)
        self.norm_std = X_flat.std(axis=0)
        self.norm_std[self.norm_std == 0] = 1.0

    def _transform_norm(self, X_flat: np.ndarray) -> np.ndarray:
        return (X_flat - self.norm_mean) / self.norm_std
    
    def _load_csvs(self, root_dir: str):
        items = []
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for csv_path in sorted(glob.glob(os.path.join(class_path, "*.csv"))):
                df = pd.read_csv(csv_path)
                items.append((df, class_name))
        return items

    def _build_windows(self, root_dirs, fit_norm: bool = False):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        raw = []
        for d in root_dirs:
            raw.extend(self._load_csvs(d))

        if self.class_to_idx is None:
            classes = sorted({lbl for _, lbl in raw})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        all_windows, all_labels = [], []
        for df, lbl in raw:
            if lbl not in self.class_to_idx:
                continue
            arr = df.values.astype(np.float32)
            diff = arr[1:] - arr[:-1]           # FOTD: (T-1, F)
            T = diff.shape[0]
            for start in range(0, T - self.window + 1, self.stride):
                all_windows.append(diff[start : start + self.window])
                all_labels.append(self.class_to_idx[lbl])

        X = np.stack(all_windows, axis=0)        # (N, W, F)
        y = np.array(all_labels, dtype=np.int64)

        N, W, F = X.shape
        X_flat = X.reshape(N * W, F)

        if fit_norm:
            self._fit_norm(X_flat)

        X_flat = self._transform_norm(X_flat)
        return X_flat.reshape(N, W, F).astype(np.float32), y

    def fit(self, train_dir: str, ckpt_path: str = "best_model.pt"):
        print(f"\nLoading training data from {train_dir} ...")
        X, y = self._build_windows(train_dir, fit_norm=True)
        num_classes = len(self.class_to_idx)
        print(f"  {num_classes} classes: {', '.join(sorted(self.class_to_idx))}")
        print(f"  Training windows: {len(X)}")

        input_size = X.shape[2]
        self.net = OdorLSTM(
            input_size=input_size,
            hidden_size=self.hidden,
            num_layers=self.n_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)
        print(f"  Parameters: {sum(p.numel() for p in self.net.parameters()):,}")

        loader = DataLoader(
            _WindowDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        best_top1 = 0.0
        print(f"\n{'Epoch':>6}  {'Loss':>9}  {'Top-1':>7}")
        print("-" * 28)

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = correct = total = 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.net(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                n = X_b.size(0)
                total_loss += loss.item() * n
                correct += logits.argmax(dim=1).eq(y_b).sum().item()
                total += n
            scheduler.step()
            top1 = correct / total
            print(f"  {epoch:>4}  {total_loss / total:>9.4f}  {top1 * 100:>6.2f}%")
            if top1 > best_top1:
                best_top1 = top1
                torch.save(self.net.state_dict(), ckpt_path)

        print(f"\nBest Top-1: {best_top1 * 100:.2f}%  checkpoint -> {ckpt_path}")
        self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Batch mode  — X shape (N, W, F): already windowed, normalise + predict.
        Stream mode — X shape (N, F):    each row is one new raw sensor frame.
            Appended to a rolling buffer (maxlen = window+1). When full, FOTD is
            applied across the buffer -> (W, F) delta frames -> normalise -> LSTM.
            Returns None for frames that haven't filled the buffer yet.
        """
        if self.net is None or self.norm_mean is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 3:
            # Batch mode: X is (N, W, F)
            N, W, F = X.shape
            X_flat = self._transform_norm(X.reshape(N * W, F))
            X_scaled = X_flat.reshape(N, W, F).astype(np.float32)
            tensor = torch.tensor(X_scaled).to(self.device)
            self.net.eval()
            with torch.no_grad():
                indices = self.net(tensor).argmax(dim=1).cpu().numpy()
            return np.array([self.idx_to_class[i] for i in indices])

        elif X.ndim == 2:
            # Stream mode: X is (N, F) — typically (1, F) per sensor tick
            results = []
            for frame in X:
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) >= self.window + 1:
                    frames = np.array(self._frame_buffer)              # (W+1, F)
                    diff = (frames[1:] - frames[:-1]).astype(np.float32)  # (W, F) FOTD
                    diff_norm = self._transform_norm(diff)
                    tensor = (
                        torch.tensor(diff_norm, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    self.net.eval()
                    with torch.no_grad():
                        idx = self.net(tensor).argmax(dim=1).item()
                    results.append(self.idx_to_class[idx])
                else:
                    results.append(None)
            return np.array(results, dtype=object)

        else:
            raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D.")

    def save(self, path: str):
        if self.net is None:
            raise ValueError("Model not fitted. Nothing to save.")
        self.net.eval()
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "LSTMModel":
        model = torch.load(path, weights_only=False)
        model.net.eval()
        model._frame_buffer = deque(maxlen=model.window + 1)
        return model
