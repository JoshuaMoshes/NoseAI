import os
import glob
import json
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

TRAIN_DIR = "../smell-net/offline_testing"
TEST_DIRS = [
    "../smell-net/online_nuts",
    "../smell-net/online_spices",
]
CKPT_PATH = "best_model.pt"
SERIAL_PORT = "/dev/cu.usbserial-5AA60782871"

WINDOW     = 50
STRIDE     = 10
HIDDEN     = 128
N_LAYERS   = 2
DROPOUT    = 0.3
LR         = 1e-3
EPOCHS     = 100
BATCH_SIZE = 64
STEP_SIZE  = 10
GAMMA      = 0.5

# Keys matching the SmellNET dataset columns (same order as read_sensor.py)
SMELLNET_FEATURE_KEYS = [
    "NO2", "C2H5OH", "VOC", "CO",
    "Alcohol", "LPG", "Benzene",
    "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude"
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csvs_from_dir(root_dir):
    items = []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for csv_path in sorted(glob.glob(os.path.join(class_path, "*.csv"))):
            df = pd.read_csv(csv_path)
            items.append((df, class_name))
    return items


def apply_fotd(df):
    arr = df.values.astype(np.float32)
    return arr[1:] - arr[:-1]          # (T-1, F)


def sliding_windows(arr, window, stride):
    T = arr.shape[0]
    windows = []
    for start in range(0, T - window + 1, stride):
        windows.append(arr[start : start + window])
    return windows


def build_dataset(root_dirs, class_to_idx=None, norm_mean=None, norm_std=None, fit_norm=False):
    """
    Returns X (N, W, F), y (N,), class_to_idx, norm_mean, norm_std.
    fit_norm=True: compute mean/std from this data (training set).
    fit_norm=False: use the provided mean/std (test / live set).
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    raw = []
    for d in root_dirs:
        raw.extend(load_csvs_from_dir(d))

    if class_to_idx is None:
        classes = sorted({lbl for _, lbl in raw})
        class_to_idx = {c: i for i, c in enumerate(classes)}

    all_windows, all_labels = [], []
    for df, lbl in raw:
        if lbl not in class_to_idx:
            continue
        diff = apply_fotd(df)
        wins = sliding_windows(diff, WINDOW, STRIDE)
        all_windows.extend(wins)
        all_labels.extend([class_to_idx[lbl]] * len(wins))

    X = np.stack(all_windows, axis=0)   # (N, W, F)
    y = np.array(all_labels, dtype=np.int64)

    N, W, F = X.shape
    X_flat = X.reshape(N * W, F)

    if fit_norm:
        norm_mean = X_flat.mean(axis=0)
        norm_std  = X_flat.std(axis=0)
        norm_std[norm_std == 0] = 1.0

    X_flat = (X_flat - norm_mean) / norm_std
    X = X_flat.reshape(N, W, F).astype(np.float32)
    return X, y, class_to_idx, norm_mean, norm_std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = top1_sum = top5_sum = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        n = X_b.size(0)
        total_loss += loss.item() * n
        total += n
        _, top1 = logits.topk(1, dim=1)
        top1_sum += top1.squeeze(1).eq(y_b).sum().item()
        k = min(5, logits.size(1))
        _, topk = logits.topk(k, dim=1)
        top5_sum += topk.eq(y_b.view(-1, 1).expand_as(topk)).any(dim=1).sum().item()
    return total_loss / total, top1_sum / total, top5_sum / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = top1_sum = top5_sum = total = 0
    all_preds, all_targets = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        n = X_b.size(0)
        total_loss += loss.item() * n
        total += n
        _, top1 = logits.topk(1, dim=1)
        top1_sum += top1.squeeze(1).eq(y_b).sum().item()
        k = min(5, logits.size(1))
        _, topk = logits.topk(k, dim=1)
        top5_sum += topk.eq(y_b.view(-1, 1).expand_as(topk)).any(dim=1).sum().item()
        all_preds.extend(top1.squeeze(1).cpu().numpy())
        all_targets.extend(y_b.cpu().numpy())
    return (
        total_loss / total,
        top1_sum / total,
        top5_sum / total,
        np.array(all_preds),
        np.array(all_targets),
    )


# ---------------------------------------------------------------------------
# Live streaming
# ---------------------------------------------------------------------------

def run_live(model, idx_to_class, norm_mean, norm_std, device):
    """
    Read live frames from the sensor over serial, maintain a rolling buffer of
    WINDOW+1 raw frames, apply FOTD across the buffer -> (WINDOW, F) deltas ->
    z-score normalise -> LSTM -> class label.

    The first WINDOW frames print a warm-up message while the buffer fills.
    After that every frame produces a prediction, matching the same preprocessing
    pipeline used during training.
    """
    import serial

    buffer = deque(maxlen=WINDOW + 1)

    print(f"\nOpening serial port {SERIAL_PORT} ...")
    print(f"Buffer warms up after {WINDOW} frames. Press Ctrl-C to stop.\n")

    with serial.Serial(SERIAL_PORT, 115200, timeout=2) as ser:
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

            features = np.array(
                [packet[k] for k in SMELLNET_FEATURE_KEYS], dtype=np.float32
            )
            buffer.append(features)

            if len(buffer) < WINDOW + 1:
                print(f"idx={packet['idx']:05d}: warming up ({len(buffer)}/{WINDOW + 1})")
                continue

            frames = np.array(buffer)                         # (W+1, F)
            diff   = (frames[1:] - frames[:-1]).astype(np.float32)  # (W, F) FOTD
            diff_norm = (diff - norm_mean) / norm_std

            tensor = (
                torch.tensor(diff_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            model.eval()
            with torch.no_grad():
                label = idx_to_class[model(tensor).argmax(dim=1).item()]

            print(f"idx={packet['idx']:05d}: {label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Train ---
    print("\nLoading training data...")
    X_train, y_train, class_to_idx, norm_mean, norm_std = build_dataset(
        TRAIN_DIR, fit_norm=True
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"  {num_classes} classes: {', '.join(sorted(class_to_idx))}")
    print(f"  Training windows: {len(X_train)}")

    train_loader = DataLoader(
        SensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    input_size = X_train.shape[2]
    model = OdorLSTM(
        input_size=input_size,
        hidden_size=HIDDEN,
        num_layers=N_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print(f"\n{'Epoch':>6}  {'Loss':>9}  {'Top-1':>7}  {'Top-5':>7}")
    print("-" * 38)

    best_top1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss, top1, top5 = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(f"  {epoch:>4}  {loss:>9.4f}  {top1 * 100:>6.2f}%  {top5 * 100:>6.2f}%")
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), CKPT_PATH)

    print(f"\nBest training Top-1: {best_top1 * 100:.2f}%")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    # --- Evaluate on test set ---
    print("\nLoading test data...")
    X_test, y_test, _, _, _ = build_dataset(
        TEST_DIRS, class_to_idx=class_to_idx,
        norm_mean=norm_mean, norm_std=norm_std, fit_norm=False,
    )
    print(f"  Test windows: {len(X_test)}")
    test_loader = DataLoader(
        SensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    _, top1, top5, preds, targets = eval_epoch(model, test_loader, criterion, device)
    print(f"\n{'=' * 40}")
    print(f"  Test Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Test Top-5 Accuracy: {top5 * 100:.2f}%")
    print(f"{'=' * 40}")

    test_label_ids = sorted(set(targets.tolist()))
    test_class_names = [idx_to_class[i] for i in test_label_ids]
    cm = confusion_matrix(targets, preds, labels=test_label_ids)
    col_w = max(len(n) for n in test_class_names) + 2
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(" " * col_w + "".join(f"{n:>{col_w}}" for n in test_class_names))
    for i, row in enumerate(cm):
        print(f"{test_class_names[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    # --- Live streaming ---
    try:
        run_live(model, idx_to_class, norm_mean, norm_std, device)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
