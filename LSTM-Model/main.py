import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_DIR = "../smell-net/offline_testing"
TEST_DIRS = [
    "../smell-net/online_nuts",
    "../smell-net/online_spices",
]
CKPT_PATH = "best_model.pt"

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


def load_csvs_from_dir(root_dir):
    """Returns (df, class_name) pairs from all CSVs in each subdir."""
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
    """Frame-to-frame deltas."""
    arr = df.values.astype(np.float32)
    return arr[1:] - arr[:-1]   # shape: (T-1, F)


def sliding_windows(arr, window, stride):
    """Chop arr into overlapping windows of shape (window, F)."""
    T = arr.shape[0]
    windows = []
    for start in range(0, T - window + 1, stride):
        windows.append(arr[start : start + window])
    return windows


def build_dataset(root_dirs, class_to_idx=None, scaler=None, fit_scaler=False):
    """
    Load CSVs, apply FOTD + sliding window, then scale.
    Returns: X (N, W, F), y (N,), class_to_idx, scaler
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

    X = np.stack(all_windows, axis=0)          # (N, W, F)
    y = np.array(all_labels, dtype=np.int64)

    N, W, F = X.shape
    X_flat = X.reshape(N * W, F)

    if fit_scaler:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)

    X = X_flat.reshape(N, W, F).astype(np.float32)
    return X, y, class_to_idx, scaler


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
        pooled = out.mean(dim=1)         # mean pooled
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def topk_counts(logits, targets, k):
    """Count top-1 and top-k correct predictions."""
    with torch.no_grad():
        k_eff = min(k, logits.size(1))
        _, top1_pred = logits.topk(1, dim=1)
        top1_ok = top1_pred.squeeze(1).eq(targets).sum().item()
        _, topk_pred = logits.topk(k_eff, dim=1)
        topk_ok = topk_pred.eq(targets.view(-1, 1).expand_as(topk_pred)).any(dim=1).sum().item()
    return top1_ok, topk_ok


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
        t1, t5 = topk_counts(logits, y_b, k=5)
        top1_sum += t1
        top5_sum += t5

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
        t1, t5 = topk_counts(logits, y_b, k=5)
        top1_sum += t1
        top5_sum += t5

        _, top1_pred = logits.topk(1, dim=1)
        all_preds.extend(top1_pred.squeeze(1).cpu().numpy())
        all_targets.extend(y_b.cpu().numpy())

    return (
        total_loss / total,
        top1_sum / total,
        top5_sum / total,
        np.array(all_preds),
        np.array(all_targets),
    )


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("\nLoading training data...")
    X_train, y_train, class_to_idx, scaler = build_dataset(
        TRAIN_DIR, fit_scaler=True
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"  {num_classes} classes: {', '.join(sorted(class_to_idx))}")
    print(f"  Training windows: {len(X_train)}")

    train_loader = DataLoader(
        SensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print("\nLoading test data...")
    X_test, y_test, _, _ = build_dataset(
        TEST_DIRS, class_to_idx=class_to_idx, scaler=scaler, fit_scaler=False
    )
    print(f"  Test windows: {len(X_test)}")
    test_loader = DataLoader(
        SensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

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
    print(f"Checkpoint saved → {CKPT_PATH}")

    print("\nLoading best checkpoint for evaluation...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    _, top1, top5, preds, targets = eval_epoch(model, test_loader, criterion, device)

    print(f"\n{'=' * 40}")
    print(f"  Test Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Test Top-5 Accuracy: {top5 * 100:.2f}%")
    print(f"{'=' * 40}")

    # only show classes that appear in the test set
    test_label_ids = sorted(set(targets.tolist()))
    test_class_names = [idx_to_class[i] for i in test_label_ids]
    cm = confusion_matrix(targets, preds, labels=test_label_ids)

    col_w = max(len(n) for n in test_class_names) + 2
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in test_class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{test_class_names[i]:>{col_w}}" + "".join(
            f"{v:>{col_w}}" for v in row
        )
        print(row_str)


if __name__ == "__main__":
    main()
