"""
evaluate_model.py — universal model evaluation graphs for NoseAI

Usage:
    python evaluate_model.py <module.ClassName> <model_path> [--name NAME] [--out DIR]

    Example:
        python evaluate_model.py models.mlr.MLRModel MLR-Model/mlr.pkl --name MLR

Requirements:
    - The model class must implement the Model ABC (models/model_abc.py):
        - ModelClass.load(path)  (classmethod that returns a fitted model instance)
        - model.predict(X)       (takes np.ndarray, returns array of class name strings)
        - model.feature_columns  (list of feature column names)
    - Optional extras (graphs are skipped if absent):
        - model.loss_history     (list of per-epoch losses → loss curve)
        - model.W                (weight matrix → feature importance, MLR-specific)
        - model.predict_proba(X) (returns list of {class: prob} dicts → top-k accuracy bars)

Outputs:
    PNG files saved to --out directory (default: <name>_graphs/ next to the model file)
"""

import argparse
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT   = os.path.dirname(os.path.abspath(__file__))
TARGET = "type"

NUT_CLASSES    = {"almond","brazil_nut","cashew","chestnuts","hazelnut","peanuts",
                  "pecans","pili_nut","pistachios","walnuts"}
SPICE_CLASSES  = {"allspice","angelica","chamomile","chervil","chives","cinnamon",
                  "cloves","coriander","cumin","dill","garlic","ginger","mint",
                  "mugwort","mustard","nutmeg","oregano","saffron","star_anise"}
FRUIT_CLASSES  = {"apple","avocado","banana","kiwi","lemon","mandarin_orange",
                  "mango","peach","pear","pineapple","strawberry"}
VEG_CLASSES    = {"asparagus","broccoli","brussel_sprouts","cabbage","cauliflower",
                  "potato","radish","sweet_potato","tomato","turnip"}
CATEGORIES     = {"Nuts": NUT_CLASSES, "Spices/Herbs": SPICE_CLASSES,
                  "Fruits": FRUIT_CLASSES, "Vegetables": VEG_CLASSES}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data…")
    train_df  = pd.read_csv(os.path.join(ROOT, "training-data.csv"))
    test_df   = pd.read_csv(os.path.join(ROOT, "testing-data.csv"))
    nuts_df   = pd.read_csv(os.path.join(ROOT, "online_nuts_validation.csv"))
    spices_df = pd.read_csv(os.path.join(ROOT, "online_spices_validation.csv"))
    return train_df, test_df, nuts_df, spices_df


def get_feature_cols(model, df):
    if hasattr(model, "feature_columns") and model.feature_columns is not None:
        return model.feature_columns
    return [c for c in df.columns if c != TARGET]


def predict_df(model, df, feature_cols):
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy()
    return np.array(model.predict(X)), y


def topk_acc(model, df, feature_cols, k):
    if not hasattr(model, "predict_proba"):
        return None
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy()
    proba_list = model.predict_proba(X)
    hits = sum(
        y[i] in sorted(p, key=p.get, reverse=True)[:k]
        for i, p in enumerate(proba_list)
    )
    return hits / len(y)


# ── figure builders ────────────────────────────────────────────────────────────

def fig_loss_curve(loss_history, out, name):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="#2563EB", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(f"{name} — Training Loss Curve", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Final loss: {loss_history[-1]:.4f}",
        xy=(len(loss_history), loss_history[-1]),
        xytext=(len(loss_history) * 0.65, loss_history[-1] + 0.15),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=11, color="#1e40af",
    )
    fig.tight_layout()
    path = os.path.join(out, "fig_loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig_accuracy_bars(metrics, overall_acc, out, name):
    """metrics: list of (label, value, color)"""
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    baseline = 1 / 50

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, [v * 100 for v in values], color=colors, width=0.5, zorder=3)
    ax.axhline(baseline * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Random baseline ({baseline * 100:.1f}%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"{name} — Accuracy Across Datasets", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(values) * 100 + 12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    path = os.path.join(out, "fig_accuracy_bars.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig_per_class_accuracy(per_class_acc, overall_acc, out, name):
    sorted_classes = sorted(per_class_acc, key=per_class_acc.get)
    sorted_vals    = [per_class_acc[c] for c in sorted_classes]
    bar_colors     = ["#ef4444" if v < 0.2 else "#f59e0b" if v < 0.4 else "#10b981"
                      for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(sorted_classes, [v * 100 for v in sorted_vals], color=bar_colors)
    ax.axvline(overall_acc * 100, color="#2563EB", linestyle="--", linewidth=1.5,
               label=f"Overall avg ({overall_acc * 100:.1f}%)")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{name} — Per-Class Accuracy (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    legend_patches = [
        mpatches.Patch(color="#ef4444", label="< 20%"),
        mpatches.Patch(color="#f59e0b", label="20–40%"),
        mpatches.Patch(color="#10b981", label="> 40%"),
        plt.Line2D([0], [0], color="#2563EB", linestyle="--",
                   label=f"Overall avg ({overall_acc * 100:.1f}%)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "fig_per_class_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig_confusion_matrix(test_true, test_preds, out, name):
    top20 = pd.Series(test_true).value_counts().head(20).index.tolist()
    mask  = np.isin(test_true, top20)
    cm    = confusion_matrix(test_true[mask], test_preds[mask], labels=top20)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_norm, annot=True, fmt=".0%", xticklabels=top20, yticklabels=top20,
                cmap="Blues", ax=ax, linewidths=0.3, annot_kws={"size": 7},
                vmin=0, vmax=1, cbar_kws={"label": "Recall"})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{name} — Confusion Matrix (Top 20 Classes, Normalised by True Label)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "fig_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig_category_accuracy(test_true, test_preds, overall_acc, out, name):
    group_acc = {}
    for gname, gset in CATEGORIES.items():
        mask = np.isin(test_true, list(gset))
        if mask.sum() > 0:
            group_acc[gname] = np.mean(test_preds[mask] == test_true[mask])

    if not group_acc:
        print("  Skipping category accuracy (no matching class names found)")
        return

    gnames = list(group_acc.keys())
    gvals  = [group_acc[g] * 100 for g in gnames]
    gcols  = ["#f59e0b", "#ef4444", "#10b981", "#6366f1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(gnames, gvals, color=gcols[:len(gnames)], width=0.5, zorder=3)
    ax.axhline(overall_acc * 100, color="#2563EB", linestyle="--", linewidth=1.5,
               label=f"Overall ({overall_acc * 100:.1f}%)")
    for bar, val in zip(bars, gvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{name} — Test Accuracy by Food Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(gvals) + 12)
    fig.tight_layout()
    path = os.path.join(out, "fig_category_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig_feature_importance(model, out, name):
    feature_importance = np.abs(model.W).mean(axis=1)
    feat_names = model.feature_columns

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(feat_names, feature_importance,
                  color=plt.cm.viridis(feature_importance / feature_importance.max()))
    ax.set_ylabel("Mean |Weight| Across Classes", fontsize=12)
    ax.set_title(f"{name} — Feature Importance (Weight Magnitudes)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, feature_importance):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "fig_feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation graphs for any NoseAI model.")
    parser.add_argument("model_class",
                        help="Dotted path to the model class, e.g. models.mlr.MLRModel")
    parser.add_argument("model_path", help="Path to the saved model file")
    parser.add_argument("--name", default=None,
                        help="Display name for the model (default: class name)")
    parser.add_argument("--out", default=None,
                        help="Output directory for graphs (default: <name>_graphs/ next to model file)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        print(f"Error: file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Dynamically import the model class
    module_path, class_name = args.model_class.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading class '{args.model_class}': {e}", file=sys.stderr)
        sys.exit(1)

    name = args.name or class_name
    out  = args.out  or os.path.join(os.path.dirname(model_path), f"{name}_graphs")
    os.makedirs(out, exist_ok=True)

    print(f"Loading model from {model_path}…")
    model = ModelClass.load(model_path)

    train_df, test_df, nuts_df, spices_df = load_data()
    feat_cols = get_feature_cols(model, train_df)

    print("Running predictions…")
    test_preds,   test_true   = predict_df(model, test_df,   feat_cols)
    nuts_preds,   nuts_true   = predict_df(model, nuts_df,   feat_cols)
    spices_preds, spices_true = predict_df(model, spices_df, feat_cols)

    test_acc   = np.mean(test_preds   == test_true)
    nuts_acc   = np.mean(nuts_preds   == nuts_true)
    spices_acc = np.mean(spices_preds == spices_true)

    print(f"Test accuracy:   {test_acc:.4f}")
    print(f"Nuts accuracy:   {nuts_acc:.4f}")
    print(f"Spices accuracy: {spices_acc:.4f}")

    top3 = topk_acc(model, test_df, feat_cols, 3)
    top5 = topk_acc(model, test_df, feat_cols, 5)
    if top3 is not None:
        print(f"Top-3 accuracy:  {top3:.4f}")
        print(f"Top-5 accuracy:  {top5:.4f}")

    print(f"\nGenerating graphs → {out}/")

    # Loss curve (optional)
    if hasattr(model, "loss_history") and model.loss_history:
        fig_loss_curve(model.loss_history, out, name)

    # Accuracy bars
    metrics = [
        ("Test Set\n(offline)",  test_acc,   "#2563EB"),
        ("Nuts\n(online)",       nuts_acc,   "#f59e0b"),
        ("Spices\n(online)",     spices_acc, "#ef4444"),
    ]
    if top3 is not None:
        metrics += [
            ("Top-3\n(test)", top3, "#10b981"),
            ("Top-5\n(test)", top5, "#8b5cf6"),
        ]
    fig_accuracy_bars(metrics, test_acc, out, name)

    # Per-class accuracy
    per_class_acc = {}
    for cls in np.unique(test_true):
        mask = test_true == cls
        per_class_acc[cls] = np.mean(test_preds[mask] == cls)
    fig_per_class_accuracy(per_class_acc, test_acc, out, name)

    # Confusion matrix
    fig_confusion_matrix(test_true, test_preds, out, name)

    # Category accuracy
    fig_category_accuracy(test_true, test_preds, test_acc, out, name)

    # Feature importance (MLR-specific)
    if hasattr(model, "W") and model.W is not None and hasattr(model, "feature_columns"):
        fig_feature_importance(model, out, name)

    print(f"\nDone! All graphs saved to: {out}/")


if __name__ == "__main__":
    main()
