# evaluate.py
#
# Offline evaluation of an already-trained checkpoint (default:
# model/retina_model.pth). Does NOT retrain anything — it rebuilds the exact
# validation split that train.py used (same ImageFolder order, same
# random_split seed) and runs inference over it to produce clinically
# meaningful metrics.
#
# Run with:   python evaluate.py
#         or: python evaluate.py --model model/retina_model_weighted.pth

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from utils.preprocess import get_val_transforms, CLASS_NAMES, CLASS_LABELS

# ─────────────────────────────────────────────
# CONFIGURATION — must mirror train.py exactly so the val split matches
# ─────────────────────────────────────────────

DATASET_PATH  = "dataset"
MODEL_PATH    = "model/retina_model.pth"   # default checkpoint; override with --model
BASELINE_STEM = "retina_model"             # checkpoint filename (no .pth) that gets NO suffix
NUM_CLASSES   = 5
BATCH_SIZE    = 32
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42

TARGET_SENSITIVITY = 0.90   # minimum sensitivity required for the operating point

# Referable DR, defined by class NAME (never by list index)
REFERABLE_CLASS_NAMES = {"Moderate", "Severe", "Proliferate_DR"}

# True clinical severity order (increasing severity) — distinct from the
# alphabetical CLASS_NAMES order used for training/indexing. Used to make
# the confusion matrix, classification report, and quadratic-weighted
# kappa reflect real ordinal distance between grades.
SEVERITY_ORDER_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def derive_output_suffix(model_path):
    """
    Derive an output-file suffix from the checkpoint filename so different
    checkpoints don't clobber each other's plots/JSON.

    "model/retina_model.pth"          -> ""            (baseline, unchanged filenames)
    "model/retina_model_weighted.pth" -> "_weighted"
    "model/some_other_name.pth"       -> "_some_other_name"  (generic fallback)
    """
    stem = Path(model_path).stem
    if stem == BASELINE_STEM:
        return ""
    if stem.startswith(BASELINE_STEM):
        return stem[len(BASELINE_STEM):]
    return "_" + stem


# ─────────────────────────────────────────────
# STEP 1: REBUILD THE EXACT VALIDATION SPLIT
# ─────────────────────────────────────────────

def build_val_dataset():
    """
    Rebuild the same train/val split as train.py:
      - Same ImageFolder root and (therefore) the same sample ordering
      - Same VAL_SPLIT / RANDOM_SEED / random_split call
    Uses validation transforms (no augmentation) directly, which is fine
    because transforms don't affect ImageFolder's sample ordering — only
    the random_split indices matter for reproducing the same split.
    """
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=get_val_transforms())

    if full_dataset.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Dataset folder order {full_dataset.classes} does not match "
            f"utils.preprocess.CLASS_NAMES {CLASS_NAMES}. The trained model "
            f"expects this exact order — investigate before trusting any "
            f"metrics below."
        )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    return full_dataset, val_dataset


# ─────────────────────────────────────────────
# STEP 2: LOAD THE TRAINED MODEL (robust to both checkpoint formats)
# ─────────────────────────────────────────────

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────
# STEP 3: RUN INFERENCE OVER THE VALIDATION SET
# ─────────────────────────────────────────────

def run_inference(model, val_dataset):
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_true.append(labels.numpy())
            all_pred.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_pred)
    y_probs = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_probs


# ─────────────────────────────────────────────
# STEP 4: PLOTS
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Validation Set (clinical severity order)")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(y_true_bin, y_score, roc_auc_value, pr_auc_value, out_path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color="#2563eb", label=f"ROC-AUC = {roc_auc_value:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Referable-DR ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(recall, precision, color="#dc2626", label=f"PR-AUC = {pr_auc_value:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Referable-DR Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline evaluation of a trained DR classifier checkpoint."
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to the model checkpoint (default: {MODEL_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = derive_output_suffix(args.model)

    json_path = f"evaluation_results{suffix}.json"
    cm_path   = f"static/confusion_matrix{suffix}.png"
    roc_path  = f"static/roc_pr_curves{suffix}.png"

    os.makedirs("static", exist_ok=True)

    print("=" * 60)
    print("  Diabetic Retinopathy Model - Offline Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.model}")

    full_dataset, val_dataset = build_val_dataset()

    print(f"\nTechnical class order (ImageFolder / CLASS_NAMES): {full_dataset.classes}")
    print(f"Clinical severity order (used for CM/report/kappa): "
          f"{[CLASS_LABELS[n] for n in SEVERITY_ORDER_NAMES]}")
    print(f"Total images:      {len(full_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    model = load_model(args.model)
    print(f"\nModel loaded from {args.model}")

    y_true, y_pred, y_probs = run_inference(model, val_dataset)

    # ---- Confusion matrix + classification report, clinical severity order ----
    severity_idx = [CLASS_NAMES.index(name) for name in SEVERITY_ORDER_NAMES]

    cm = confusion_matrix(y_true, y_pred, labels=severity_idx)
    print("\n" + "-" * 60)
    print(f"Confusion Matrix (rows=true, cols=predicted)")
    print(f"Order: {SEVERITY_ORDER_NAMES}")
    print("-" * 60)
    print(cm)

    report_str = classification_report(
        y_true, y_pred, labels=severity_idx, target_names=SEVERITY_ORDER_NAMES,
        digits=3, zero_division=0,
    )
    print("\n" + "-" * 60)
    print("Per-Class Precision / Recall / F1 / Support")
    print("-" * 60)
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred, labels=severity_idx, target_names=SEVERITY_ORDER_NAMES,
        digits=3, zero_division=0, output_dict=True,
    )

    qwk = cohen_kappa_score(y_true, y_pred, labels=severity_idx, weights="quadratic")
    overall_acc = float((y_true == y_pred).mean())

    print("-" * 60)
    print(f"Quadratic-Weighted Cohen's Kappa: {qwk:.4f}")
    print(f"Overall accuracy (context only):  {overall_acc * 100:.2f}%")

    # ---- Referable-DR screening metrics ----
    referable_idx = [CLASS_NAMES.index(n) for n in REFERABLE_CLASS_NAMES]
    y_true_bin = np.isin(y_true, referable_idx).astype(int)
    y_score    = y_probs[:, referable_idx].sum(axis=1)

    roc_auc_value = roc_auc_score(y_true_bin, y_score)
    pr_auc_value  = average_precision_score(y_true_bin, y_score)

    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
    candidates = np.where(tpr >= TARGET_SENSITIVITY)[0]
    best_i = int(candidates[0]) if len(candidates) > 0 else int(np.argmax(tpr))

    operating_threshold = float(thresholds[best_i])
    sensitivity          = float(tpr[best_i])
    specificity          = float(1 - fpr[best_i])
    referable_prevalence = float(y_true_bin.mean())

    print("\n" + "-" * 60)
    print("Referable-DR Screening (referable = Moderate DR, Severe DR, Proliferate DR)")
    print("-" * 60)
    print(f"Referable prevalence:    {referable_prevalence * 100:.1f}%")
    print(f"ROC-AUC:                 {roc_auc_value:.4f}")
    print(f"PR-AUC:                  {pr_auc_value:.4f}")
    print(f"Operating threshold:     {operating_threshold:.4f}  "
          f"(targeting >= {TARGET_SENSITIVITY * 100:.0f}% sensitivity)")
    print(f"Sensitivity @ threshold: {sensitivity * 100:.2f}%")
    print(f"Specificity @ threshold: {specificity * 100:.2f}%")

    # ---- Plots ----
    plot_confusion_matrix(cm, SEVERITY_ORDER_NAMES, cm_path)
    plot_roc_pr(y_true_bin, y_score, roc_auc_value, pr_auc_value, roc_path)
    print(f"\nSaved {cm_path}")
    print(f"Saved {roc_path}")

    # ---- Save raw numbers ----
    results = {
        "checkpoint":            args.model,
        "class_order_technical": CLASS_NAMES,
        "class_order_clinical":  SEVERITY_ORDER_NAMES,
        "val_set_size":          len(val_dataset),
        "total_dataset_size":    len(full_dataset),
        "confusion_matrix":            cm.tolist(),
        "confusion_matrix_label_order": SEVERITY_ORDER_NAMES,
        "classification_report":      report_dict,
        "quadratic_weighted_kappa":   qwk,
        "overall_accuracy":           overall_acc,
        "referable_dr": {
            "referable_classes":       sorted(REFERABLE_CLASS_NAMES),
            "prevalence":              referable_prevalence,
            "roc_auc":                 roc_auc_value,
            "pr_auc":                  pr_auc_value,
            "operating_threshold":     operating_threshold,
            "target_sensitivity":      TARGET_SENSITIVITY,
            "sensitivity_at_threshold": sensitivity,
            "specificity_at_threshold": specificity,
        },
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
