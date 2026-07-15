# diagnose_split.py
#
# Diagnostic-only: rebuilds the exact train/val split that train.py and
# evaluate.py use, and reports per-class image counts in each subset.
# Does not load or touch the model.
#
# Run with:   python diagnose_split.py

from collections import Counter

import torch
from torch.utils.data import random_split
from torchvision import datasets

from utils.preprocess import get_val_transforms, CLASS_NAMES

DATASET_PATH = "dataset"
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

LOW_SUPPORT_THRESHOLD = 20


def class_counts(subset):
    """Count images per class within a random_split Subset."""
    full_targets = subset.dataset.targets
    counts = Counter(full_targets[i] for i in subset.indices)
    return counts


def print_subset_report(name, subset):
    counts = class_counts(subset)
    total = len(subset)

    print(f"\n{name} subset - {total} images")
    print("-" * 45)
    for idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(idx, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        flag = ""
        if name.lower() == "validation" and count < LOW_SUPPORT_THRESHOLD:
            flag = "  <-- low support - metrics unreliable"
        print(f"  {class_name:<15} {count:>5}  ({pct:5.1f}%){flag}")


def main():
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=get_val_transforms())

    if full_dataset.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Dataset folder order {full_dataset.classes} does not match "
            f"utils.preprocess.CLASS_NAMES {CLASS_NAMES}."
        )

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print("=" * 60)
    print("  Train/Validation Split - Class Distribution Diagnosis")
    print("=" * 60)
    print(f"Class order: {CLASS_NAMES}")
    print(f"Total images: {len(full_dataset)}")
    print(f"Split: {int((1 - VAL_SPLIT) * 100)}% train / {int(VAL_SPLIT * 100)}% val "
          f"(seed={RANDOM_SEED})")

    print_subset_report("Train", train_dataset)
    print_subset_report("Validation", val_dataset)


if __name__ == "__main__":
    main()
