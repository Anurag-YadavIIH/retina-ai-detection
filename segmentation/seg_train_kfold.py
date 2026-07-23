# segmentation/seg_train_kfold.py
#
# 5-fold cross-validation for Hard Exudate segmentation, replacing the
# single 45/9 split with a result that reports error bars. Motivation: two
# runs of the single-split trainer with "identical" hyperparameters produced
# val dice 0.6283 and 0.5264 — a single 9-image validation split has wide
# enough variance that one number overclaims. This trains 5 independent
# models, one per fold, and reports mean +/- std dice/IoU across them.
#
# The 27-image IDRiD test set is never touched here, same as seg_train.py.
# Does NOT touch model/exudate_unet.pth (the existing single-split
# checkpoint) — each fold gets its own model/exudate_unet_fold{k}.pth.
#
# Run with:   python -u segmentation/seg_train_kfold.py
# (the -u/unbuffered flag matters for a ~1 hour run: buffered stdout can be
# lost entirely if the process is interrupted before a normal exit.)

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from seg_dataset import IDRiDExudateDataset
from seg_model import build_unet, count_parameters
from seg_losses import BCEDiceLoss
from seg_train import (
    set_seed,
    AugmentedTrainSplit,
    run_training_loop,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOP_PATIENCE,
    RANDOM_SEED,
    DEVICE,
)

N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_DIR = Path(__file__).resolve().parent
RESULTS_JSON_PATH = SEGMENTATION_DIR / "kfold_results.json"
SUMMARY_PNG_PATH = SEGMENTATION_DIR / "kfold_summary.png"


def fold_checkpoint_path(fold_num):
    return PROJECT_ROOT / "model" / f"exudate_unet_fold{fold_num}.pth"


def plot_kfold_summary(fold_dices, mean_dice, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    folds = list(range(1, len(fold_dices) + 1))

    ax.bar(folds, fold_dices, color="#2563eb", alpha=0.85)
    ax.axhline(mean_dice, color="#dc2626", linestyle="--", label=f"Mean = {mean_dice:.4f}")

    ax.set_xlabel("Fold")
    ax.set_ylabel("Best Val Dice")
    ax.set_title(f"{N_SPLITS}-Fold CV — Hard Exudate Segmentation")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this training run is meant for the GPU.")

    set_seed(RANDOM_SEED)

    print("=" * 60)
    print(f"  Hard Exudate Segmentation — {N_SPLITS}-Fold Cross-Validation")
    print("=" * 60)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"Seed:   {RANDOM_SEED} (random, numpy, torch, torch.cuda; cudnn deterministic)")

    # --- The 54-image IDRiD TRAIN set only. The 27-image test set is never
    # touched here, and model/exudate_unet.pth (single-split checkpoint) is
    # never written to by this script. ---
    full_train = IDRiDExudateDataset(split="train")
    print(f"Total images: {len(full_train)}   ({N_SPLITS}-fold CV, test set untouched)")

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(kfold.split(range(len(full_train))))

    fold_results = []
    overall_start = time.time()

    for fold_num, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        print("\n" + "=" * 60)
        print(f"  Fold {fold_num}/{N_SPLITS}  (train={len(train_idx)}, val={len(val_idx)})")
        print("=" * 60, flush=True)

        train_subset = Subset(full_train, train_idx.tolist())
        val_subset = Subset(full_train, val_idx.tolist())
        train_dataset = AugmentedTrainSplit(train_subset)   # augmentation on train only
        val_dataset = val_subset                             # clean, no augmentation

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Fresh model each fold — no weights carried over from the previous fold.
        model = build_unet().to(DEVICE)
        if fold_num == 1:
            total, trainable = count_parameters(model)
            print(f"Model: U-Net (resnet34 encoder) — {total:,} total params, "
                  f"{trainable:,} trainable", flush=True)

        criterion = BCEDiceLoss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3,
        )

        checkpoint_path = fold_checkpoint_path(fold_num)
        fold_start = time.time()

        (history, best_val_dice, best_val_iou, best_epoch,
         stopped_early, _checkpoint_updated) = run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            checkpoint_path=checkpoint_path, num_epochs=NUM_EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE, log_prefix=f"[Fold {fold_num}] ",
        )

        fold_time = time.time() - fold_start
        print(f"[Fold {fold_num}] Done in {fold_time / 60:.1f} min "
              f"({'stopped early' if stopped_early else 'ran full duration'})")
        print(f"[Fold {fold_num}] Best val dice: {best_val_dice:.4f}  |  "
              f"Best val IoU: {best_val_iou:.4f}  |  Epoch: {best_epoch}", flush=True)

        fold_results.append({
            "fold": fold_num,
            "best_epoch": best_epoch,
            "val_dice": best_val_dice,
            "val_iou": best_val_iou,
            "stopped_early": stopped_early,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "checkpoint": str(checkpoint_path),
        })

    total_time = time.time() - overall_start

    # --- Aggregate report ---
    dices = np.array([r["val_dice"] for r in fold_results])
    ious = np.array([r["val_iou"] for r in fold_results])
    mean_dice, std_dice = float(dices.mean()), float(dices.std())
    mean_iou, std_iou = float(ious.mean()), float(ious.std())
    best_fold = max(fold_results, key=lambda r: r["val_dice"])

    print("\n" + "=" * 60)
    print(f"  {N_SPLITS}-Fold Cross-Validation — Summary")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes\n")

    print(f"{'Fold':<6}{'Best Epoch':<12}{'Val Dice':<12}{'Val IoU':<12}")
    print("-" * 42)
    for r in fold_results:
        print(f"{r['fold']:<6}{r['best_epoch']:<12}{r['val_dice']:<12.4f}{r['val_iou']:<12.4f}")
    print("-" * 42)

    print(f"\n5-fold CV Dice: {mean_dice:.3f} +/- {std_dice:.3f}  |  IoU: {mean_iou:.3f} +/- {std_iou:.3f}")

    print(f"\nBest fold: Fold {best_fold['fold']} (val_dice={best_fold['val_dice']:.4f}) — "
          f"checkpoint {best_fold['checkpoint']} is the candidate model for final test evaluation.")

    # --- Save results JSON ---
    results = {
        "n_splits": N_SPLITS,
        "seed": RANDOM_SEED,
        "folds": fold_results,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "best_fold": best_fold["fold"],
        "best_fold_checkpoint": best_fold["checkpoint"],
        "total_time_minutes": total_time / 60,
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON_PATH}")

    # --- Save summary plot ---
    plot_kfold_summary(list(dices), mean_dice, SUMMARY_PNG_PATH)
    print(f"Saved {SUMMARY_PNG_PATH}")


if __name__ == "__main__":
    main()
