# segmentation/seg_train.py
#
# Real training loop for Hard Exudate segmentation. The model/loss/backward
# path was already verified by seg_overfit_test.py (dice 0.986 on 2 images).
# This trains on the full 45-image train split, validates on 9 held-out
# images from the SAME 54-image IDRiD training set, and never touches the
# 27-image IDRiD test set — that stays held out for a separate final
# evaluation step.
#
# Run with:   python segmentation/seg_train.py

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from seg_dataset import IDRiDExudateDataset
from seg_model import build_unet, count_parameters
from seg_losses import BCEDiceLoss, dice_coefficient, iou_score

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VAL_SIZE      = 9          # 45 train / 9 val out of the 54-image train split
RANDOM_SEED   = 42
BATCH_SIZE    = 2          # safe for 4GB VRAM (confirmed by the overfit test)
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-4       # lower than the overfit test's 1e-3 — real training needs stability
EARLY_STOP_PATIENCE = 15   # stop if val dice doesn't improve for this many consecutive epochs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / "model" / "exudate_unet.pth"
CURVES_PATH = Path(__file__).resolve().parent / "seg_training_curves.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Seed every source of randomness this training run touches: the 90-degree
    rotation / flip choices in AugmentedTrainSplit (Python's random module),
    the U-Net decoder's random weight init and DataLoader(shuffle=True)'s
    batch ordering (torch's RNG, CPU and CUDA), plus cuDNN's algorithm
    selection (which is non-deterministic by default even with a fixed seed
    elsewhere). Without this, two runs with identical hyperparameters can
    diverge in training trajectory and final plateau — which is exactly what
    happened between the first two training runs on this dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# TRAIN-ONLY AUGMENTATION
#
# Fundus images have no canonical orientation, so horizontal flip, vertical
# flip, and rotation are all valid. Rotation is restricted to exact 90-degree
# multiples (torch.rot90) rather than arbitrary angles: an arbitrary-angle
# rotation needs interpolation and a border fill value, which risks
# corrupting the mask's binary values — the same reason seg_dataset.py uses
# nearest-neighbor (not bilinear) when resizing masks. 90-degree rotation is
# an exact pixel permutation, so it stays perfectly binary with zero
# interpolation artifacts. The same random transform is applied to both the
# image and mask tensors so they stay spatially aligned.
# ─────────────────────────────────────────────

class AugmentedTrainSplit(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, mask = self.subset[index]

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])   # horizontal flip (width)
            mask = torch.flip(mask, dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])   # vertical flip (height)
            mask = torch.flip(mask, dims=[1])

        k = random.randint(0, 3)                  # 0/90/180/270 degrees
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return image, mask


# ─────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n += images.size(0)

    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dice_sum += dice_coefficient(preds, masks) * batch_size
        iou_sum += iou_score(preds, masks) * batch_size
        n += batch_size

    return running_loss / n, dice_sum / n, iou_sum / n


# ─────────────────────────────────────────────
# SHARED TRAIN-WITH-EARLY-STOPPING-AND-SAFE-CHECKPOINTING LOOP
#
# Used by both this single-split trainer and each fold of
# seg_train_kfold.py's cross-validation. Kept as one function rather than
# duplicated per-fold: this logic already had one real bug (a run overwriting
# a previous run's better checkpoint) fixed in it, and duplicating
# bug-prone logic across files is how fixes drift out of sync.
# ─────────────────────────────────────────────

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       checkpoint_path, num_epochs, early_stop_patience, log_prefix=""):
    """
    Trains `model`, evaluating on val_loader each epoch, with:
      - ReduceLROnPlateau stepping on val dice
      - early stopping after `early_stop_patience` epochs without val dice improvement
      - checkpointing to checkpoint_path that NEVER overwrites a better val_dice
        already saved there (reads the existing file, if any, as a floor)

    Returns (history, run_best_val_dice, run_best_val_iou, run_best_epoch,
             stopped_early, checkpoint_updated).
    """
    disk_best_val_dice = -1.0
    if checkpoint_path.is_file():
        try:
            existing_ckpt = torch.load(checkpoint_path, map_location="cpu")
            disk_best_val_dice = existing_ckpt.get("val_dice", -1.0)
            print(f"{log_prefix}Existing checkpoint found: val_dice={disk_best_val_dice:.4f} "
                  f"(epoch {existing_ckpt.get('epoch')}) — this run will only overwrite it "
                  f"if it beats that.", flush=True)
        except Exception as e:
            print(f"{log_prefix}Warning: could not read existing checkpoint ({e}); treating as none.",
                  flush=True)
    else:
        print(f"{log_prefix}No existing checkpoint found.", flush=True)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    run_best_val_dice = -1.0
    run_best_val_iou = None
    run_best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_updated = False

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion)
        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"{log_prefix}Epoch {epoch:3d}/{num_epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}", flush=True)

        if val_dice > run_best_val_dice:
            run_best_val_dice = val_dice
            run_best_val_iou = val_iou
            run_best_epoch = epoch
            epochs_without_improvement = 0

            if val_dice > disk_best_val_dice:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                }, checkpoint_path)
                disk_best_val_dice = val_dice
                checkpoint_updated = True
                print(f"{log_prefix}  *** New best model saved (val_dice={val_dice:.4f}) -> "
                      f"{checkpoint_path} ***", flush=True)
            else:
                print(f"{log_prefix}  (New best for this run: val_dice={val_dice:.4f}, but existing "
                      f"checkpoint ({disk_best_val_dice:.4f}) is still better — NOT overwriting)", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"{log_prefix}Early stopping: val dice has not improved for "
                      f"{early_stop_patience} consecutive epochs (since epoch {run_best_epoch}).",
                      flush=True)
                stopped_early = True
                break

    return history, run_best_val_dice, run_best_val_iou, run_best_epoch, stopped_early, checkpoint_updated


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_dice"], "g-o", label="Val Dice", markersize=3)
    ax2.plot(epochs, history["val_iou"], "m-o", label="Val IoU", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Dice / IoU")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this training run is meant for the GPU.")

    set_seed(RANDOM_SEED)

    print("=" * 60)
    print("  Hard Exudate Segmentation — Training")
    print("=" * 60)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"Seed:   {RANDOM_SEED} (random, numpy, torch, torch.cuda; cudnn deterministic)")

    # --- Data: 45/9 split of the 54-image IDRiD TRAIN set only.
    # The 27-image IDRiD test set is never touched here. ---
    full_train = IDRiDExudateDataset(split="train")
    train_size = len(full_train) - VAL_SIZE
    train_subset, val_subset = random_split(
        full_train,
        [train_size, VAL_SIZE],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_dataset = AugmentedTrainSplit(train_subset)   # augmentation on train only
    val_dataset = val_subset                             # clean, no augmentation

    print(f"Train images: {len(train_dataset)}   Val images: {len(val_dataset)}"
          f"   (split seed={RANDOM_SEED}, test set untouched)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- Model / loss / optimizer ---
    model = build_unet().to(DEVICE)
    total, trainable = count_parameters(model)
    print(f"Model: U-Net (resnet34 encoder) — {total:,} total params, {trainable:,} trainable")

    criterion = BCEDiceLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
          f"early stop patience={EARLY_STOP_PATIENCE})...\n")
    start_time = time.time()

    (history, run_best_val_dice, run_best_val_iou, run_best_epoch,
     stopped_early, checkpoint_updated) = run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        checkpoint_path=MODEL_SAVE_PATH, num_epochs=NUM_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes "
          f"({'stopped early' if stopped_early else 'ran full duration'})")

    plot_training_curves(history, CURVES_PATH)
    print(f"Saved {CURVES_PATH}")

    current_disk_val_dice = torch.load(MODEL_SAVE_PATH, map_location="cpu").get("val_dice", -1.0)

    print("\n" + "-" * 60)
    print(f"This run's best val dice: {run_best_val_dice:.4f}  |  val IoU: {run_best_val_iou:.4f}  "
          f"|  Epoch: {run_best_epoch}")
    if checkpoint_updated:
        print(f"Checkpoint updated: YES — new on-disk best is {current_disk_val_dice:.4f} -> {MODEL_SAVE_PATH}")
    else:
        print(f"Checkpoint updated: NO — on-disk checkpoint (val_dice={current_disk_val_dice:.4f}) "
              f"was better and was kept unchanged")
    print(f"Stopped early: {stopped_early}  "
          f"({'yes — no improvement for ' + str(EARLY_STOP_PATIENCE) + ' epochs' if stopped_early else 'no — completed all ' + str(NUM_EPOCHS) + ' epochs'})")


if __name__ == "__main__":
    main()
