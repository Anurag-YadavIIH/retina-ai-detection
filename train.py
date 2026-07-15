# train.py
#
# This file defines the model architecture and trains it.
# Run this file to produce: model/retina_model.pth
#
# Usage:
#   python train.py

import os
import copy
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models

from utils.preprocess import (
    get_train_transforms,
    get_val_transforms,
    CLASS_NAMES,
)

# ─────────────────────────────────────────────
# CONFIGURATION — tweak these as needed
# ─────────────────────────────────────────────

DATASET_PATH  = "dataset"   # path to your dataset folder
MODEL_SAVE_PATH = "model/retina_model.pth"          # baseline checkpoint — NOT overwritten by this run
WEIGHTED_MODEL_SAVE_PATH = "model/retina_model_weighted.pth"  # class-weighted-loss experiment output
NUM_CLASSES   = 5                  # No_DR, Mild, Moderate, Severe, Proliferative
BATCH_SIZE    = 32                 # images processed together; lower if you run out of RAM
NUM_EPOCHS    = 10                 # how many full passes through the dataset
LEARNING_RATE = 0.001              # how fast the model adjusts weights
VAL_SPLIT     = 0.2               # 20% of data used for validation
RANDOM_SEED   = 42                 # for reproducibility


# ─────────────────────────────────────────────
# DEVICE SETUP
# Uses GPU automatically if available, otherwise CPU.
# GPU trains ~10x faster, but CPU works fine for this project.
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# torchvision's ImageFolder reads folders like:
#   dataset/train/No_DR/img001.png  → label 0
#   dataset/train/Mild/img002.png   → label 1
# It assigns integer labels automatically based on folder name order.
# ─────────────────────────────────────────────

def compute_class_weights(full_dataset):
    """
    Inverse-frequency class weights computed from the FULL (pre-split)
    dataset, so the weighting reflects the true class distribution rather
    than whatever happens to land in the train split:

        weight[c] = total / (num_classes * count[c])

    Then normalised so the weights average to 1.0 (keeps the overall loss
    magnitude comparable to the unweighted baseline run).
    """
    num_classes = len(full_dataset.classes)
    counts = Counter(full_dataset.targets)
    total = len(full_dataset.targets)

    raw_weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    mean_weight = sum(raw_weights) / num_classes
    weights = [w / mean_weight for w in raw_weights]

    print("\nClass weights (inverse-frequency, normalised to mean 1.0):")
    for i, name in enumerate(full_dataset.classes):
        print(f"  {name:<15} count={counts[i]:>5}  weight={weights[i]:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def load_datasets(dataset_path):
    """Load and split the dataset into train and validation sets."""

    # Load the full dataset with TRAINING transforms (augmentation on)
    full_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=get_train_transforms()
    )

    print(f"Total images found: {len(full_dataset)}")
    print(f"Classes detected:   {full_dataset.classes}")

    # Compute class weights from the FULL dataset, before splitting
    class_weights = compute_class_weights(full_dataset)

    # Split: 80% train, 20% validation
    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    # random_split randomly assigns images to each split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)  # reproducible split
    )

    # Override val transforms — no augmentation for validation
    # We do this by accessing the underlying dataset and transforms
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = get_val_transforms()

    print(f"Training images:    {train_size}")
    print(f"Validation images:  {val_size}")

    return train_dataset, val_dataset, class_weights


# ─────────────────────────────────────────────
# STEP 2: CREATE DATA LOADERS
# DataLoader feeds batches of images into the model during training.
# Instead of loading 3662 images at once (crashes RAM),
# it loads BATCH_SIZE images at a time.
# ─────────────────────────────────────────────

def create_dataloaders(train_dataset, val_dataset):
    """Wrap datasets in DataLoaders for efficient batch loading."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,        # shuffle each epoch so the model sees images in different order
        num_workers=2,       # parallel loading threads (use 0 on Windows if errors occur)
        pin_memory=True      # speeds up CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,       # no need to shuffle validation
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────
# STEP 3: BUILD THE MODEL
# We take ResNet18 with pre-trained ImageNet weights,
# then replace its final layer for our 5-class problem.
# ─────────────────────────────────────────────

def build_model(num_classes):
    """
    Load pre-trained ResNet18 and adapt it for retinal disease classification.
    """

    # Download ResNet18 pre-trained on ImageNet (first run downloads ~45MB)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # --- FREEZE all existing layers ---
    # "Freezing" means: don't update these weights during training.
    # We're keeping all the visual knowledge ResNet already learned.
    for param in model.parameters():
        param.requires_grad = False   # requires_grad=False means "don't train this"

    # --- REPLACE the final fully-connected layer ---
    # ResNet18's original final layer: fc = Linear(512, 1000)  → 1000 ImageNet classes
    # We replace it with:             fc = Linear(512, 5)      → our 5 DR classes
    #
    # model.fc.in_features gives us 512 (the input size, which we must keep the same)
    in_features = model.fc.in_features   # 512

    model.fc = nn.Linear(in_features, num_classes)
    # This new layer has requires_grad=True by default → it WILL be trained

    # Move the entire model to GPU (if available) or keep on CPU
    model = model.to(device)

    print(f"\nModel: ResNet18 (transfer learning)")
    print(f"  Frozen layers:     all except final fc layer")
    print(f"  Output classes:    {num_classes}")
    print(f"  Trainable params:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Frozen params:     {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    return model
# ─────────────────────────────────────────────
# STEP 4: LOSS FUNCTION AND OPTIMISER
# ─────────────────────────────────────────────

def build_optimizer(model, class_weights):
    """
    CrossEntropyLoss: standard for multi-class classification.
    Weighted by inverse class frequency (see compute_class_weights) so the
    minority Severe / Proliferate_DR classes contribute proportionally more
    to the loss instead of being drowned out by No_DR/Moderate.
    Adam optimiser: self-adjusting learning rate, works great for most tasks.
    We only pass parameters that require gradients (our new fc layer).
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Only optimise the trainable parameters (the new final layer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler: reduce LR by half if val loss stops improving
    # This helps squeeze out extra accuracy in later epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # reduce when val_loss stops going down
        factor=0.5,       # multiply LR by 0.5
        patience=2,       # wait 2 epochs before reducing
    )

    return criterion, optimizer, scheduler


# ─────────────────────────────────────────────
# STEP 5: TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Run one full pass through the TRAINING data.
    Returns average loss and accuracy for this epoch.
    """
    model.train()   # puts model in training mode (enables dropout, batchnorm updates)

    running_loss     = 0.0
    correct          = 0
    total            = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to the same device as the model (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward pass ---
        optimizer.zero_grad()          # clear gradients from previous batch
        outputs = model(images)        # shape: [batch_size, num_classes]
        loss    = criterion(outputs, labels)  # scalar value — how wrong we were

        # --- Backward pass ---
        loss.backward()     # compute gradients via backpropagation
        optimizer.step()    # update weights using those gradients

        # --- Track metrics ---
        running_loss += loss.item() * images.size(0)   # accumulate total loss
        _, predicted  = outputs.max(dim=1)             # class with highest score
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"| Acc: {100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc  = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    """
    Run one full pass through the VALIDATION data.
    No gradient updates — we're just measuring performance.
    """
    model.eval()   # puts model in evaluation mode (disables dropout etc.)

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():   # disable gradient tracking — saves memory, speeds up
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(dim=1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = 100. * correct / total
    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────
# STEP 6: MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────

def train_model():
    """
    Full training pipeline: load data → build model → train → save best model.
    """

    print("=" * 55)
    print("  Retinal Disease Detection — Model Training")
    print("=" * 55)

    # 1. Load data
    train_dataset, val_dataset, class_weights = load_datasets(DATASET_PATH)
    train_loader, val_loader   = create_dataloaders(train_dataset, val_dataset)

    # 2. Build model
    model = build_model(NUM_CLASSES)

    # 3. Build optimizer
    criterion, optimizer, scheduler = build_optimizer(model, class_weights)

    # 4. Track the best model so far
    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())  # save weights as a dict

    # History for plotting later
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Step the scheduler (reduce LR if val_loss isn't improving)
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\n  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  *** New best model saved (val acc: {val_acc:.2f}%) ***")

        print()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # 5. Restore and save best weights
    # NOTE: saved ONLY to WEIGHTED_MODEL_SAVE_PATH for this class-weighted-loss
    # experiment — MODEL_SAVE_PATH (the baseline checkpoint) is intentionally
    # left untouched so before/after per-class metrics stay comparable.
    model.load_state_dict(best_model_wts)
    os.makedirs("model", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names":      CLASS_NAMES,
        "num_classes":      NUM_CLASSES,
        "val_accuracy":     best_val_acc,
    }, WEIGHTED_MODEL_SAVE_PATH)

    print(f"\nModel saved to: {WEIGHTED_MODEL_SAVE_PATH}")

    # 6. Plot training curves
    plot_training_history(history)

    return model, history


# ─────────────────────────────────────────────
# STEP 7: PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_training_history(history):
    """Save a loss/accuracy plot — great for your portfolio README."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("static/training_curves.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Training curves saved to static/training_curves.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_model()