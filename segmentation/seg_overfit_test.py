# segmentation/seg_overfit_test.py
#
# CRITICAL sanity test, run BEFORE any real training: overfit the U-Net on
# a single batch of 2 images. If the model/loss/backward path is wired up
# correctly, dice on those 2 images MUST climb toward ~0.9+ within ~100
# iterations, since there's nothing to generalize to — it just has to
# memorize 2 examples. If it doesn't, something upstream (model output
# shape, loss sign, label alignment, learning rate) is broken, and that
# needs fixing before scaling up to full training.
#
# Run with:   python segmentation/seg_overfit_test.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_dataset import IDRiDExudateDataset
from seg_model import build_unet, count_parameters
from seg_losses import BCEDiceLoss, dice_coefficient, iou_score
from seg_visualize import denormalize

NUM_ITERATIONS = 250
PRINT_EVERY = 25
LEARNING_RATE = 1e-3
OUTPUT_PATH = Path(__file__).resolve().parent / "overfit_check.png"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this test is meant to run on the GPU.")
    device = torch.device("cuda")

    # --- Single batch of 2 training images+masks ---
    dataset = IDRiDExudateDataset(split="train")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    sample_ids = [dataset.image_id_at(i) for i in range(2)]
    print(f"Overfitting on samples: {[f'IDRiD_{i:02d}' for i in sample_ids]}")

    # --- Model ---
    model = build_unet().to(device)
    total, trainable = count_parameters(model)
    print(f"Model: U-Net (resnet34 encoder) — {total:,} total params, {trainable:,} trainable")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Overfit loop ---
    model.train()
    print(f"\nTraining on {len(images)} images for {NUM_ITERATIONS} iterations...")
    print("-" * 60)

    for it in range(1, NUM_ITERATIONS + 1):
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        if it == 1 or it % PRINT_EVERY == 0:
            dice = dice_coefficient(preds, masks)
            iou = iou_score(preds, masks)
            print(f"Iter {it:3d}/{NUM_ITERATIONS} | loss={loss.item():.4f} | "
                  f"dice={dice:.4f} | iou={iou:.4f}")

    final_dice = dice_coefficient(preds, masks)
    print("-" * 60)
    print(f"Final dice: {final_dice:.4f}")
    if final_dice >= 0.9:
        print("PASS: dice reached >= 0.9 — model/loss/backward path checks out.")
    else:
        print("WARNING: dice did NOT reach 0.9 — investigate before scaling up.")

    # --- Predictions for the figure ---
    model.eval()
    with torch.no_grad():
        pred_logits = model(images)
        pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()

    # --- Save [fundus | true mask | predicted mask] figure ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for row in range(2):
        image_np = denormalize(images[row].cpu())
        true_mask_np = masks[row].squeeze(0).cpu().numpy()
        pred_mask_np = pred_masks[row].squeeze(0).cpu().numpy()
        image_id = sample_ids[row]

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"IDRiD_{image_id:02d} - Fundus")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(true_mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"IDRiD_{image_id:02d} - True Mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"IDRiD_{image_id:02d} - Predicted (overfit)")
        axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=120)
    plt.close(fig)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
