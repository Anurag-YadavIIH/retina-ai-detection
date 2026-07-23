# segmentation/seg_visualize.py
#
# Sanity-check the segmentation data pipeline: loads a few train samples
# through IDRiDExudateDataset and saves a figure showing
# [original fundus | mask | mask overlaid in red on fundus] per sample.
# This proves images and masks are correctly paired and spatially aligned.
#
# Run with:   python segmentation/seg_visualize.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch

from seg_dataset import IDRiDExudateDataset, IMAGENET_MEAN, IMAGENET_STD

NUM_SAMPLES = 3
OUTPUT_PATH = Path(__file__).resolve().parent / "pipeline_check.png"
OVERLAY_ALPHA = 0.5


def denormalize(image_tensor):
    """Undo ImageNet normalization for display purposes only."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image_tensor * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()  # CHW -> HWC for matplotlib


def main():
    dataset = IDRiDExudateDataset(split="train")

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, 4 * NUM_SAMPLES))

    for row in range(NUM_SAMPLES):
        image_tensor, mask_tensor = dataset[row]
        image_id = dataset.image_id_at(row)

        image_np = denormalize(image_tensor)
        mask_np = mask_tensor.squeeze(0).numpy()  # [512, 512], values in {0,1}

        unique_vals = np.unique(mask_np)
        fg_fraction = mask_np.mean()
        print(f"Sample IDRiD_{image_id:02d}: mask unique values = {unique_vals}, "
              f"foreground fraction = {fg_fraction * 100:.3f}%")

        # Red overlay: alpha-blend red onto the fundus wherever mask == 1
        red = np.array([1.0, 0.0, 0.0])
        mask_3d = mask_np[:, :, None]
        overlay = image_np * (1 - mask_3d * OVERLAY_ALPHA) + red * (mask_3d * OVERLAY_ALPHA)

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"IDRiD_{image_id:02d} - Original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"IDRiD_{image_id:02d} - Hard Exudate Mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"IDRiD_{image_id:02d} - Overlay")
        axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=120)
    plt.close(fig)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
