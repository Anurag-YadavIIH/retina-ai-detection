# segmentation/seg_evaluate_test.py
#
# FINAL, one-time honest evaluation of Hard Exudate segmentation on the
# held-out 27-image IDRiD test set. Run exactly once — no tuning against
# this result. Uses model/exudate_unet_fold4.pth, the best-performing fold
# from 5-fold CV (val_dice=0.6492), as the single candidate model.
#
# Run with:   python segmentation/seg_evaluate_test.py

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_dataset import IDRiDExudateDataset
from seg_model import build_unet
from seg_losses import dice_coefficient, iou_score
from seg_train import DEVICE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "exudate_unet_fold4.pth"
RESULTS_JSON_PATH = SEGMENTATION_DIR / "test_results.json"
QUALITATIVE_PNG_PATH = SEGMENTATION_DIR / "test_qualitative.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(checkpoint_path):
    model = build_unet().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def denormalize(image_tensor):
    """Undo ImageNet normalization for display purposes only."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image_tensor * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def make_error_overlay(image_np, gt_np, pred_np):
    """
    TP=green, FP=red, FN=cyan, on a dimmed fundus. More informative than
    repeating the predicted mask alone in the 4th column — this shows
    exactly where the model over- or under-segments relative to ground
    truth, which matters most for the worst-case rows.
    """
    tp = (pred_np == 1) & (gt_np == 1)
    fp = (pred_np == 1) & (gt_np == 0)
    fn = (pred_np == 0) & (gt_np == 1)

    overlay = image_np * 0.5   # dim the fundus so overlay colors stand out
    overlay[tp] = [0.0, 1.0, 0.0]
    overlay[fp] = [1.0, 0.0, 0.0]
    overlay[fn] = [0.0, 1.0, 1.0]
    return overlay


@torch.no_grad()
def run_inference(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = []
    cache = {}   # image_id -> tensors/arrays needed for the qualitative figure

    for i, (image, mask) in enumerate(loader):
        image_id = dataset.image_id_at(i)
        image_gpu = image.to(DEVICE)
        mask_gpu = mask.to(DEVICE)

        pred_logits = model(image_gpu)
        dice = dice_coefficient(pred_logits, mask_gpu)
        iou = iou_score(pred_logits, mask_gpu)

        pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
        gt_mask = mask.squeeze(0).squeeze(0).numpy()

        results.append({"image_id": image_id, "dice": dice, "iou": iou})
        cache[image_id] = {
            "image": image.squeeze(0).cpu(),
            "gt_mask": gt_mask,
            "pred_mask": pred_mask,
        }

    return results, cache


def plot_qualitative(selected, cache, out_path):
    """selected: list of (label, image_id, dice) tuples."""
    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    for row, (label, image_id, dice) in enumerate(selected):
        data = cache[image_id]
        image_np = denormalize(data["image"])
        gt_np = data["gt_mask"]
        pred_np = data["pred_mask"]
        overlay = make_error_overlay(image_np, gt_np, pred_np)

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"IDRiD_{image_id:02d} ({label}) - Fundus")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"IDRiD_{image_id:02d} - Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"IDRiD_{image_id:02d} - Predicted")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title(f"Dice={dice:.3f}  (green=TP, red=FP, cyan=FN)")
        axes[row, 3].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this evaluation is meant for the GPU.")

    print("=" * 60)
    print("  Hard Exudate Segmentation - FINAL Test Set Evaluation")
    print("=" * 60)
    print("One-time evaluation on the held-out 27-image IDRiD test set.")
    print("No tuning will be done against this result.\n")

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    model, checkpoint = load_model(CHECKPOINT_PATH)
    print(f"  (from CV: epoch={checkpoint.get('epoch')}, "
          f"val_dice={checkpoint.get('val_dice'):.4f}, val_iou={checkpoint.get('val_iou'):.4f})")

    test_dataset = IDRiDExudateDataset(split="test")
    print(f"\nTest images: {len(test_dataset)}  (touched for the first time)")

    results, cache = run_inference(model, test_dataset)

    dices = np.array([r["dice"] for r in results])
    ious = np.array([r["iou"] for r in results])

    mean_dice, std_dice = float(dices.mean()), float(dices.std())
    mean_iou, std_iou = float(ious.mean()), float(ious.std())
    median_dice = float(np.median(dices))

    best = results[int(np.argmax(dices))]
    worst = results[int(np.argmin(dices))]

    print("\n" + "-" * 60)
    print("Per-Image Results (sorted by Dice, descending)")
    print("-" * 60)
    print(f"{'Image ID':<12}{'Dice':<10}{'IoU':<10}")
    for r in sorted(results, key=lambda r: r["dice"], reverse=True):
        print(f"IDRiD_{r['image_id']:02d}      {r['dice']:<10.4f}{r['iou']:<10.4f}")

    print("\n" + "-" * 60)
    print("Aggregate Test Set Metrics (n=27)")
    print("-" * 60)
    print(f"Mean Dice:   {mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"Mean IoU:    {mean_iou:.4f} +/- {std_iou:.4f}")
    print(f"Median Dice: {median_dice:.4f}")
    print(f"Best:  IDRiD_{best['image_id']:02d}  (dice={best['dice']:.4f}, iou={best['iou']:.4f})")
    print(f"Worst: IDRiD_{worst['image_id']:02d}  (dice={worst['dice']:.4f}, iou={worst['iou']:.4f})")

    # --- Save JSON ---
    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_cv_epoch": checkpoint.get("epoch"),
        "checkpoint_cv_val_dice": checkpoint.get("val_dice"),
        "checkpoint_cv_val_iou": checkpoint.get("val_iou"),
        "n_test_images": len(test_dataset),
        "per_image": results,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "median_dice": median_dice,
        "best": best,
        "worst": worst,
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {RESULTS_JSON_PATH}")

    # --- Pick 6 qualitative samples: 2 best, 2 median, 2 worst ---
    sorted_by_dice = sorted(results, key=lambda r: r["dice"], reverse=True)
    two_best = sorted_by_dice[:2]
    two_worst = sorted_by_dice[-2:]

    diffs = sorted(results, key=lambda r: abs(r["dice"] - median_dice))
    two_median = diffs[:2]

    selected = (
        [("best", r["image_id"], r["dice"]) for r in two_best] +
        [("median", r["image_id"], r["dice"]) for r in two_median] +
        [("worst", r["image_id"], r["dice"]) for r in two_worst]
    )

    plot_qualitative(selected, cache, QUALITATIVE_PNG_PATH)
    print(f"Saved {QUALITATIVE_PNG_PATH}")


if __name__ == "__main__":
    main()
