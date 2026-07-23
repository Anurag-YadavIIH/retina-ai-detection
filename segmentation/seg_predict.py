# segmentation/seg_predict.py
#
# Inference-only wrapper around the Hard Exudate U-Net, for use by the Flask
# app (app.py). Mirrors predict.py's module-level singleton pattern for the
# classifier: load once, cache, never reload per request.
#
# This entire feature is best-effort and MUST NOT break the existing
# classifier + Grad-CAM app if segmentation is disabled, the checkpoint is
# missing, or anything here raises. Every public function returns None
# instead of raising when segmentation isn't available.

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    # Normal case: imported as a package from app.py at the project root
    # (`from segmentation.seg_predict import segment_exudates`).
    from segmentation.seg_model import build_unet
except ImportError:
    # Fallback: running this file directly as a script from inside
    # segmentation/ (sys.path[0] is the segmentation/ folder itself, so the
    # "segmentation" package name isn't visible — import the sibling module
    # by its bare name instead).
    from seg_model import build_unet

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IMAGE_SIZE = 512

# Same ImageNet stats used throughout the segmentation pipeline
# (seg_dataset.py) — duplicated here rather than imported, so this module
# doesn't depend on import context (see the try/except above).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "exudate_unet_fold4.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Semi-transparent orange (splits the difference between "red" and
# "yellow") for the lesion overlay — visually distinct from the Grad-CAM's
# blue-green-yellow-red JET colormap.
OVERLAY_COLOR = np.array([1.0, 0.647, 0.0])
OVERLAY_ALPHA = 0.45

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# MODEL LOADER  (singleton pattern, same as predict.py)
# ─────────────────────────────────────────────

_seg_model = None
_seg_load_attempted = False


def load_seg_model():
    """
    Load the Hard Exudate U-Net once and cache it. Returns None (never
    raises) if:
      - ENABLE_SEGMENTATION is set to anything other than "true"
      - the checkpoint file is missing
      - loading the checkpoint fails for any reason
    A None result is also cached, so a disabled/missing/broken model is
    only checked once per process, not on every request.
    """
    global _seg_model, _seg_load_attempted

    if _seg_load_attempted:
        return _seg_model

    _seg_load_attempted = True

    enabled = os.environ.get("ENABLE_SEGMENTATION", "true").strip().lower() == "true"
    if not enabled:
        print("Segmentation disabled (ENABLE_SEGMENTATION != true) — classifier-only mode.")
        return None

    if not CHECKPOINT_PATH.is_file():
        print(f"Segmentation checkpoint not found at {CHECKPOINT_PATH} — classifier-only mode.")
        return None

    try:
        model = build_unet().to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        _seg_model = model
        print(f"Segmentation model loaded from {CHECKPOINT_PATH} (device={DEVICE})")
    except Exception as e:
        print(f"Failed to load segmentation model ({e}) — classifier-only mode.")
        _seg_model = None

    return _seg_model


# ─────────────────────────────────────────────
# INFERENCE + OVERLAY
# ─────────────────────────────────────────────

def segment_exudates(image_path, output_path):
    """
    Run Hard Exudate segmentation on one image and write an overlay to
    output_path.

    Returns {"overlay_path": output_path, "lesion_area_pct": float} on
    success, or None if segmentation is disabled/unavailable, or if
    anything goes wrong — this function never raises, since a broken
    segmentation step must not break the classifier response.
    """
    model = load_seg_model()
    if model is None:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size   # (W, H) — overlay is written at original resolution

        input_tensor = _IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask_small = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()  # [512,512]

        lesion_area_pct = float(pred_mask_small.mean() * 100)

        # Resize the binary mask back up to the original resolution with
        # NEAREST — never blur a mask, same rule as the training pipeline.
        mask_img = Image.fromarray((pred_mask_small * 255).astype(np.uint8))
        mask_resized = mask_img.resize(original_size, resample=Image.Resampling.NEAREST)
        mask_full = np.array(mask_resized) > 0

        # Semi-transparent overlay on the original (not resized) fundus image
        original_np = np.array(image).astype(np.float32) / 255.0
        mask_3d = mask_full[:, :, None].astype(np.float32)
        overlay = original_np * (1 - mask_3d * OVERLAY_ALPHA) + OVERLAY_COLOR * (mask_3d * OVERLAY_ALPHA)

        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        overlay_img.save(output_path, quality=90)

        return {
            "overlay_path": output_path,
            "lesion_area_pct": round(lesion_area_pct, 2),
        }

    except Exception as e:
        print(f"Segmentation inference failed for {image_path} ({e}) — skipping.")
        return None
