# predict.py
#
# Handles all inference logic:
#   - Load the saved model
#   - Preprocess a single uploaded image
#   - Run prediction + confidence scores
#   - Generate Grad-CAM heatmap
#
# Used by app.py (Flask backend)

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from utils.preprocess import (
    get_inference_transforms,
    CLASS_NAMES,
    CLASS_LABELS,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH  = "model/retina_model.pth"
NUM_CLASSES = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL LOADER  (singleton pattern)
# Load the model once when the server starts —
# not on every request (that would be very slow).
# ─────────────────────────────────────────────

_model = None   # module-level cache

def load_model():
    """
    Load the trained ResNet18 model from disk.
    Uses a module-level cache so it's only loaded once.
    """
    global _model

    if _model is not None:
        return _model   # already loaded — return cached version

    # Rebuild the exact same architecture used in train.py
    model = models.resnet18(weights=None)   # no pretrained weights — we load our own
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Load the saved weights. Support both formats:
    #   - a checkpoint dict with a "model_state_dict" key (train.py's format)
    #   - a raw state_dict saved directly via torch.save(model.state_dict(), ...)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()   # IMPORTANT: always set eval mode for inference

    _model = model
    print(f"Model loaded from {MODEL_PATH}")
    return _model


# ─────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: visualise which image regions the model focuses on.

    How it works:
    1. Register a "hook" on the last conv layer to capture its output (feature maps)
    2. Run a forward pass → get prediction
    3. Run a backward pass for the predicted class → get gradients at that layer
    4. Weight feature maps by their mean gradient → produce heatmap
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None   # will store gradients from backward pass
        self.activations  = None   # will store feature maps from forward pass

        # Register hooks — these are callbacks that fire during forward/backward
        self.forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Called automatically during forward pass — saves feature maps."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Called automatically during backward pass — saves gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a Grad-CAM heatmap for the given input image tensor.

        Args:
            input_tensor: preprocessed image tensor [1, 3, 224, 224]
            class_idx:    which class to visualise (None = use predicted class)

        Returns:
            cam: numpy heatmap array, same size as input image (224x224)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)   # shape: [1, num_classes]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()   # use predicted class

        # Backward pass — compute gradients only for the predicted class
        self.model.zero_grad()
        output[0, class_idx].backward()   # scalar output → single backward pass

        # Pool gradients across spatial dimensions (H, W) → one weight per channel
        # Shape: [1, C, H, W] → mean over H,W → [1, C, 1, 1]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weight each feature map (channel) by its pooled gradient
        # activations shape: [1, C, H, W]
        cam = (weights * self.activations).sum(dim=1, keepdim=False)   # [1, H, W]
        cam = cam.squeeze()                                             # [H, W]

        # ReLU: only keep positive contributions (what activated the class)
        cam = torch.relu(cam).numpy()

        # Normalise to 0-1 range
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        """Clean up hooks when done — prevents memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


# ─────────────────────────────────────────────
# HEATMAP OVERLAY GENERATOR
# ─────────────────────────────────────────────

def apply_heatmap_overlay(original_image_path, cam, output_path):
    """
    Overlay the Grad-CAM heatmap on the original retinal image.

    Args:
        original_image_path: path to the uploaded image file
        cam:                 numpy array heatmap (H, W), values 0-1
        output_path:         where to save the overlaid image

    Returns:
        output_path: path to the saved heatmap image
    """
    # Read original image with OpenCV (returns BGR format)
    original = cv2.imread(original_image_path)
    original = cv2.resize(original, (224, 224))

    # Resize CAM to match image size (CAM from last conv layer is 7x7)
    cam_resized = cv2.resize(cam, (224, 224))

    # Convert CAM to a colourmap (COLORMAP_JET: blue→green→yellow→red)
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    # Blend heatmap with original image (40% heatmap, 60% original)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Save the result
    cv2.imwrite(output_path, overlay)

    return output_path


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# Called by Flask for every uploaded image
# ─────────────────────────────────────────────

def predict(image_path):
    """
    Full inference pipeline for one uploaded image.

    Args:
        image_path: path to the uploaded retinal image

    Returns:
        dict with keys:
            predicted_class  - raw class name, e.g. "Moderate"
            label            - human-readable name, e.g. "Moderate DR"
            confidence       - e.g. 87.3  (percentage)
            all_scores       - dict of class→probability for all 5 classes
            description      - human-readable severity description
            color            - hex color for the UI badge, e.g. "#f97316"
            heatmap_path     - path to the Grad-CAM overlay image
    """
    model = load_model()

    # ── 1. Load and preprocess the image ──────────────
    image = Image.open(image_path).convert("RGB")  # ensure 3-channel RGB

    transform    = get_inference_transforms()
    input_tensor = transform(image)          # shape: [3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0) # add batch dim → [1, 3, 224, 224]
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad_(True)        # needed for Grad-CAM backward pass

    # ── 2. Run prediction ─────────────────────────────
    with torch.set_grad_enabled(True):       # enable gradients for Grad-CAM

        # Set up Grad-CAM on the last residual block's conv layer
        # layer4 is the deepest feature extractor in ResNet18
        grad_cam = GradCAM(model, model.layer4[-1].conv2)

        # Forward pass
        output = model(input_tensor)         # shape: [1, 5]

        # Convert raw scores (logits) to probabilities using softmax
        probabilities = torch.softmax(output, dim=1)  # values sum to 1.0

        # Get the winning class and its confidence
        confidence, pred_idx = probabilities.max(dim=1)
        pred_idx   = pred_idx.item()         # tensor → Python int
        confidence = confidence.item() * 100  # 0-1 → percentage

        predicted_class = CLASS_NAMES[pred_idx]

        # ── 3. Generate Grad-CAM heatmap ──────────────
        cam = grad_cam.generate(input_tensor, class_idx=pred_idx)
        grad_cam.remove_hooks()

    # ── 4. Build heatmap overlay image ────────────────
    # Save heatmap next to original image with "_heatmap" suffix
    base_name    = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(
        "static", "uploads", f"{base_name}_heatmap.jpg"
    )
    apply_heatmap_overlay(image_path, cam, heatmap_path)

    # ── 5. Build all-class probability scores ─────────
    probs_np   = probabilities.squeeze().detach().numpy()
    all_scores = {
        CLASS_NAMES[i]: round(float(probs_np[i]) * 100, 1)
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_class": predicted_class,
        "label":           CLASS_LABELS[predicted_class],
        "confidence":      round(confidence, 1),
        "all_scores":      all_scores,
        "description":     CLASS_DESCRIPTIONS[predicted_class],
        "color":           CLASS_COLORS[predicted_class],
        "heatmap_path":    heatmap_path,
    }