# utils/preprocess.py
#
# This file handles all image transformations.
# PyTorch uses "transforms" — a pipeline of operations applied to each image.

import torch
from torchvision import transforms

# -------------------------------------------------------------------
# CLASS NAMES — must match your dataset folder names exactly.
# torchvision.datasets.ImageFolder assigns integer labels by sorting
# the folder names alphabetically, so this list (and its order) MUST
# mirror that alphabetical sort of the real "dataset/" subfolders:
#   Mild, Moderate, No_DR, Proliferate_DR, Severe
# This is also the order the currently-trained model/retina_model.pth
# was trained with — do not reorder this list without retraining.
# -------------------------------------------------------------------
CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Map folder name → integer index (PyTorch needs numbers, not strings)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# Result: {'Mild': 0, 'Moderate': 1, 'No_DR': 2, 'Proliferate_DR': 3, 'Severe': 4}

# -------------------------------------------------------------------
# HUMAN-READABLE DISPLAY NAMES
# Keyed by the real folder/CLASS_NAMES values above — used by the web
# app and predict.py so the UI never shows raw folder-name strings.
# -------------------------------------------------------------------
CLASS_LABELS = {
    'No_DR':          'Healthy (No DR)',
    'Mild':           'Mild DR',
    'Moderate':       'Moderate DR',
    'Severe':         'Severe DR',
    'Proliferate_DR': 'Proliferative DR',
}

# One clinical sentence per class, shown in the results card.
CLASS_DESCRIPTIONS = {
    'No_DR':          'No signs of diabetic retinopathy — the retina appears healthy.',
    'Mild':           'Mild nonproliferative diabetic retinopathy: microaneurysms present; routine monitoring advised.',
    'Moderate':       'Moderate nonproliferative diabetic retinopathy: more extensive retinal damage; closer follow-up and possible treatment recommended.',
    'Severe':         'Severe nonproliferative diabetic retinopathy: extensive hemorrhages and vascular abnormalities; urgent ophthalmology referral required.',
    'Proliferate_DR': 'Proliferative diabetic retinopathy: abnormal new blood vessel growth; immediate treatment critical to prevent vision loss.',
}

# Severity colour for the UI badge — hex values, not CSS colour names.
CLASS_COLORS = {
    'No_DR':          '#22c55e',
    'Mild':           '#84cc16',
    'Moderate':       '#f97316',
    'Severe':         '#ef4444',
    'Proliferate_DR': '#dc2626',
}

# -------------------------------------------------------------------
# IMAGE SIZE
# ResNet18 was designed for 224x224 images.
# This is the standard — do not change it.
# -------------------------------------------------------------------
IMAGE_SIZE = 224

# -------------------------------------------------------------------
# NORMALISATION CONSTANTS
# These exact numbers are the mean and std of the ImageNet dataset.
# ResNet18 was pre-trained on ImageNet, so we must use the same
# normalisation it expects. This is a standard constant — memorise it.
# -------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Transforms applied to TRAINING images only.
    We add augmentation (random flips, rotations) to make the model
    more robust — it learns to recognise DR regardless of image angle.
    """
    return transforms.Compose([
        # Step 1: Resize to slightly larger than needed, then random crop
        # This gives spatial variety — the model sees different crops each epoch
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),

        # Step 2: Augmentations — simulate real-world variation
        transforms.RandomHorizontalFlip(p=0.5),   # 50% chance of left-right flip
        transforms.RandomVerticalFlip(p=0.5),     # retinal images look similar flipped
        transforms.RandomRotation(degrees=15),    # rotate up to 15 degrees
        transforms.ColorJitter(                   # slight colour variation
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),

        # Step 3: Convert PIL Image → PyTorch Tensor (values become 0.0–1.0)
        transforms.ToTensor(),

        # Step 4: Normalise using ImageNet stats
        # Formula: (pixel - mean) / std   →   values centred around 0
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Transforms applied to VALIDATION and TEST images.
    No augmentation here — we want a fair, consistent evaluation.
    Just resize, centre crop, and normalise.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),   # always the centre, not random
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms():
    """
    Transforms for a SINGLE image uploaded through the web app.
    Same as validation — no augmentation, just clean preprocessing.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])