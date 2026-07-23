# segmentation/seg_dataset.py
#
# Phase 2 — lesion segmentation. Kept separate from the Phase 1 classifier
# (utils/, predict.py, train.py etc.) since it's a distinct capability with
# its own data format and model.
#
# Pairs IDRiD fundus images with their Hard Exudate (EX) ground-truth masks.

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Same ImageNet normalisation stats used by the Phase 1 classifier
# (utils/preprocess.py). Duplicated here (rather than imported) so this
# segmentation/ package stays self-contained and runnable as a plain script
# without needing the project root on sys.path.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 512

# Raw string + pathlib handles the spaces, parentheses, and D: drive in this
# path correctly — no special escaping needed at the Python string level.
DATA_ROOT = Path(
    r"D:\Trasferred download(2025-2026)\dataset_3_IDRiD (lesion segmentation)\A. Segmentation"
)

SPLIT_FOLDERS = {
    "train": "a. Training Set",
    "test":  "b. Testing Set",
}

IMAGE_FILENAME_RE = re.compile(r"^IDRiD_(\d+)\.jpg$")


class IDRiDExudateDataset(Dataset):
    """
    Pairs each IDRiD fundus image with its Hard Exudate segmentation mask.

    Image and mask are resized to IMAGE_SIZE x IMAGE_SIZE with different
    interpolation on purpose:
      - image: bilinear (smooth photographic content)
      - mask:  nearest-neighbor (preserves exact {0,1} values — bilinear or
               bicubic would blur the mask into non-binary intermediate values)
    """

    def __init__(self, split):
        if split not in SPLIT_FOLDERS:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.split = split
        split_folder = SPLIT_FOLDERS[split]

        self.image_dir = DATA_ROOT / "1. Original Images" / split_folder
        self.mask_dir = (
            DATA_ROOT / "2. All Segmentation Groundtruths" / split_folder / "3. Hard Exudates"
        )

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        ids = []
        for f in sorted(self.image_dir.iterdir()):
            m = IMAGE_FILENAME_RE.match(f.name)
            if m:
                ids.append(int(m.group(1)))
        ids.sort()

        if not ids:
            raise RuntimeError(f"No IDRiD_NN.jpg images found in {self.image_dir}")

        self.ids = ids

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ids)

    def _paths_for(self, image_id):
        image_path = self.image_dir / f"IDRiD_{image_id:02d}.jpg"
        mask_path = self.mask_dir / f"IDRiD_{image_id:02d}_EX.tif"
        return image_path, mask_path

    def image_id_at(self, index):
        """IDRiD_NN id for a given dataset index — used by seg_visualize.py to label samples."""
        return self.ids[index]

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path, mask_path = self._paths_for(image_id)

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"No Hard Exudate mask found for IDRiD_{image_id:02d} at {mask_path}"
            )

        # --- Image: RGB, bilinear resize, ImageNet-normalized tensor ---
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)  # [3, IMAGE_SIZE, IMAGE_SIZE]

        # --- Mask: palette-indexed TIFF with raw values {0,1}.
        # NEVER call .convert('L') here: this TIFF's palette maps index 1 to
        # RGB (255,0,0) (red), so .convert('L') would turn it into luminance
        # ~76 instead of a clean binary value. Resize with NEAREST so no new
        # intermediate values are introduced (never blur a mask).
        mask = Image.open(mask_path)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            # A few IDRiD mask files (e.g. IDRiD_81_EX.tif in the test set)
            # are stored as RGB/RGBA instead of palette-indexed. Same
            # red-foreground/black-background convention as the palette
            # files (index 0 -> RGB(0,0,0), index 1 -> RGB(255,0,0)) —
            # treat any nonzero RGB channel as foreground, ignoring alpha
            # (which is a constant 255 either way and carries no signal).
            mask_arr = mask_arr[..., :3].any(axis=-1)
        mask_bin = (mask_arr > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, IMAGE_SIZE, IMAGE_SIZE]

        return image_tensor, mask_tensor
