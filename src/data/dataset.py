"""
LEVIR-CD PyTorch Dataset implementation for SpaceNetra.
Supports random patch cropping during training and deterministic 4x4 grid patching during val/test.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import config
from src.data.augmentations import get_train_transforms, get_val_transforms


class LEVIRCDDataset(Dataset):
    """
    PyTorch Dataset for LEVIR-CD bi-temporal change detection pairs.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        patch_size: int = 256,
        transform=None,
    ):
        """
        Args:
            root: Root path to LEVIR-CD directory (defaults to config.paths.levir_cd_dir)
            split: 'train', 'val', or 'test'
            patch_size: Desired patch crop size (default 256)
            transform: Custom Albumentations transform pipeline (optional)
        """
        super().__init__()
        self.root = Path(root) if root is not None else config.paths.levir_cd_dir
        self.split = split.lower()
        self.patch_size = patch_size

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

        self.split_dir = self.root / self.split
        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        self.file_list = sorted([p.name for p in self.a_dir.glob("*.png")])
        if len(self.file_list) == 0:
            raise RuntimeError(f"No PNG files found in {self.a_dir}")

        # Default transforms if not provided
        if transform is None:
            if self.split == "train":
                self.transform = get_train_transforms(self.patch_size)
            else:
                self.transform = get_val_transforms()
        else:
            self.transform = transform

        # For val/test, build a deterministic grid index of 256x256 patches from 1024x1024 images
        self.grid_patches: List[Tuple[int, int, int]] = []
        if self.split in ("val", "test"):
            # Assuming 1024x1024 images -> 4x4 grid of 256x256 patches = 16 patches per image
            steps = 1024 // self.patch_size  # 4
            for img_idx in range(len(self.file_list)):
                for r in range(steps):
                    for c in range(steps):
                        y1 = r * self.patch_size
                        x1 = c * self.patch_size
                        self.grid_patches.append((img_idx, y1, x1))

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.file_list)
        else:
            return len(self.grid_patches)

    def _load_triplet(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads T1 (Image A), T2 (Image B), and Label Mask as numpy arrays."""
        a_path = self.a_dir / filename
        b_path = self.b_dir / filename
        label_path = self.label_dir / filename

        img_a = np.array(Image.open(a_path).convert("RGB"))
        img_b = np.array(Image.open(b_path).convert("RGB"))
        mask = np.array(Image.open(label_path).convert("L"))

        return img_a, img_b, mask

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if self.split == "train":
            filename = self.file_list[idx]
            img_a, img_b, mask = self._load_triplet(filename)

            # Apply synchronized Albumentations transform (RandomCrop + Augmentations)
            if self.transform is not None:
                augmented = self.transform(image=img_a, image2=img_b, mask=mask)
                img_a = augmented["image"]
                img_b = augmented["image2"]
                mask = augmented["mask"]
        else:
            img_idx, y1, x1 = self.grid_patches[idx]
            filename = self.file_list[img_idx]
            img_a_full, img_b_full, mask_full = self._load_triplet(filename)

            # Crop deterministic patch
            y2 = y1 + self.patch_size
            x2 = x1 + self.patch_size
            img_a = img_a_full[y1:y2, x1:x2]
            img_b = img_b_full[y1:y2, x1:x2]
            mask = mask_full[y1:y2, x1:x2]

            if self.transform is not None:
                augmented = self.transform(image=img_a, image2=img_b, mask=mask)
                img_a = augmented["image"]
                img_b = augmented["image2"]
                mask = augmented["mask"]

        # Convert images to FloatTensor (3, H, W) scaled to [0.0, 1.0]
        tensor_a = torch.from_numpy(img_a.transpose(2, 0, 1)).float() / 255.0
        tensor_b = torch.from_numpy(img_b.transpose(2, 0, 1)).float() / 255.0

        # Convert binary mask to FloatTensor (1, H, W) scaled to {0.0, 1.0}
        mask_binary = (mask > 0).astype(np.float32)
        tensor_mask = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return {
            "img_a": tensor_a,
            "img_b": tensor_b,
            "mask": tensor_mask,
            "filename": filename,
        }
