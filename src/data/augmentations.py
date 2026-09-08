"""
Synchronized data augmentation pipelines for SpaceNetra bi-temporal satellite image pairs.
Provides spatial transforms (RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90)
applied identically across T1, T2, and Mask using deterministic random state seeds.
"""

from typing import Dict, Tuple, Optional
import random
import numpy as np


class SynchronizedTransforms:
    """
    Applies synchronized spatial augmentations to bi-temporal image pairs (T1, T2) and ground truth masks.
    Guarantees that T1, T2, and Mask undergo identical spatial crops, flips, and rotations.
    """

    def __init__(
        self,
        patch_size: int = 256,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        is_train: bool = True,
    ):
        self.patch_size = patch_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.is_train = is_train

    def __call__(
        self, image: np.ndarray, image2: np.ndarray, mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            image: T1 image array (H, W, C)
            image2: T2 image array (H, W, C)
            mask: Label mask array (H, W)

        Returns:
            Dict containing transformed 'image', 'image2', and 'mask'
        """
        h, w = image.shape[:2]

        if self.is_train:
            # 1. Random Crop to patch_size x patch_size
            if h > self.patch_size or w > self.patch_size:
                max_y = max(0, h - self.patch_size)
                max_x = max(0, w - self.patch_size)
                y1 = random.randint(0, max_y)
                x1 = random.randint(0, max_x)
                y2 = y1 + self.patch_size
                x2 = x1 + self.patch_size

                image = image[y1:y2, x1:x2]
                image2 = image2[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]

            # 2. Horizontal Flip
            if random.random() < self.hflip_prob:
                image = np.fliplr(image)
                image2 = np.fliplr(image2)
                mask = np.fliplr(mask)

            # 3. Vertical Flip
            if random.random() < self.vflip_prob:
                image = np.flipud(image)
                image2 = np.flipud(image2)
                mask = np.flipud(mask)

            # 4. Random 90-degree rotation (1, 2, or 3 times)
            if random.random() < self.rotate90_prob:
                k = random.choice([1, 2, 3])
                image = np.rot90(image, k=k)
                image2 = np.rot90(image2, k=k)
                mask = np.rot90(mask, k=k)

        # Ensure contiguous arrays for PyTorch conversion
        return {
            "image": np.ascontiguousarray(image),
            "image2": np.ascontiguousarray(image2),
            "mask": np.ascontiguousarray(mask),
        }


def get_train_transforms(patch_size: int = 256) -> SynchronizedTransforms:
    """Returns training transforms."""
    return SynchronizedTransforms(
        patch_size=patch_size,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotate90_prob=0.5,
        is_train=True,
    )


def get_val_transforms(patch_size: int = 256) -> SynchronizedTransforms:
    """Returns validation/test transforms."""
    return SynchronizedTransforms(
        patch_size=patch_size,
        is_train=False,
    )
