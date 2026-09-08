"""
Unit tests for SpaceNetra LEVIR-CD Dataset, DataModule, and Augmentations.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.dataset import LEVIRCDDataset
from src.data.datamodule import LEVIRCDDataModule
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.config import config


def test_train_dataset_len():
    dataset = LEVIRCDDataset(split="train", patch_size=256)
    assert len(dataset) > 0, "Train dataset should contain at least 1 sample"


def test_val_dataset_len():
    dataset = LEVIRCDDataset(split="val", patch_size=256)
    # 10 images * 16 patches per image = 160 for synthetic dataset
    assert len(dataset) > 0, "Val dataset should contain at least 1 grid patch"
    assert len(dataset) % 16 == 0, "Val dataset length should be a multiple of 16 patches"


def test_item_tensor_shapes():
    dataset = LEVIRCDDataset(split="train", patch_size=256)
    sample = dataset[0]

    assert "img_a" in sample and "img_b" in sample and "mask" in sample and "filename" in sample
    assert sample["img_a"].shape == (3, 256, 256), f"Expected (3, 256, 256), got {sample['img_a'].shape}"
    assert sample["img_b"].shape == (3, 256, 256), f"Expected (3, 256, 256), got {sample['img_b'].shape}"
    assert sample["mask"].shape == (1, 256, 256), f"Expected (1, 256, 256), got {sample['mask'].shape}"
    assert isinstance(sample["filename"], str)


def test_mask_value_range():
    dataset = LEVIRCDDataset(split="train", patch_size=256)
    sample = dataset[0]
    mask = sample["mask"]

    unique_vals = torch.unique(mask)
    for val in unique_vals:
        assert float(val) in (0.0, 1.0), f"Invalid mask value found: {val}"


def test_image_value_range():
    dataset = LEVIRCDDataset(split="train", patch_size=256)
    sample = dataset[0]

    img_a = sample["img_a"]
    img_b = sample["img_b"]

    assert img_a.min() >= 0.0 and img_a.max() <= 1.0, f"img_a out of range: [{img_a.min()}, {img_a.max()}]"
    assert img_b.min() >= 0.0 and img_b.max() <= 1.0, f"img_b out of range: [{img_b.min()}, {img_b.max()}]"


def test_datamodule_train_batch():
    dm = LEVIRCDDataModule(batch_size=4, num_workers=0)
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch["img_a"].shape == (4, 3, 256, 256)
    assert batch["img_b"].shape == (4, 3, 256, 256)
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert len(batch["filename"]) == 4


def test_augmentation_sync():
    transform = get_train_transforms(patch_size=256)
    dummy_img1 = np.ones((512, 512, 3), dtype=np.uint8) * 100
    dummy_img2 = np.ones((512, 512, 3), dtype=np.uint8) * 200
    dummy_mask = np.ones((512, 512), dtype=np.uint8) * 255

    res = transform(image=dummy_img1, image2=dummy_img2, mask=dummy_mask)

    assert res["image"].shape == (256, 256, 3)
    assert res["image2"].shape == (256, 256, 3)
    assert res["mask"].shape == (256, 256)
    assert np.all(res["image"] == 100)
    assert np.all(res["image2"] == 200)
    assert np.all(res["mask"] == 255)
