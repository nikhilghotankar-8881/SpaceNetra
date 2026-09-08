"""
LEVIRCDDataModule for managing train, val, and test DataLoaders for SpaceNetra.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.data.dataset import LEVIRCDDataset


class LEVIRCDDataModule:
    """
    DataModule wrapper managing PyTorch DataLoaders across splits.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
        patch_size: int = 256,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        """
        Args:
            data_dir: Directory containing levir_cd dataset splits (defaults to config)
            batch_size: Batch size per DataLoader (defaults to config.change_detection.training['batch_size'])
            patch_size: Patch size for cropping (defaults to 256)
            num_workers: Number of workers for data loading (defaults to config.system.num_workers)
            pin_memory: Whether to pin memory in DataLoaders (defaults to config.system.pin_memory)
        """
        self.data_dir = Path(data_dir) if data_dir is not None else config.paths.levir_cd_dir
        self.batch_size = batch_size if batch_size is not None else config.change_detection.training.get("batch_size", 8)
        self.patch_size = patch_size
        self.num_workers = num_workers if num_workers is not None else config.system.num_workers
        self.pin_memory = pin_memory if pin_memory is not None else config.system.pin_memory

        self.train_dataset: Optional[LEVIRCDDataset] = None
        self.val_dataset: Optional[LEVIRCDDataset] = None
        self.test_dataset: Optional[LEVIRCDDataset] = None

    def setup(self):
        """Initializes train, val, and test dataset instances."""
        self.train_dataset = LEVIRCDDataset(
            root=self.data_dir,
            split="train",
            patch_size=self.patch_size,
        )
        self.val_dataset = LEVIRCDDataset(
            root=self.data_dir,
            split="val",
            patch_size=self.patch_size,
        )
        self.test_dataset = LEVIRCDDataset(
            root=self.data_dir,
            split="test",
            patch_size=self.patch_size,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
