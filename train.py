"""
Command-Line Interface (CLI) entry point for training SpaceNetra models.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/baseline.yaml --epochs 1 --dry-run
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.siamese_unet import SiameseUNet
from src.models.changeformer import ChangeFormer
from src.models.losses import BCEDiceLoss
from src.training.trainer import ChangeDetectionTrainer



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceNetra Change Detection Model Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to training YAML configuration file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1 epoch on synthetic data to verify pipeline without dataset dependency",
    )
    return parser


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_synthetic_dataloaders(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Generates synthetic PyTorch dataloaders for dry-run verification."""
    t1 = torch.randn(8, 3, 256, 256)
    t2 = torch.randn(8, 3, 256, 256)
    target = torch.randint(0, 2, (8, 1, 256, 256)).float()

    dataset = TensorDataset(t1, t2, target)
    # PyTorch TensorDataset unpacks as tuple (t1, t2, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, loader


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded training config from: {args.config}")

    epochs = args.epochs if args.epochs is not None else config["training"].get("epochs", 50)
    batch_size = args.batch_size if args.batch_size is not None else config["data"].get("batch_size", 8)

    checkpoint_dir = config.get("checkpoint", {}).get("save_dir", "checkpoints")
    arch = config["model"].get("architecture", "siamese_unet").lower()
    if arch == "changeformer":
        embed_dims = config["model"].get("embed_dims", [64, 128, 320, 512])
        model = ChangeFormer(embed_dims=embed_dims)
    else:
        backbone = config["model"].get("backbone", "resnet18")
        pretrained = config["model"].get("pretrained", True)
        model = SiameseUNet(backbone=backbone, pretrained=pretrained)


    criterion = BCEDiceLoss(
        bce_weight=config["loss"].get("bce_weight", 1.0),
        dice_weight=config["loss"].get("dice_weight", 1.0),
        smooth=config["loss"].get("smooth", 1.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(config["scheduler"].get("eta_min", 1e-6)),
    )

    trainer = ChangeDetectionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["training"].get("device", "auto"),
        checkpoint_dir=checkpoint_dir,
    )

    if args.dry_run:
        print(f"Running DRY-RUN mode for {epochs} epoch(s)...")

        train_loader, val_loader = create_synthetic_dataloaders(batch_size=batch_size)
        metrics = trainer.fit(train_loader, val_loader, epochs=epochs)
        print("Dry-run completed successfully! Final metrics:", metrics)
    else:
        print(f"Starting model training for {epochs} epochs...")
        from src.data.datamodule import LEVIRCDDataModule

        data_dir = config["data"].get("data_dir", "data/levir_cd")
        datamodule = LEVIRCDDataModule(data_dir=data_dir, batch_size=batch_size)
        datamodule.setup()

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        metrics = trainer.fit(train_loader, val_loader, epochs=epochs)
        print("Training completed! Final metrics:", metrics)


if __name__ == "__main__":
    main()
