"""
CLI entry point for evaluating trained SpaceNetra models on test sets.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/baseline.yaml
    python evaluate.py --dry-run
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.siamese_unet import SiameseUNet
from src.evaluation.evaluator import ModelEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceNetra Model Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/metrics",
        help="Target directory for evaluation reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run evaluation on synthetic data to verify pipeline without dataset dependency",
    )
    return parser


def create_synthetic_test_loader(batch_size: int = 4) -> DataLoader:
    """Generates synthetic PyTorch DataLoader for evaluation dry-run."""
    t1 = torch.randn(8, 3, 256, 256)
    t2 = torch.randn(8, 3, 256, 256)
    target = torch.randint(0, 2, (8, 1, 256, 256)).float()
    dataset = TensorDataset(t1, t2, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {"backbone": "resnet18", "pretrained": False}, "data": {"batch_size": 8}}

    backbone = config.get("model", {}).get("backbone", "resnet18")
    model = SiameseUNet(backbone=backbone, pretrained=False)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint not found at '{args.checkpoint}'. Using un-trained initialized weights.")

    evaluator = ModelEvaluator(model=model)

    if args.dry_run:
        print("Running DRY-RUN evaluation on synthetic test set...")
        test_loader = create_synthetic_test_loader(batch_size=args.batch_size if hasattr(args, "batch_size") and args.batch_size else 4)
    else:
        from src.data.datamodule import LEVIRCDDataModule

        data_dir = config.get("data", {}).get("data_dir", "data/levir_cd")
        batch_size = config.get("data", {}).get("batch_size", 8)

        datamodule = LEVIRCDDataModule(data_dir=data_dir, batch_size=batch_size)
        datamodule.setup()
        test_loader = datamodule.test_dataloader()

    print("Executing model evaluation...")
    metrics = evaluator.evaluate(test_loader)

    txt_file, json_file = evaluator.generate_report(
        metrics=metrics,
        output_dir=args.output_dir,
        checkpoint_name=ckpt_path.name if ckpt_path.exists() else "uninitialized_model",
    )

    print(f"\nEvaluation Complete! Reports saved to:")
    print(f"  - Text Report: {txt_file}")
    print(f"  - JSON Data:   {json_file}\n")

    with open(txt_file, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
