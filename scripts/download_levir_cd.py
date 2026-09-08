"""
LEVIR-CD Dataset Download, Synthetic Generation, and Organization Script

Usage:
    python scripts/download_levir_cd.py --synthetic
    python scripts/download_levir_cd.py --download
    python scripts/download_levir_cd.py --organize path/to/levir_cd.zip
    python scripts/download_levir_cd.py --verify
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config import config

# Standard LEVIR-CD split counts:
EXPECTED_COUNTS = {
    "train": 445,
    "val": 64,
    "test": 128,
}

# Known Google Drive File IDs or URLs
GDRIVE_FILE_ID = "1g6K0N_3jYcWc7jO46HwI5aM_1ZpE0p8_"


def create_synthetic_pair(
    idx: int, split: str, size: Tuple[int, int] = (1024, 1024)
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Generate a realistic synthetic satellite image pair (T1, T2) and binary change mask."""
    width, height = size
    rng = np.random.default_rng(seed=idx + (100 if split == "val" else 200 if split == "test" else 0))

    # Base background (terrain/grass/soil)
    bg_t1 = rng.integers(60, 140, size=(height, width, 3), dtype=np.uint8)
    bg_t2 = bg_t1.copy()

    # Add slight seasonal / illumination shift between T1 and T2
    t2_shift = rng.integers(-15, 15, size=(1, 1, 3), dtype=np.int16)
    bg_t2 = np.clip(bg_t2.astype(np.int16) + t2_shift, 0, 255).astype(np.uint8)

    img_t1 = Image.fromarray(bg_t1, mode="RGB")
    img_t2 = Image.fromarray(bg_t2, mode="RGB")
    mask = Image.new("L", (width, height), 0)

    draw_t1 = ImageDraw.Draw(img_t1)
    draw_t2 = ImageDraw.Draw(img_t2)
    draw_mask = ImageDraw.Draw(mask)

    # Draw unchanged buildings (present in both T1 and T2)
    num_unchanged = rng.integers(15, 30)
    for _ in range(num_unchanged):
        x1 = rng.integers(50, width - 150)
        y1 = rng.integers(50, height - 150)
        w = rng.integers(30, 80)
        h = rng.integers(30, 80)
        color = tuple(rng.integers(180, 240, size=3).tolist())
        draw_t1.rectangle([x1, y1, x1 + w, y1 + h], fill=color, outline=(40, 40, 40))
        draw_t2.rectangle([x1, y1, x1 + w, y1 + h], fill=color, outline=(40, 40, 40))

    # Draw new buildings (added in T2 -> change mask = 255)
    num_changed = rng.integers(3, 8)
    for _ in range(num_changed):
        x1 = rng.integers(50, width - 150)
        y1 = rng.integers(50, height - 150)
        w = rng.integers(40, 100)
        h = rng.integers(40, 100)
        color = tuple(rng.integers(200, 255, size=3).tolist())

        # Building only in T2
        draw_t2.rectangle([x1, y1, x1 + w, y1 + h], fill=color, outline=(20, 20, 20))
        # Mask has 255 for change area
        draw_mask.rectangle([x1, y1, x1 + w, y1 + h], fill=255)

    return img_t1, img_t2, mask


def generate_synthetic_dataset(output_dir: Path, samples_per_split: Dict[str, int] = None) -> None:
    """Generate a synthetic mini-dataset for development and testing."""
    if samples_per_split is None:
        samples_per_split = {"train": 10, "val": 10, "test": 10}

    print(f"[*] Generating synthetic LEVIR-CD dataset at: {output_dir}")

    for split, count in samples_per_split.items():
        split_dir = output_dir / split
        dir_a = split_dir / "A"
        dir_b = split_dir / "B"
        dir_mask = split_dir / "label"

        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)
        dir_mask.mkdir(parents=True, exist_ok=True)

        print(f"    Generating {count} synthetic pairs for '{split}' split...")
        for i in tqdm(range(1, count + 1)):
            filename = f"{split}_{i:03d}.png"
            t1, t2, mask = create_synthetic_pair(idx=i, split=split)

            t1.save(dir_a / filename)
            t2.save(dir_b / filename)
            mask.save(dir_mask / filename)

    print("[+] Synthetic dataset generation complete!")


def validate_structure(dataset_dir: Path) -> bool:
    """Inspect dataset directory, print stats, and check validity."""
    print(f"\n[*] Validating dataset structure at: {dataset_dir}")
    if not dataset_dir.exists():
        print(f"[-] Directory does not exist: {dataset_dir}")
        return False

    all_valid = True
    total_triplets = 0

    print("-" * 65)
    print(f"{'Split':<10} | {'T1 (A)':<10} | {'T2 (B)':<10} | {'Label':<10} | {'Status':<15}")
    print("-" * 65)

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        dir_a = split_dir / "A"
        dir_b = split_dir / "B"
        dir_mask = split_dir / "label"

        count_a = len(list(dir_a.glob("*.png"))) if dir_a.exists() else 0
        count_b = len(list(dir_b.glob("*.png"))) if dir_b.exists() else 0
        count_mask = len(list(dir_mask.glob("*.png"))) if dir_mask.exists() else 0

        is_matching = (count_a == count_b == count_mask) and count_a > 0
        status = "[OK]" if is_matching else "[WARN] Mismatch/Empty"
        if not is_matching:
            all_valid = False

        print(f"{split:<10} | {count_a:<10} | {count_b:<10} | {count_mask:<10} | {status:<15}")
        total_triplets += count_a

    print("-" * 65)
    print(f"Total Image Triplets: {total_triplets}")

    if total_triplets > 0 and all_valid:
        # Check first sample dimensions
        sample_path = next((dataset_dir / "train" / "A").glob("*.png"), None)
        if sample_path:
            img = Image.open(sample_path)
            print(f"Sample Image Dimensions: {img.size[0]}x{img.size[1]} RGB ({img.mode})")
        print("[+] Dataset validation SUCCESSFUL!")
    elif total_triplets == 0:
        print("[i] Dataset is empty. Run with --synthetic or --download.")

    return all_valid


def download_dataset(target_dir: Path) -> None:
    """Download LEVIR-CD using gdown or requests fallback."""
    raw_zip = target_dir.parent / "raw" / "LEVIR-CD.zip"
    raw_zip.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading LEVIR-CD dataset...")
    try:
        import gdown

        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(raw_zip), quiet=False)
    except Exception as e:
        print(f"[!] Automated download failed or link unavailable: {e}")
        print("\n[*] Manual Download Instructions:")
        print("1. Download LEVIR-CD from official repository or Google Drive mirror.")
        print(f"2. Save the ZIP file to: {raw_zip.resolve()}")
        print(f"3. Run: python scripts/download_levir_cd.py --organize {raw_zip.resolve()}")
        return

    if raw_zip.exists():
        organize_archive(raw_zip, target_dir)


def organize_archive(zip_path: Path, output_dir: Path) -> None:
    """Extract archive and organize into standard LEVIR-CD split layout."""
    print(f"[*] Extracting archive: {zip_path} -> {output_dir}")
    temp_extract = output_dir.parent / "temp_extract"
    temp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract)

    # Locate train/val/test folders within extracted content
    for split in ["train", "val", "test"]:
        target_split = output_dir / split
        target_split.mkdir(parents=True, exist_ok=True)

        found_split = list(temp_extract.rglob(f"*{split}*"))
        for candidate in found_split:
            if candidate.is_dir():
                for sub in ["A", "B", "label"]:
                    sub_src = candidate / sub
                    if sub_src.exists():
                        sub_dst = target_split / sub
                        sub_dst.mkdir(parents=True, exist_ok=True)
                        for item in sub_src.glob("*.png"):
                            shutil.copy(item, sub_dst / item.name)

    shutil.rmtree(temp_extract, ignore_errors=True)
    print("[+] Extraction and reorganization complete!")
    validate_structure(output_dir)


def main():
    parser = argparse.ArgumentParser(description="LEVIR-CD Data Management")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic mini dataset")
    parser.add_argument("--download", action="store_true", help="Download official dataset")
    parser.add_argument("--organize", type=str, help="Organize specified ZIP archive")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity")

    args = parser.parse_args()
    target_dir = config.paths.levir_cd_dir

    if args.synthetic:
        generate_synthetic_dataset(target_dir)
        validate_structure(target_dir)
    elif args.download:
        download_dataset(target_dir)
    elif args.organize:
        organize_archive(Path(args.organize), target_dir)
    elif args.verify:
        validate_structure(target_dir)
    else:
        # Default behavior if no flag passed: verify, if empty generate synthetic
        if not validate_structure(target_dir):
            print("\nNo dataset found. Generating synthetic dataset for offline dev...")
            generate_synthetic_dataset(target_dir)
            validate_structure(target_dir)


if __name__ == "__main__":
    main()
