"""
Data Inspection & Validation Script for SpaceNetra (LEVIR-CD Dataset).
Performs 6 automated quality gates on LEVIR-CD data (A, B, and Label masks)
and saves an inspection report to JSON.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Add root directory to path for src import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config


def inspect_split(split_name: str, base_path: Path):
    """
    Runs quality gates 1-4 on a given split (train, val, test).
    Returns split statistics and a list of issues found.
    """
    split_dir = base_path / split_name
    a_dir = split_dir / "A"
    b_dir = split_dir / "B"
    label_dir = split_dir / "label"

    issues = []
    stats = {
        "split": split_name,
        "total_triplets": 0,
        "valid_triplets": 0,
        "change_ratios": [],
        "mean_change_ratio": 0.0,
        "min_change_ratio": 0.0,
        "max_change_ratio": 0.0,
        "zero_change_count": 0,
    }

    if not split_dir.exists():
        issues.append(f"Split directory missing: {split_dir}")
        return stats, issues

    a_files = sorted(list(a_dir.glob("*.png"))) if a_dir.exists() else []
    stats["total_triplets"] = len(a_files)

    if len(a_files) == 0:
        issues.append(f"No PNG files found in {a_dir}")
        return stats, issues

    valid_count = 0

    for a_path in a_files:
        filename = a_path.name
        b_path = b_dir / filename
        label_path = label_dir / filename

        # Quality Gate 1: Triplet Existence & Integrity
        if not b_path.exists():
            issues.append(f"Missing image B for triplet: {filename}")
            continue
        if not label_path.exists():
            issues.append(f"Missing label mask for triplet: {filename}")
            continue

        try:
            img_a = Image.open(a_path)
            img_b = Image.open(b_path)
            img_label = Image.open(label_path)

            img_a.verify()
            img_b.verify()
            img_label.verify()

            # Re-open after verify() as PIL requires
            img_a = Image.open(a_path)
            img_b = Image.open(b_path)
            img_label = Image.open(label_path)
        except Exception as e:
            issues.append(f"Corrupt image in triplet {filename}: {str(e)}")
            continue

        # Quality Gate 2: Dimension & Mode Consistency
        w_a, h_a = img_a.size
        w_b, h_b = img_b.size
        w_lbl, h_lbl = img_label.size

        if (w_a, h_a) != (1024, 1024) or (w_b, h_b) != (1024, 1024) or (w_lbl, h_lbl) != (1024, 1024):
            issues.append(
                f"Dimension mismatch in {filename}: A=({w_a}x{h_a}), B=({w_b}x{h_b}), Label=({w_lbl}x{h_lbl})"
            )

        if img_a.mode not in ("RGB", "L") or img_b.mode not in ("RGB", "L"):
            issues.append(f"Unexpected image mode in {filename}: A={img_a.mode}, B={img_b.mode}")

        # Quality Gate 3: Mask Value Validity
        label_arr = np.array(img_label)
        unique_vals = np.unique(label_arr)
        invalid_vals = set(unique_vals) - {0, 255}

        if len(invalid_vals) > 0:
            issues.append(f"Invalid mask values in {filename}: {invalid_vals} (expected 0 and 255)")

        # Quality Gate 4: Content Quality & Change Ratio
        arr_a = np.array(img_a)
        arr_b = np.array(img_b)

        if np.std(arr_a) < 1.0 or np.std(arr_b) < 1.0:
            issues.append(f"Blank / solid color image detected in {filename}")

        change_pixels = np.count_nonzero(label_arr > 0)
        total_pixels = label_arr.size
        change_ratio = float(change_pixels / total_pixels)

        stats["change_ratios"].append(change_ratio)
        if change_ratio == 0.0:
            stats["zero_change_count"] += 1

        valid_count += 1

    stats["valid_triplets"] = valid_count

    if stats["change_ratios"]:
        stats["mean_change_ratio"] = float(np.mean(stats["change_ratios"]))
        stats["min_change_ratio"] = float(np.min(stats["change_ratios"]))
        stats["max_change_ratio"] = float(np.max(stats["change_ratios"]))

    return stats, issues


def main():
    print("=" * 60)
    print("[*] SpaceNetra - Data Inspection & Validation Pipeline")
    print("=" * 60)

    data_dir = config.paths.levir_cd_dir
    print(f"[*] Inspecting dataset at: {data_dir}")

    if not data_dir.exists():
        print(f"[-] ERROR: Dataset directory does not exist: {data_dir}")
        sys.exit(1)

    splits = ["train", "val", "test"]
    all_stats = {}
    all_issues = []

    total_all_triplets = 0

    for split in splits:
        print(f"\n[*] Processing split: {split.upper()}...")
        stats, issues = inspect_split(split, data_dir)
        all_stats[split] = stats
        all_issues.extend([(split, issue) for issue in issues])
        total_all_triplets += stats["total_triplets"]

        print(f"    - Total Triplets : {stats['total_triplets']}")
        print(f"    - Valid Triplets : {stats['valid_triplets']}")
        print(f"    - Mean Change %  : {stats['mean_change_ratio']*100:.2f}%")
        print(f"    - Min Change %   : {stats['min_change_ratio']*100:.2f}%")
        print(f"    - Max Change %   : {stats['max_change_ratio']*100:.2f}%")
        print(f"    - Zero Change %  : {stats['zero_change_count']}")

    # Quality Gate 5: Split Size & Integrity Check
    print("\n" + "=" * 60)
    print("[*] QUALITY GATE 5: Split Size Verification")
    print("=" * 60)

    expected_real = {"train": 445, "val": 64, "test": 128}
    is_synthetic = total_all_triplets < 100

    if is_synthetic:
        print("[WARN] Synthetic / Mock dataset detected.")
        print(f"       Total samples across splits: {total_all_triplets}")
    else:
        print("[OK] Real LEVIR-CD dataset detected.")
        for split in splits:
            exp = expected_real[split]
            actual = all_stats[split]["total_triplets"]
            if actual == exp:
                print(f"[OK] {split}: {actual}/{exp} triplets present.")
            else:
                issue_msg = f"{split} split count mismatch: found {actual}, expected {exp}"
                print(f"[WARN] {issue_msg}")
                all_issues.append((split, issue_msg))

    # Summary Report
    print("\n" + "=" * 60)
    print("[*] INSPECTION SUMMARY")
    print("=" * 60)

    if len(all_issues) == 0:
        print("[OK] ALL DATASET QUALITY GATES PASSED PERFECTLY!")
    else:
        print(f"[WARN] Found {len(all_issues)} issue(s) across splits:")
        for split, issue in all_issues:
            print(f"  - [{split.upper()}] {issue}")

    # Output JSON Report
    report_path = config.paths.metrics_dir / "data_inspection_report.json"
    report_data = {
        "is_synthetic": is_synthetic,
        "total_triplets": total_all_triplets,
        "splits": all_stats,
        "total_issues_found": len(all_issues),
        "issues": [f"[{s}] {i}" for s, i in all_issues],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Inspection report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
