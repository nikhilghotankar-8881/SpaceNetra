"""
Model Integrity Verification Module for SpaceNetra AI Models.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelIntegrityVerifier:
    """Verifies SHA-256 integrity checksums for PyTorch model weights."""

    @staticmethod
    def compute_checksum(filepath: Path) -> str:
        """Compute SHA-256 hash of a file in chunks."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def verify_checksum(filepath: Path, expected_hash: str) -> bool:
        """Verify file against expected SHA-256 hash."""
        if not filepath.exists():
            logger.error(f"Model checkpoint not found: {filepath}")
            return False
        computed = ModelIntegrityVerifier.compute_checksum(filepath)
        matches = computed.lower() == expected_hash.lower()
        if not matches:
            logger.warning(
                f"Model integrity checksum mismatch for {filepath.name}! "
                f"Expected: {expected_hash}, Computed: {computed}"
            )
        return matches

    @staticmethod
    def generate_manifest(checkpoint_dir: Path, output_manifest: Optional[Path] = None) -> Dict[str, str]:
        """Generate SHA-256 manifest dictionary for all model files in directory."""
        manifest = {}
        if not checkpoint_dir.exists():
            return manifest

        for pth in checkpoint_dir.glob("*.pth"):
            manifest[pth.name] = ModelIntegrityVerifier.compute_checksum(pth)
        for pt in checkpoint_dir.glob("*.pt"):
            manifest[pt.name] = ModelIntegrityVerifier.compute_checksum(pt)

        if output_manifest:
            output_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(output_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        return manifest

    @staticmethod
    def verify_manifest(manifest_path: Path, checkpoint_dir: Path) -> bool:
        """Verify all files in manifest against files in checkpoint_dir."""
        if not manifest_path.exists():
            logger.info(f"No checksum manifest found at {manifest_path}. Skipping integrity verify.")
            return True

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        all_ok = True
        for filename, expected_hash in manifest.items():
            filepath = checkpoint_dir / filename
            if not ModelIntegrityVerifier.verify_checksum(filepath, expected_hash):
                all_ok = False
        return all_ok
