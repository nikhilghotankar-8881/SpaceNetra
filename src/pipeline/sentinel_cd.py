"""
Sentinel-2 End-to-End Satellite Change Detection Pipeline Engine for SpaceNetra.

Handles reading bi-temporal Sentinel-2 imagery, cloud masking, radiometry scaling,
sliding window patch tiling, model inference (Siamese U-Net / ChangeFormer), patch
stitching, and georeferenced GeoTIFF change map generation.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.data.tiling import ImageTiler, PatchStitcher
from src.ingestion.geotiff import GeoTIFFHandler
from src.ingestion.preprocessing import Sentinel2Preprocessor
from src.models.changeformer import ChangeFormer
from src.models.siamese_unet import SiameseUNet


class SentinelChangeDetector:
    """
    End-to-End Operational Sentinel-2 Change Detection Engine.
    """

    def __init__(
        self,
        model_or_arch: Union[str, nn.Module] = "siamese_unet",
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        tile_size: int = 256,
        overlap: int = 0,
        scale_factor: float = 10000.0,
    ):
        # 1. Device Setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Model Initialization
        if isinstance(model_or_arch, str):
            arch_lower = model_or_arch.lower()
            if arch_lower == "changeformer":
                self.model = ChangeFormer()
            else:
                self.model = SiameseUNet()
        elif isinstance(model_or_arch, nn.Module):
            self.model = model_or_arch
        else:
            raise TypeError(f"Invalid model_or_arch parameter type: {type(model_or_arch)}")

        # 3. Checkpoint Loading
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # 4. Pipeline Drivers
        self.preprocessor = Sentinel2Preprocessor(scale_factor=scale_factor)
        self.tiler = ImageTiler(tile_size=tile_size, overlap=overlap)
        self.stitcher = PatchStitcher(overlap=overlap)

    def predict_pair(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        scl_t1: Optional[np.ndarray] = None,
        scl_t2: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        batch_size: int = 4,
        is_scaled: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Execute end-to-end change detection on a pair of bi-temporal images.

        Args:
            img_t1: Pre-change scene array of shape (H, W, C) or (C, H, W).
            img_t2: Post-change scene array of shape (H, W, C) or (C, H, W).
            scl_t1: Optional SCL cloud layer for scene 1 (H, W).
            scl_t2: Optional SCL cloud layer for scene 2 (H, W).
            threshold: Binary classification threshold for change mask [0.0 - 1.0].
            batch_size: Inference batch size.
            is_scaled: Whether reflectance scaling has already been applied.

        Returns:
            Dict containing 'probability_map' (H, W) float32 and 'change_mask' (H, W) uint8 (0 or 255).
        """
        # Ensure HWC array shape layout
        if img_t1.ndim == 3 and img_t1.shape[0] in [3, 4] and img_t1.shape[0] < img_t1.shape[1]:
            img_t1 = np.transpose(img_t1, (1, 2, 0))
        if img_t2.ndim == 3 and img_t2.shape[0] in [3, 4] and img_t2.shape[0] < img_t2.shape[1]:
            img_t2 = np.transpose(img_t2, (1, 2, 0))

        # Ensure RGB 3-channel input for standard CD models
        if img_t1.ndim == 3 and img_t1.shape[-1] > 3:
            img_t1 = img_t1[..., :3]
        if img_t2.ndim == 3 and img_t2.shape[-1] > 3:
            img_t2 = img_t2[..., :3]

        # 1. Preprocessing (reflectance scaling + SCL cloud masking)
        prep_t1 = self.preprocessor.process(img_t1, scl=scl_t1, is_scaled=is_scaled)
        prep_t2 = self.preprocessor.process(img_t2, scl=scl_t2, is_scaled=is_scaled)

        # 2. Patch Tiling
        patches_t1, tile_info = self.tiler.split_image(prep_t1)
        patches_t2, _ = self.tiler.split_image(prep_t2)

        # 3. Model Inference in Batches
        num_patches = len(patches_t1)
        pred_patches_list = []

        with torch.no_grad():
            for i in range(0, num_patches, batch_size):
                batch_p1 = patches_t1[i : i + batch_size]
                batch_p2 = patches_t2[i : i + batch_size]

                # Convert HWC numpy -> CHW torch tensors
                tensor_p1 = torch.from_numpy(np.transpose(batch_p1, (0, 3, 1, 2))).float().to(self.device)
                tensor_p2 = torch.from_numpy(np.transpose(batch_p2, (0, 3, 1, 2))).float().to(self.device)

                # Forward pass
                outputs = self.model(tensor_p1, tensor_p2)
                # Output shape (B, 1, H_tile, W_tile) -> squeeze to (B, H_tile, W_tile)
                if outputs.ndim == 4:
                    outputs = outputs.squeeze(1)

                pred_patches_list.append(outputs.cpu().numpy())

        pred_patches = np.concatenate(pred_patches_list, axis=0)

        # 4. Patch Stitching back to full scene dimensions
        prob_map = self.stitcher.stitch_patches(pred_patches, tile_info)

        # 5. Thresholding to create binary change mask
        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

        return {
            "probability_map": prob_map,
            "change_mask": binary_mask,
        }

    def predict_geotiff_pair(
        self,
        t1_path: str,
        t2_path: str,
        output_path: str,
        scl_t1_path: Optional[str] = None,
        scl_t2_path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> str:
        """
        Execute change detection directly on input GeoTIFF files and export georeferenced output.

        Args:
            t1_path: Path to pre-change GeoTIFF scene.
            t2_path: Path to post-change GeoTIFF scene.
            output_path: Target path to write output change map GeoTIFF.
            scl_t1_path: Optional path to T1 SCL GeoTIFF.
            scl_t2_path: Optional path to T2 SCL GeoTIFF.
            threshold: Binary classification threshold [0.0 - 1.0].

        Returns:
            Absolute path to output GeoTIFF file.
        """
        # Read T1 and T2 rasters
        data_t1, meta_t1 = GeoTIFFHandler.read_raster(t1_path)
        data_t2, _ = GeoTIFFHandler.read_raster(t2_path)

        scl_t1 = GeoTIFFHandler.read_raster(scl_t1_path)[0] if scl_t1_path else None
        scl_t2 = GeoTIFFHandler.read_raster(scl_t2_path)[0] if scl_t2_path else None

        results = self.predict_pair(
            img_t1=data_t1,
            img_t2=data_t2,
            scl_t1=scl_t1,
            scl_t2=scl_t2,
            threshold=threshold,
        )

        change_mask = results["change_mask"]

        # Write output raster preserving CRS and Transform
        GeoTIFFHandler.write_raster(
            output_path=output_path,
            data=change_mask,
            transform=meta_t1["transform"],
            crs=meta_t1["crs"],
        )

        return str(Path(output_path).resolve())
