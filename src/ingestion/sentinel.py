"""
Sentinel-2 Satellite Data Ingestion Handler for SpaceNetra.

Handles 10m resolution bands (B02 Blue, B03 Green, B04 Red, B08 NIR), cloud cover filtering,
and reflectance normalization for Indian Areas of Interest (AOIs).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class Sentinel2Handler:
    """
    Handler for loading, stacking, and filtering Sentinel-2 satellite scenes.
    """

    # Priority 10m bands from DATA_PIPELINE.md
    BAND_MAPPING = {
        "B02": "Blue",
        "B03": "Green",
        "B04": "Red",
        "B08": "NIR",
    }

    def __init__(self, max_cloud_cover: float = 10.0):
        self.max_cloud_cover = max_cloud_cover

    @staticmethod
    def stack_10m_bands(
        b02: np.ndarray,
        b03: np.ndarray,
        b04: np.ndarray,
        b08: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Stacks 10m Sentinel-2 bands into RGB (3 channels) or RGB+NIR (4 channels).

        Args:
            b02: Blue band array (H, W).
            b03: Green band array (H, W).
            b04: Red band array (H, W).
            b08: NIR band array (H, W) (optional).
            normalize: Scale 12-bit reflectance (0..10000) to [0.0, 1.0].

        Returns:
            Stacked array of shape (H, W, 3) or (H, W, 4).
        """
        # Ensure 2D (H, W)
        b02_2d = b02.squeeze()
        b03_2d = b03.squeeze()
        b04_2d = b04.squeeze()

        if b08 is not None:
            b08_2d = b08.squeeze()
            # Order: Red (B04), Green (B03), Blue (B02), NIR (B08)
            stacked = np.stack([b04_2d, b03_2d, b02_2d, b08_2d], axis=-1)
        else:
            # RGB composite: Red (B04), Green (B03), Blue (B02)
            stacked = np.stack([b04_2d, b03_2d, b02_2d], axis=-1)

        if normalize:
            if stacked.dtype == np.uint8:
                stacked = stacked.astype(np.float32) / 255.0
            else:
                stacked = np.clip(stacked.astype(np.float32) / 10000.0, 0.0, 1.0)

        return stacked

    def filter_scenes(self, scenes: List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, float]]]:
        """
        Filters scene metadata list based on max_cloud_cover threshold.

        Args:
            scenes: List of dicts containing 'scene_id' and 'cloud_cover'.

        Returns:
            Filtered list of scene dicts.
        """
        return [
            scene
            for scene in scenes
            if scene.get("cloud_cover", 100.0) <= self.max_cloud_cover
        ]
