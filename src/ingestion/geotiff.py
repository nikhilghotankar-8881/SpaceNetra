"""
GeoTIFF and Cloud-Optimized GeoTIFF (COG) reader for SpaceNetra.

Extracts spatial metadata (CRS, bounds, transform, resolution) and reads raster data
into normalized NumPy float32 arrays.
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Any
import numpy as np


class GeoTIFFReader:
    """
    Reader for satellite GeoTIFF rasters and COG files.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def read_raster(self, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads GeoTIFF data and spatial metadata.

        Args:
            normalize: If True, scales 12-bit digital numbers (0..10000) to [0.0, 1.0].

        Returns:
            Tuple of (raster_array, metadata_dict)
        """
        try:
            import rasterio

            with rasterio.open(self.filepath) as src:
                data = src.read()  # (C, H, W)
                metadata = {
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": str(src.crs),
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "nodata": src.nodata,
                }
        except ImportError:
            # Fallback for environments where rasterio is not installed
            from PIL import Image

            img = Image.open(self.filepath)
            data = np.array(img)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim == 3 and data.shape[2] in (3, 4):
                data = data.transpose(2, 0, 1)

            c, h, w = data.shape
            metadata = {
                "width": w,
                "height": h,
                "count": c,
                "crs": "EPSG:4326",
                "bounds": (0.0, 0.0, float(w), float(h)),
                "transform": None,
                "nodata": None,
            }

        # Transpose to (H, W, C)
        raster_array = data.transpose(1, 2, 0)

        if normalize:
            if raster_array.dtype == np.uint8:
                raster_array = raster_array.astype(np.float32) / 255.0
            elif raster_array.dtype in (np.uint16, np.int32, np.int64):
                # Standard Sentinel-2 reflectance scaling (0..10,000 -> 0.0..1.0)
                raster_array = np.clip(raster_array.astype(np.float32) / 10000.0, 0.0, 1.0)
            elif np.issubdtype(raster_array.dtype, np.floating):
                raster_array = np.clip(raster_array.astype(np.float32), 0.0, 1.0)

        return raster_array, metadata
