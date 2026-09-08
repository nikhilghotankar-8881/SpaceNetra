"""
GeoTIFF and Cloud-Optimized GeoTIFF (COG) reader & writer for SpaceNetra.

Extracts spatial metadata (CRS, bounds, transform, resolution) and reads/writes raster data.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
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
        """
        return GeoTIFFHandler.read_raster(self.filepath, normalize=normalize)


class GeoTIFFHandler:
    """
    Unified Handler for reading and writing GeoTIFF rasters with spatial metadata.
    """

    @staticmethod
    def read_raster(filepath: Union[str, Path], normalize: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        filepath = Path(filepath)
        try:
            import rasterio

            with rasterio.open(filepath) as src:
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
        except Exception:
            # Fallback using PIL for non-rasterio environments or test synthetic images
            from PIL import Image

            img = Image.open(filepath)
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

        # Transpose (C, H, W) -> (H, W, C) or (H, W) if C==1
        raster_array = data.transpose(1, 2, 0)
        if raster_array.shape[-1] == 1:
            raster_array = raster_array.squeeze(-1)

        if normalize:
            if raster_array.dtype == np.uint8:
                raster_array = raster_array.astype(np.float32) / 255.0
            elif raster_array.dtype in (np.uint16, np.int32, np.int64):
                raster_array = np.clip(raster_array.astype(np.float32) / 10000.0, 0.0, 1.0)
            elif np.issubdtype(raster_array.dtype, np.floating):
                raster_array = np.clip(raster_array.astype(np.float32), 0.0, 1.0)

        return raster_array, metadata

    @staticmethod
    def write_raster(
        output_path: Union[str, Path],
        data: np.ndarray,
        transform: Optional[Any] = None,
        crs: Optional[str] = None,
    ) -> str:
        """
        Writes a numpy array to GeoTIFF raster file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if data.ndim == 2:
            h, w = data.shape
            c = 1
            data_chw = data[np.newaxis, ...]
        elif data.ndim == 3:
            h, w, c = data.shape
            data_chw = data.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported array shape for writing raster: {data.shape}")

        try:
            import rasterio
            from rasterio.crs import CRS

            crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs

            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=c,
                dtype=data.dtype,
                crs=crs_obj,
                transform=transform,
            ) as dst:
                dst.write(data_chw)
        except Exception:
            # Fallback for PIL/Non-rasterio environments
            from PIL import Image

            if data.dtype != np.uint8:
                data_img = (np.clip(data, 0, 255)).astype(np.uint8)
            else:
                data_img = data

            img = Image.fromarray(data_img)
            img.save(out_path)

        return str(out_path.resolve())
