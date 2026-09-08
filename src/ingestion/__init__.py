"""
Ingestion package for satellite rasters, GeoTIFFs, and Sentinel-2 data products.
"""

from src.ingestion.geotiff import GeoTIFFReader
from src.ingestion.sentinel import Sentinel2Handler

__all__ = [
    "GeoTIFFReader",
    "Sentinel2Handler",
]
