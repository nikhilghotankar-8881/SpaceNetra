"""
Spatial Change Event Creation & GeoJSON Exporter Engine for SpaceNetra.

Aggregates pixel-level binary change maps into discrete physical change events
(e.g., building construction, deforestation), computes spatial metadata
(area in m², bounding box, centroid, mean intensity), and exports GeoJSON feature collections.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import label


class EventBuilder:
    """
    Spatial Clustering & Event Extraction Engine.
    """

    def __init__(
        self,
        min_area_pixels: int = 10,
        pixel_resolution_m: float = 10.0,
    ):
        self.min_area_pixels = min_area_pixels
        self.pixel_resolution_m = pixel_resolution_m

    def extract_events(
        self,
        change_mask: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
        transform: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts spatial change events from a binary change mask.

        Args:
            change_mask: 2D binary numpy array (H, W) where change > 0.
            prob_map: Optional 2D float32 prediction probability map (H, W).
            transform: Optional rasterio Affine transform for georeferencing.

        Returns:
            List of event dictionaries with spatial metrics.
        """
        mask_2d = np.squeeze(change_mask) > 0
        if mask_2d.ndim != 2:
            raise ValueError(f"change_mask must be 2D after squeezing, got shape {change_mask.shape}")

        # Connected component spatial labeling
        labeled_mask, num_features = label(mask_2d)

        events = []
        pixel_area_m2 = self.pixel_resolution_m ** 2

        for cluster_id in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_mask == cluster_id)
            pixel_count = len(y_indices)

            if pixel_count < self.min_area_pixels:
                continue  # Filter out noise clusters smaller than min_area_pixels

            min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
            min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))

            centroid_y = float(np.mean(y_indices))
            centroid_x = float(np.mean(x_indices))

            area_m2 = float(pixel_count * pixel_area_m2)

            mean_prob = 1.0
            max_prob = 1.0
            if prob_map is not None:
                prob_2d = np.squeeze(prob_map)
                cluster_probs = prob_2d[y_indices, x_indices]
                mean_prob = float(np.mean(cluster_probs))
                max_prob = float(np.max(cluster_probs))

            # Geo-coordinate transformation if transform is provided
            if transform is not None:
                # Top-left and bottom-right geo points
                geo_min_x, geo_min_y = transform * (min_x, min_y)
                geo_max_x, geo_max_y = transform * (max_x + 1, max_y + 1)
                geo_centroid_x, geo_centroid_y = transform * (centroid_x, centroid_y)

                bbox_coords = [geo_min_x, geo_min_y, geo_max_x, geo_max_y]
                centroid_coords = (geo_centroid_x, geo_centroid_y)
                polygon_coords = [
                    [geo_min_x, geo_min_y],
                    [geo_max_x, geo_min_y],
                    [geo_max_x, geo_max_y],
                    [geo_min_x, geo_max_y],
                    [geo_min_x, geo_min_y],
                ]
            else:
                bbox_coords = [min_x, min_y, max_x, max_y]
                centroid_coords = (centroid_x, centroid_y)
                polygon_coords = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y],
                ]

            event_id = f"EVENT_{len(events) + 1:04d}"

            events.append({
                "event_id": event_id,
                "cluster_index": cluster_id,
                "pixel_count": pixel_count,
                "area_m2": area_m2,
                "centroid": centroid_coords,
                "bbox": bbox_coords,
                "polygon": polygon_coords,
                "mean_probability": mean_prob,
                "max_probability": max_prob,
                "status": "DETECTED",
            })

        return events

    def export_geojson(
        self,
        events: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> str:
        """
        Exports extracted change events list to standard GeoJSON FeatureCollection file.

        Args:
            events: List of event dictionaries from extract_events.
            output_path: Path to write .geojson file.

        Returns:
            Absolute path string to output file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for event in events:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [event["polygon"]],
                },
                "properties": {
                    "event_id": event["event_id"],
                    "area_m2": event["area_m2"],
                    "pixel_count": event["pixel_count"],
                    "mean_probability": event["mean_probability"],
                    "max_probability": event["max_probability"],
                    "status": event["status"],
                },
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return str(out_file.resolve())
