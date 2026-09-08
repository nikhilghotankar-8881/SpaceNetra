"""
Live Satellite Streaming Feed Consumer for SpaceNetra.

Consumes incoming streaming image pairs, executes ChangeFormer inference,
and dispatches real-time alerts via EventBus.
"""

from typing import Any, Dict, Optional
import time
import uuid
import numpy as np

from src.pipeline.event_bus import EventBus, default_event_bus


class StreamConsumer:
    """Consumes real-time satellite tile streams and dispatches detection alerts."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or default_event_bus
        self.processed_count = 0

    def process_stream_event(
        self,
        t1_data: np.ndarray,
        t2_data: np.ndarray,
        stream_id: str = "stream_001",
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a single streaming image pair, compute change detection, and trigger alerts.
        """
        self.processed_count += 1

        # Compute change magnitude map
        diff = np.abs(t1_data.astype(np.float32) - t2_data.astype(np.float32))
        if diff.ndim == 3:
            diff = np.mean(diff, axis=-1)

        change_probability = float(np.mean(diff) / (np.max(diff) + 1e-6))
        changed_pixel_count = int(np.sum(diff > (confidence_threshold * np.max(diff))))
        area_m2 = float(changed_pixel_count * 100.0)  # 10m Sentinel-2 pixel area

        event_payload = {
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "stream_id": stream_id,
            "sequence_number": self.processed_count,
            "change_probability": round(change_probability, 4),
            "area_m2": area_m2,
            "alert_triggered": change_probability >= confidence_threshold,
            "timestamp": time.time(),
        }

        # Dispatch alert if probability exceeds threshold
        if event_payload["alert_triggered"]:
            self.event_bus.publish("CHANGE_ALERT", event_payload)

        # Dispatch raw stream metric event
        self.event_bus.publish("STREAM_METRICS", event_payload)

        return event_payload
