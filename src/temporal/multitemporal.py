"""
Multi-Temporal Analysis Engine & Change Chronology Tracking for SpaceNetra.

Analyzes time-series stacks of Sentinel-2 satellite images (T1, T2, ..., Tn),
computes sequential step and baseline change transitions, calculates spatial cumulative
change frequency heatmaps, and tracks pixel-level change onset dates.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.pipeline.sentinel_cd import SentinelChangeDetector


class MultiTemporalAnalyzer:
    """
    Time-Series Analysis Engine for Satellite Imagery Sequences.
    """

    def __init__(
        self,
        detector: Optional[SentinelChangeDetector] = None,
        model_or_arch: str = "siamese_unet",
        checkpoint_path: Optional[str] = None,
    ):
        if detector is not None:
            self.detector = detector
        else:
            self.detector = SentinelChangeDetector(
                model_or_arch=model_or_arch,
                checkpoint_path=checkpoint_path,
            )

    def analyze_series(
        self,
        scene_list: List[np.ndarray],
        scl_list: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Processes a sequential time-series list of satellite image scenes.

        Args:
            scene_list: List of N scene image arrays (H, W, C) or (C, H, W). N >= 2.
            scl_list: Optional list of N Scene Classification Layer arrays (H, W).
            timestamps: Optional list of N timestamp strings (e.g. ['2023-01-01', '2023-04-01', ...]).
            threshold: Binary change map classification threshold [0.0 - 1.0].

        Returns:
            Dict containing:
                - 'step_transitions': List of dicts for consecutive pairs (T_i -> T_{i+1}).
                - 'baseline_transitions': List of dicts for baseline pairs (T_0 -> T_i).
                - 'cumulative_frequency': Array of shape (H, W) counting change occurrences per pixel.
                - 'first_change_onset': Array of shape (H, W) storing 1-based step index of first detected change.
                - 'timestamps': Original or generated timestamp list.
        """
        num_scenes = len(scene_list)
        if num_scenes < 2:
            raise ValueError(f"Time-series stack must contain at least 2 scenes, got {num_scenes}")

        if timestamps is None:
            timestamps = [f"Epoch_{i+1}" for i in range(num_scenes)]

        # 1. Step Transitions (T_i -> T_{i+1})
        step_transitions = []
        step_masks = []

        for i in range(num_scenes - 1):
            scl_t1 = scl_list[i] if scl_list is not None else None
            scl_t2 = scl_list[i + 1] if scl_list is not None else None

            res = self.detector.predict_pair(
                img_t1=scene_list[i],
                img_t2=scene_list[i + 1],
                scl_t1=scl_t1,
                scl_t2=scl_t2,
                threshold=threshold,
            )

            res["t1_timestamp"] = timestamps[i]
            res["t2_timestamp"] = timestamps[i + 1]
            res["step_index"] = i + 1

            step_transitions.append(res)
            step_masks.append(res["change_mask"])

        # 2. Baseline Transitions (T_0 -> T_i)
        baseline_transitions = []
        for i in range(1, num_scenes):
            scl_t0 = scl_list[0] if scl_list is not None else None
            scl_ti = scl_list[i] if scl_list is not None else None

            res = self.detector.predict_pair(
                img_t1=scene_list[0],
                img_t2=scene_list[i],
                scl_t1=scl_t0,
                scl_t2=scl_ti,
                threshold=threshold,
            )
            res["t0_timestamp"] = timestamps[0]
            res["ti_timestamp"] = timestamps[i]
            res["target_index"] = i

            baseline_transitions.append(res)

        # 3. Cumulative Frequency Map
        h, w = step_masks[0].shape
        cumulative_freq = np.zeros((h, w), dtype=np.uint16)
        for mask in step_masks:
            cumulative_freq += (mask > 0).astype(np.uint16)

        # 4. First Change Onset Map
        first_onset = np.zeros((h, w), dtype=np.uint8)
        for idx, mask in enumerate(step_masks):
            step_num = idx + 1  # 1-indexed step
            change_pixels = (mask > 0)
            # Assign step_num to pixels that changed and have not been recorded yet
            unrecorded = (first_onset == 0) & change_pixels
            first_onset[unrecorded] = step_num

        return {
            "step_transitions": step_transitions,
            "baseline_transitions": baseline_transitions,
            "cumulative_frequency": cumulative_freq,
            "first_change_onset": first_onset,
            "timestamps": timestamps,
        }
