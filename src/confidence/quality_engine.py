"""
Confidence Scoring & Alignment Quality Engine for SpaceNetra.

Evaluates spatial co-registration alignment quality, cloud/shadow contamination ratio,
prediction certainty, and overall confidence score for satellite change detection predictions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

DEFAULT_INVALID_SCL = [3, 8, 9, 10, 11]


class ConfidenceEngine:
    """
    Quality Evaluation & Confidence Scoring Engine for Satellite Change Detection.
    """

    def __init__(
        self,
        weight_alignment: float = 0.4,
        weight_cloud_free: float = 0.3,
        weight_certainty: float = 0.3,
        invalid_scl_classes: Optional[List[int]] = None,
    ):
        self.weight_alignment = weight_alignment
        self.weight_cloud_free = weight_cloud_free
        self.weight_certainty = weight_certainty
        self.invalid_scl_classes = invalid_scl_classes or DEFAULT_INVALID_SCL

    def compute_alignment_quality(self, img_t1: np.ndarray, img_t2: np.ndarray) -> float:
        """
        Computes spatial co-registration alignment score based on normalized mean absolute error.

        Returns:
            Float quality score in range [0.0, 1.0] (1.0 = perfectly aligned/consistent).
        """
        if img_t1.shape != img_t2.shape:
            raise ValueError(f"Image shapes must match! Got {img_t1.shape} vs {img_t2.shape}")

        t1_f = img_t1.astype(np.float32)
        t2_f = img_t2.astype(np.float32)

        # Convert to 2D grayscale if multi-channel
        if t1_f.ndim == 3:
            t1_f = np.mean(t1_f, axis=-1)
            t2_f = np.mean(t2_f, axis=-1)

        # Normalize arrays to [0, 1] for error calculation
        max_val1 = t1_f.max() if t1_f.max() > 0 else 1.0
        max_val2 = t2_f.max() if t2_f.max() > 0 else 1.0

        norm_t1 = t1_f / max_val1
        norm_t2 = t2_f / max_val2

        mae = np.mean(np.abs(norm_t1 - norm_t2))
        # Map MAE range [0.0, 0.5] to alignment score [1.0, 0.0]
        alignment_score = float(np.clip(1.0 - (mae / 0.5), 0.0, 1.0))
        return alignment_score

    def compute_cloud_contamination(
        self,
        scl_t1: Optional[np.ndarray] = None,
        scl_t2: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculates cloud and cloud-shadow contamination ratio across scenes.

        Returns:
            Float ratio in range [0.0, 1.0] (0.0 = completely cloud-free, 1.0 = 100% cloudy).
        """
        cloud_ratios = []

        for scl in [scl_t1, scl_t2]:
            if scl is not None:
                scl_2d = np.squeeze(scl)
                cloud_mask = np.isin(scl_2d, self.invalid_scl_classes)
                ratio = np.mean(cloud_mask)
                cloud_ratios.append(ratio)

        if not cloud_ratios:
            return 0.0  # Default 0% cloud contamination if no SCL provided

        return float(np.mean(cloud_ratios))

    def compute_prediction_confidence(self, prob_map: np.ndarray) -> float:
        """
        Computes model prediction certainty based on probability margin |2*p - 1|.

        Returns:
            Float confidence score in range [0.0, 1.0] (1.0 = 100% confident predictions).
        """
        p = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
        margin = np.abs(2.0 * p - 1.0)
        confidence_score = float(np.mean(margin))
        return confidence_score

    def evaluate_quality(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
        scl_t1: Optional[np.ndarray] = None,
        scl_t2: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes comprehensive quality evaluation and computes composite confidence score.

        Returns:
            Dict containing detailed quality metrics and composite confidence rating.
        """
        alignment_score = self.compute_alignment_quality(img_t1, img_t2)
        cloud_ratio = self.compute_cloud_contamination(scl_t1, scl_t2)
        cloud_free_score = 1.0 - cloud_ratio

        if prob_map is not None:
            certainty_score = self.compute_prediction_confidence(prob_map)
        else:
            certainty_score = 1.0  # Default if prob_map is not available

        composite_score = (
            self.weight_alignment * alignment_score
            + self.weight_cloud_free * cloud_free_score
            + self.weight_certainty * certainty_score
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        if composite_score >= 0.8:
            status = "HIGH_CONFIDENCE"
        elif composite_score >= 0.5:
            status = "MODERATE_CONFIDENCE"
        else:
            status = "LOW_CONFIDENCE"

        return {
            "composite_confidence": composite_score,
            "status": status,
            "alignment_score": alignment_score,
            "cloud_free_score": cloud_free_score,
            "cloud_contamination_ratio": cloud_ratio,
            "prediction_certainty_score": certainty_score,
        }
