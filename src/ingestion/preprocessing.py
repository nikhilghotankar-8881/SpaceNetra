"""
Sentinel-2 Image Pre-processing & Cloud Masking Module for SpaceNetra.

Provides Scene Classification Layer (SCL) cloud masking, reflectance scaling,
and band harmonization for satellite imagery prior to change detection.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


# SCL (Scene Classification Layer) class constants for Sentinel-2 L2A
SCL_CLASSES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "DARK_AREA_PIXELS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW",
}

DEFAULT_INVALID_SCL_CLASSES = [3, 8, 9, 10, 11]


def mask_clouds_scl(
    data: np.ndarray,
    scl: np.ndarray,
    invalid_classes: Optional[List[int]] = None,
    fill_value: float = 0.0,
    channel_first: Optional[bool] = None,
) -> np.ndarray:
    """
    Mask out clouds, cloud shadows, and invalid pixels using Sentinel-2 SCL layer.

    Args:
        data: Image array of shape (H, W), (H, W, C), or (C, H, W).
        scl: Scene classification map of shape (H, W).
        invalid_classes: List of SCL integer class codes to mask out.
                         Defaults to [3, 8, 9, 10, 11] (shadows, clouds, cirrus, snow).
        fill_value: Value to replace masked pixels with (default 0.0).
        channel_first: Force channel layout (True for (C, H, W), False for (H, W, C)).
                       If None, inferred automatically from spatial dimensions.

    Returns:
        Masked image array of the same shape and type as data.
    """
    if invalid_classes is None:
        invalid_classes = DEFAULT_INVALID_SCL_CLASSES

    scl_2d = np.squeeze(scl)
    if scl_2d.ndim != 2:
        raise ValueError(f"SCL array must be 2D after squeezing, got shape {scl.shape}")

    # Build binary boolean mask where True indicates an invalid/cloudy pixel
    cloud_mask = np.isin(scl_2d, invalid_classes)

    output = data.copy()

    # Apply mask based on input array dimensions
    if output.ndim == 2:
        output[cloud_mask] = fill_value
    elif output.ndim == 3:
        is_chw = channel_first
        if is_chw is None:
            # Infer from shape
            if output.shape[1:] == scl_2d.shape and output.shape[:2] != scl_2d.shape:
                is_chw = True
            else:
                is_chw = False

        if is_chw:
            output[:, cloud_mask] = fill_value
        else:
            output[cloud_mask, :] = fill_value
    else:
        raise ValueError(f"Unsupported data array shape: {output.shape}")

    return output


def scale_reflectance(
    data: np.ndarray,
    scale_factor: float = 10000.0,
    offset: float = 0.0,
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> np.ndarray:
    """
    Rescale Sentinel-2 Digital Numbers (DN) to BOA Surface Reflectance values.

    Args:
        data: Input raw integer array or float array.
        scale_factor: Scaling factor (default 10000.0 for Sentinel-2).
        offset: Additive offset (default 0.0).
        clip_range: Optional tuple (min, max) to clip values. Defaults to (0.0, 1.0).

    Returns:
        Float32 array scaled to reflectance.
    """
    scaled = (data.astype(np.float32) + offset) / scale_factor
    if clip_range is not None:
        scaled = np.clip(scaled, clip_range[0], clip_range[1])
    return scaled


def normalize_bands(
    data: np.ndarray,
    mean: Optional[Union[List[float], np.ndarray]] = None,
    std: Optional[Union[List[float], np.ndarray]] = None,
) -> np.ndarray:
    """
    Normalize image channels with mean and standard deviation.

    Args:
        data: Floating point image array (H, W, C) or (C, H, W).
        mean: Channel means.
        std: Channel standard deviations.

    Returns:
        Normalized image array of float32 type.
    """
    output = data.astype(np.float32)
    if mean is None or std is None:
        return output

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    if output.ndim == 3:
        if output.shape[-1] == len(mean_arr):  # (H, W, C)
            output = (output - mean_arr) / (std_arr + 1e-7)
        elif output.shape[0] == len(mean_arr):  # (C, H, W)
            output = (output - mean_arr[:, None, None]) / (std_arr[:, None, None] + 1e-7)
        else:
            raise ValueError("Channel dimension mismatch with mean/std vector length.")
    return output


class Sentinel2Preprocessor:
    """
    Comprehensive Sentinel-2 Preprocessing Pipeline Engine.
    Combines cloud masking, reflectance scaling, and band normalization.
    """

    def __init__(
        self,
        scale_factor: float = 10000.0,
        invalid_scl_classes: Optional[List[int]] = None,
        fill_value: float = 0.0,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.scale_factor = scale_factor
        self.invalid_scl_classes = invalid_scl_classes or DEFAULT_INVALID_SCL_CLASSES
        self.fill_value = fill_value
        self.mean = mean
        self.std = std

    def process(
        self,
        data: np.ndarray,
        scl: Optional[np.ndarray] = None,
        is_scaled: bool = False,
    ) -> np.ndarray:
        """
        Execute full preprocessing sequence on Sentinel-2 data array.

        Args:
            data: Raw or pre-scaled multi-spectral image array (H, W, C) or (C, H, W).
            scl: Optional Scene Classification Layer array (H, W).
            is_scaled: If True, skips scale_reflectance step.

        Returns:
            Preprocessed float32 image array.
        """
        # Step 1: Scale raw DN values to reflectance [0, 1] if needed
        if not is_scaled:
            output = scale_reflectance(data, scale_factor=self.scale_factor)
        else:
            output = data.astype(np.float32)

        # Step 2: Apply SCL cloud & shadow masking if SCL is provided
        if scl is not None:
            output = mask_clouds_scl(
                output,
                scl=scl,
                invalid_classes=self.invalid_scl_classes,
                fill_value=self.fill_value,
            )

        # Step 3: Channel normalization if mean & std are configured
        if self.mean is not None and self.std is not None:
            output = normalize_bands(output, mean=self.mean, std=self.std)

        return output
