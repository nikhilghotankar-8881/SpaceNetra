"""
Image tiling and patch stitching utilities for SpaceNetra.

Provides functions to split arbitrary-sized satellite images into model-compatible
patches (with optional overlap) and stitch predicted patch masks back into full-resolution
change maps using weighted blending.
"""

from typing import List, Tuple
import numpy as np


def split_into_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Splits an image array into square patches using a sliding window.

    If image dimensions are not evenly divisible by stride, reflect-padding is
    applied to ensure full coverage up to the image boundaries.

    Args:
        image: Input array of shape (H, W) or (H, W, C).
        patch_size: Side length of square patches in pixels.
        stride: Step size between patches. If stride < patch_size, patches overlap.

    Returns:
        patches: List of patch arrays of shape (patch_size, patch_size, ...).
        positions: List of (y, x) top-left corner coordinates relative to padded image.
        padded_size: (padded_H, padded_W) dimensions of the canvas.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D (H, W) or 3D (H, W, C), got shape {image.shape}")

    h, w = image.shape[:2]

    # Calculate required padded height
    if h <= patch_size:
        padded_h = patch_size
    else:
        rem_h = (h - patch_size) % stride
        padded_h = h if rem_h == 0 else h + (stride - rem_h)

    # Calculate required padded width
    if w <= patch_size:
        padded_w = patch_size
    else:
        rem_w = (w - patch_size) % stride
        padded_w = w if rem_w == 0 else w + (stride - rem_w)

    pad_h = padded_h - h
    pad_w = padded_w - w

    if pad_h > 0 or pad_w > 0:
        if image.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        padded_image = np.pad(image, pad_width, mode="reflect")
    else:
        padded_image = image

    patches = []
    positions = []

    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            if image.ndim == 2:
                patch = padded_image[y : y + patch_size, x : x + patch_size]
            else:
                patch = padded_image[y : y + patch_size, x : x + patch_size, :]
            patches.append(patch)
            positions.append((y, x))

    return patches, positions, (padded_h, padded_w)


def stitch_patches(
    patches: List[np.ndarray],
    positions: List[Tuple[int, int]],
    original_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    patch_size: int = 256,
) -> np.ndarray:
    """
    Reconstructs a full image from patches using weighted mean blending across overlaps.

    Args:
        patches: List of patch arrays (predictions or image tiles).
        positions: List of (y, x) top-left coordinates for each patch.
        original_size: (H, W) target dimensions for final cropped output.
        padded_size: (padded_H, padded_W) canvas size from split_into_patches.
        patch_size: Side length of square patches.

    Returns:
        Reconstructed array cropped to original_size.
    """
    if not patches:
        raise ValueError("patches list cannot be empty")

    sample_patch = patches[0]
    padded_h, padded_w = padded_size

    if sample_patch.ndim == 2:
        canvas_shape = (padded_h, padded_w)
    else:
        canvas_shape = (padded_h, padded_w, sample_patch.shape[2])

    canvas = np.zeros(canvas_shape, dtype=np.float32)
    weight_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)

    for patch, (y, x) in zip(patches, positions):
        patch_f = patch.astype(np.float32)
        if sample_patch.ndim == 2:
            canvas[y : y + patch_size, x : x + patch_size] += patch_f
        else:
            canvas[y : y + patch_size, x : x + patch_size, :] += patch_f

        weight_canvas[y : y + patch_size, x : x + patch_size] += 1.0

    # Normalize by overlap counts
    weight_denom = np.maximum(weight_canvas, 1e-7)
    if sample_patch.ndim == 2:
        canvas = canvas / weight_denom
    else:
        canvas = canvas / weight_denom[:, :, None]

    # Crop to original un-padded size
    orig_h, orig_w = original_size
    if sample_patch.ndim == 2:
        cropped = canvas[:orig_h, :orig_w]
    else:
        cropped = canvas[:orig_h, :orig_w, :]

    # Cast back to original integer dtype if applicable
    if np.issubdtype(sample_patch.dtype, np.integer):
        if np.issubdtype(sample_patch.dtype, np.unsignedinteger):
            info = np.iinfo(sample_patch.dtype)
            cropped = np.clip(np.round(cropped), info.min, info.max)
        cropped = cropped.astype(sample_patch.dtype)

    return cropped


def split_and_stitch_triplet(
    img_a: np.ndarray,
    img_b: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[int, int]],
    Tuple[int, int],
    Tuple[int, int],
]:
    """
    Splits a bi-temporal image pair (T1, T2) into spatial patch pairs.

    Ensures that T1 and T2 are split using the exact same grid positions.

    Args:
        img_a: T1 image array (H, W) or (H, W, C).
        img_b: T2 image array (H, W) or (H, W, C).
        patch_size: Side length of square patches.
        stride: Step size between patches.

    Returns:
        patch_pairs: List of (patch_a, patch_b) tuples.
        positions: List of (y, x) top-left coordinates.
        original_size: Original (H, W) dimensions.
        padded_size: Padded (H, W) canvas dimensions.
    """
    if img_a.shape[:2] != img_b.shape[:2]:
        raise ValueError(
            f"Image dimensions must match! Got img_a {img_a.shape[:2]} vs img_b {img_b.shape[:2]}"
        )

    patches_a, positions, padded_size = split_into_patches(
        img_a, patch_size=patch_size, stride=stride
    )
    patches_b, _, _ = split_into_patches(img_b, patch_size=patch_size, stride=stride)

    patch_pairs = list(zip(patches_a, patches_b))
    original_size = (img_a.shape[0], img_a.shape[1])

    return patch_pairs, positions, original_size, padded_size
