"""
Image loading utilities for Repressilator analysis.

This module provides functions to load and sort time-series fluorescence
microscopy images in the correct temporal order.
"""

import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


def parse_timepoint(filename: str) -> float:
    """
    Extract the timepoint in minutes from a filename.

    Args:
        filename: Image filename in format 'sample_t+{time}m.png'

    Returns:
        Time in minutes as a float

    Examples:
        >>> parse_timepoint('sample_t+75m.png')
        75.0
        >>> parse_timepoint('sample_t+1200m_phase.png')
        1200.0
    """
    match = re.search(r't\+(\d+)m', filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse timepoint from filename: {filename}")


def load_image_paths(directory: str, sort_by_time: bool = True) -> List[Tuple[float, Path]]:
    """
    Load all image paths from a directory and optionally sort by timepoint.

    Args:
        directory: Path to directory containing images
        sort_by_time: If True, sort images by timepoint

    Returns:
        List of (timepoint, Path) tuples
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_paths = []
    for img_path in dir_path.glob("*.png"):
        try:
            timepoint = parse_timepoint(img_path.name)
            image_paths.append((timepoint, img_path))
        except ValueError as e:
            print(f"Warning: Skipping file {img_path.name}: {e}")

    if sort_by_time:
        image_paths.sort(key=lambda x: x[0])

    return image_paths


def load_image(path: Path) -> np.ndarray:
    """
    Load a single image as a numpy array.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array with shape (height, width, channels) for RGB
        or (height, width) for grayscale
    """
    img = Image.open(path)
    return np.array(img)


def load_timeseries(
    intensity_dir: str,
    phase_dir: str,
) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """
    Load complete time-series of intensity and phase images.

    Args:
        intensity_dir: Directory containing fluorescence intensity images
        phase_dir: Directory containing phase contrast images

    Returns:
        Tuple of:
            - timepoints: List of timepoints in minutes
            - intensity_images: List of intensity image arrays
            - phase_images: List of phase contrast image arrays
    """
    intensity_paths = load_image_paths(intensity_dir, sort_by_time=True)
    phase_paths = load_image_paths(phase_dir, sort_by_time=True)

    # Verify we have matching timepoints
    intensity_times = [t for t, _ in intensity_paths]
    phase_times = [t for t, _ in phase_paths]

    if intensity_times != phase_times:
        print("Warning: Intensity and phase images have different timepoints")
        print(f"Intensity: {len(intensity_times)} images")
        print(f"Phase: {len(phase_times)} images")

    # Load images
    timepoints = intensity_times
    intensity_images = [load_image(path) for _, path in intensity_paths]
    phase_images = [load_image(path) for _, path in phase_paths]

    return timepoints, intensity_images, phase_images


def get_channel_data(rgb_image: np.ndarray, channel: str = 'red') -> np.ndarray:
    """
    Extract a specific color channel from an RGB image.

    Args:
        rgb_image: RGB image array with shape (height, width, 3)
        channel: Channel to extract ('red', 'green', or 'blue')

    Returns:
        Single channel array with shape (height, width)
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}
    if channel.lower() not in channel_map:
        raise ValueError(f"Invalid channel: {channel}. Must be 'red', 'green', or 'blue'")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")

    return rgb_image[:, :, channel_map[channel.lower()]]
