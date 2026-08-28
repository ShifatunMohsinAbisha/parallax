"""Unit tests for src.segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing import preprocess
from src.segmentation import (
    SegmentationResult,
    compute_mask_bbox,
    create_side_by_side_panel,
    filter_largest_component,
    overlay_mask,
    segment_object,
    segment_saliency_grabcut,
    smooth_mask,
    soft_mask_from_binary,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def sample_object_image() -> np.ndarray:
    """Create a synthetic 128×128 image with a colored circular object on a light background."""
    img = np.full((128, 128, 3), 240, dtype=np.uint8)
    # Draw a vibrant blue-red circular object in center
    cv2.circle(img, (64, 64), 32, (220, 50, 40), -1)
    cv2.circle(img, (64, 64), 16, (40, 60, 230), -1)
    return img


@pytest.fixture()
def sample_image_file(sample_object_image: np.ndarray, tmp_path: Path) -> Path:
    """Save the synthetic object image to disk."""
    p = tmp_path / "sample_toy.png"
    Image.fromarray(sample_object_image, mode="RGB").save(str(p))
    return p


@pytest.fixture()
def binary_test_mask() -> np.ndarray:
    """Create a 64×64 binary mask with a primary circle and a small noise dot."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), 16, 1, -1)  # Main component
    mask[5, 5] = 1                         # Stray noise pixel
    return mask


# ──────────────────────────────────────────────
# Mask Utilities Tests
# ──────────────────────────────────────────────

class TestMaskUtilities:
    def test_filter_largest_component_removes_noise(self, binary_test_mask: np.ndarray) -> None:
        cleaned = filter_largest_component(binary_test_mask)
        assert cleaned.shape == binary_test_mask.shape
        assert cleaned[5, 5] == 0          # Noise pixel removed
        assert cleaned[32, 32] == 1        # Main component preserved

    def test_filter_largest_component_empty_mask(self) -> None:
        empty = np.zeros((32, 32), dtype=np.uint8)
        cleaned = filter_largest_component(empty)
        assert (cleaned == 0).all()

    def test_smooth_mask(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        # Add a tiny hole in the center
        mask[25, 25] = 0
        smoothed = smooth_mask(mask, kernel_size=3, morph_close=True)
        assert smoothed[25, 25] == 1       # Hole filled

    def test_compute_mask_bbox(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:75] = 1
        ymin, xmin, ymax, xmax = compute_mask_bbox(mask)
        assert ymin == 20
        assert xmin == 30
        assert ymax == 59
        assert xmax == 74

    def test_compute_mask_bbox_empty(self) -> None:
        empty = np.zeros((50, 50), dtype=np.uint8)
        assert compute_mask_bbox(empty) == (0, 0, 0, 0)

    def test_soft_mask_from_binary_range_and_type(self, binary_test_mask: np.ndarray) -> None:
        soft = soft_mask_from_binary(binary_test_mask, blur_radius=3)
        assert soft.dtype == np.float32
        assert soft.shape == binary_test_mask.shape
        assert soft.min() >= 0.0
        assert soft.max() <= 1.0
        # Check transition values between 0.0 and 1.0 along the boundary
        boundary_vals = soft[(soft > 0.05) & (soft < 0.95)]
        assert len(boundary_vals) > 0


# ──────────────────────────────────────────────
# Visualization Tests
# ──────────────────────────────────────────────

class TestVisualization:
    def test_overlay_mask_output_properties(
        self, sample_object_image: np.ndarray, binary_test_mask: np.ndarray
    ) -> None:
        # Resize mask to image size for testing
        mask_resized = cv2.resize(binary_test_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask(sample_object_image, mask_resized, alpha=0.5, color=(0, 255, 0))

        assert overlay.shape == sample_object_image.shape
        assert overlay.dtype == np.uint8
        # Ensure base image wasn't mutated
        assert sample_object_image[0, 0, 0] == 240

    def test_overlay_mask_with_different_alphas(
        self, sample_object_image: np.ndarray
    ) -> None:
        mask = np.ones((128, 128), dtype=np.uint8)
        overlay_transparent = overlay_mask(sample_object_image, mask, alpha=0.0, draw_contour=False)
        np.testing.assert_array_equal(overlay_transparent, sample_object_image)

    def test_side_by_side_panel_dimensions(
        self, sample_object_image: np.ndarray, binary_test_mask: np.ndarray
    ) -> None:
        h, w = sample_object_image.shape[:2]
        mask = cv2.resize(binary_test_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask(sample_object_image, mask)
        masked = sample_object_image * mask[:, :, None]

        panel = create_side_by_side_panel(sample_object_image, mask, overlay, masked)
        assert panel.shape == (2 * h, 2 * w, 3)
        assert panel.dtype == np.uint8


# ──────────────────────────────────────────────
# Object Segmentation Pipeline Tests
# ──────────────────────────────────────────────

class TestObjectSegmentation:
    def test_segment_saliency_grabcut(self, sample_object_image: np.ndarray) -> None:
        res = segment_saliency_grabcut(sample_object_image, num_iterations=2)
        assert isinstance(res, SegmentationResult)
        assert res.binary_mask.shape == sample_object_image.shape[:2]
        assert set(np.unique(res.binary_mask)).issubset({0, 1})
        assert res.soft_mask.dtype == np.float32
        assert res.masked_image.shape == sample_object_image.shape
        assert res.foreground_ratio > 0.0
        assert res.bbox != (0, 0, 0, 0)
        assert res.method == "saliency_grabcut"

    def test_segment_object_from_path(self, sample_image_file: Path) -> None:
        res = segment_object(sample_image_file, method="saliency_grabcut")
        assert isinstance(res, SegmentationResult)
        assert res.binary_mask.shape == (128, 128)

    def test_segment_object_from_preprocessed_dict(self, sample_image_file: Path) -> None:
        prep = preprocess(sample_image_file, target_size=(128, 128))
        res = segment_object(prep, method="saliency_grabcut")
        assert isinstance(res, SegmentationResult)
        assert res.binary_mask.shape == (128, 128)
        assert res.masked_image.shape == (128, 128, 3)

    def test_segment_object_from_normalized_float_array(
        self, sample_object_image: np.ndarray
    ) -> None:
        float_img = (sample_object_image.astype(np.float32) / 255.0 - 0.5) / 0.5
        res = segment_object(float_img, method="saliency_grabcut")
        assert isinstance(res, SegmentationResult)
        assert res.binary_mask.shape == (128, 128)

    def test_segment_object_auto_fallback(self, sample_object_image: np.ndarray) -> None:
        res = segment_object(sample_object_image, method="auto")
        assert isinstance(res, SegmentationResult)
        assert res.binary_mask.shape == sample_object_image.shape[:2]

    def test_segment_object_invalid_method_raises(self, sample_object_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            segment_object(sample_object_image, method="non_existent_method")
