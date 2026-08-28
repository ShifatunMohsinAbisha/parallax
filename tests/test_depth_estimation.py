"""Unit tests for src.depth_estimation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.depth_estimation import (
    COLORMAP_REGISTRY,
    AutoDepthEstimator,
    BaseDepthEstimator,
    DepthResult,
    GeometricShadingEstimator,
    apply_depth_mask,
    create_depth_diagnostic_panel,
    estimate_depth,
    load_depth_model,
    normalize_depth,
    render_depth_heatmap,
)
from src.preprocessing import preprocess


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def sample_sphere_image() -> np.ndarray:
    """Create a 128×128 synthetic image with a shaded sphere on a dark background."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Shaded sphere in center
    cv2.circle(img, (64, 64), 40, (180, 120, 60), -1)
    cv2.circle(img, (54, 54), 20, (230, 180, 120), -1)
    cv2.circle(img, (50, 50), 8, (255, 240, 200), -1)
    return img


@pytest.fixture()
def sample_image_path(sample_sphere_image: np.ndarray, tmp_path: Path) -> Path:
    """Save the synthetic sphere image to a temporary file."""
    p = tmp_path / "test_sphere.png"
    Image.fromarray(sample_sphere_image, mode="RGB").save(str(p))
    return p


@pytest.fixture()
def sample_depth_map() -> np.ndarray:
    """Create a gradient depth map."""
    y, x = np.mgrid[:100, :100]
    return (x + y).astype(np.float32)


# ──────────────────────────────────────────────
# Model Factory & Interface Tests
# ──────────────────────────────────────────────

class TestDepthModelLoading:
    def test_load_geometric_model(self) -> None:
        model = load_depth_model("geometric_shading")
        assert isinstance(model, BaseDepthEstimator)
        assert isinstance(model, GeometricShadingEstimator)
        assert model.name == "geometric_shading"

    def test_load_auto_model(self) -> None:
        model = load_depth_model("auto")
        assert isinstance(model, BaseDepthEstimator)
        assert isinstance(model, AutoDepthEstimator)

    def test_load_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown depth model"):
            load_depth_model("unsupported_super_depth")


# ──────────────────────────────────────────────
# Geometric Depth Estimator Tests
# ──────────────────────────────────────────────

class TestGeometricEstimator:
    def test_prediction_shape_and_dtype(self, sample_sphere_image: np.ndarray) -> None:
        estimator = GeometricShadingEstimator()
        depth = estimator.predict(sample_sphere_image)

        assert isinstance(depth, np.ndarray)
        assert depth.shape == sample_sphere_image.shape[:2]
        assert depth.dtype == np.float32

    def test_sphere_center_has_higher_depth_than_border(
        self, sample_sphere_image: np.ndarray
    ) -> None:
        estimator = GeometricShadingEstimator()
        depth = estimator.predict(sample_sphere_image)

        center_depth = depth[64, 64]
        corner_depth = depth[5, 5]
        # Convex object center should be closer / higher depth value than background corner
        assert center_depth > corner_depth


# ──────────────────────────────────────────────
# Normalization & Masking Tests
# ──────────────────────────────────────────────

class TestNormalizationAndMasking:
    def test_normalize_depth_bounds(self, sample_depth_map: np.ndarray) -> None:
        norm = normalize_depth(sample_depth_map)
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_normalize_depth_invert(self, sample_depth_map: np.ndarray) -> None:
        norm_std = normalize_depth(sample_depth_map, invert=False)
        norm_inv = normalize_depth(sample_depth_map, invert=True)

        np.testing.assert_allclose(norm_std + norm_inv, 1.0, atol=1e-5)

    def test_normalize_uniform_depth(self) -> None:
        uniform = np.full((50, 50), 42.0, dtype=np.float32)
        norm = normalize_depth(uniform)
        assert (norm == 0.0).all()

    def test_apply_depth_mask(self, sample_depth_map: np.ndarray) -> None:
        mask = np.zeros_like(sample_depth_map, dtype=np.uint8)
        mask[20:80, 20:80] = 1

        masked_depth = apply_depth_mask(sample_depth_map, mask, background_val=-1.0)
        assert masked_depth[0, 0] == -1.0
        assert masked_depth[50, 50] == sample_depth_map[50, 50]


# ──────────────────────────────────────────────
# Visualization Tests
# ──────────────────────────────────────────────

class TestVisualization:
    @pytest.mark.parametrize("cmap", list(COLORMAP_REGISTRY.keys()))
    def test_render_depth_heatmap_colormaps(
        self, sample_depth_map: np.ndarray, cmap: str
    ) -> None:
        heatmap = render_depth_heatmap(sample_depth_map, colormap=cmap)
        assert heatmap.shape == (100, 100, 3)
        assert heatmap.dtype == np.uint8

    def test_render_depth_heatmap_with_mask(self, sample_depth_map: np.ndarray) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        heatmap = render_depth_heatmap(
            sample_depth_map,
            colormap="inferno",
            mask=mask,
            background_color=(10, 20, 30),
        )

        np.testing.assert_array_equal(heatmap[0, 0], [10, 20, 30])

    def test_create_depth_diagnostic_panel(
        self, sample_sphere_image: np.ndarray, sample_depth_map: np.ndarray
    ) -> None:
        # Match shapes
        h, w = sample_sphere_image.shape[:2]
        d = cv2.resize(sample_depth_map, (w, h))
        hm = render_depth_heatmap(d)

        panel = create_depth_diagnostic_panel(sample_sphere_image, d, hm)
        assert panel.shape == (h, 3 * w, 3)
        assert panel.dtype == np.uint8


# ──────────────────────────────────────────────
# High-Level Pipeline Tests
# ──────────────────────────────────────────────

class TestEstimateDepthPipeline:
    def test_estimate_depth_from_array(self, sample_sphere_image: np.ndarray) -> None:
        res = estimate_depth(sample_sphere_image, method="geometric_shading")

        assert isinstance(res, DepthResult)
        assert res.depth_map.shape == (128, 128)
        assert res.normalized_depth.shape == (128, 128)
        assert res.heatmap.shape == (128, 128, 3)
        assert res.min_depth <= res.max_depth
        assert res.method == "geometric_shading"

    def test_estimate_depth_from_filepath(self, sample_image_path: Path) -> None:
        res = estimate_depth(sample_image_path, method="geometric_shading")

        assert isinstance(res, DepthResult)
        assert res.depth_map.shape == (128, 128)

    def test_estimate_depth_from_preprocessed_dict(self, sample_image_path: Path) -> None:
        prep = preprocess(sample_image_path, target_size=(128, 128))
        res = estimate_depth(prep, method="geometric_shading")

        assert isinstance(res, DepthResult)
        assert res.depth_map.shape == (128, 128)

    def test_estimate_depth_with_mask(self, sample_sphere_image: np.ndarray) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 30, 1, -1)

        res = estimate_depth(sample_sphere_image, method="geometric_shading", mask=mask)
        assert res.depth_map[0, 0] == 0.0
        assert res.depth_map[64, 64] > 0.0
