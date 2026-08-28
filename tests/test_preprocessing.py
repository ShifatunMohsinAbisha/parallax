"""Unit tests for src.preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    DEFAULT_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_mask,
    denormalize,
    estimate_background_mask,
    from_tensor_format,
    load_image,
    normalize,
    preprocess,
    resize_with_padding,
    save_image,
    to_tensor_format,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def sample_rgb() -> np.ndarray:
    """A small 60×100 RGB uint8 image with a mid‑gray foreground."""
    return np.full((60, 100, 3), 128, dtype=np.uint8)


@pytest.fixture()
def landscape_image() -> np.ndarray:
    """A 200×400 landscape‑oriented test image."""
    rng = np.random.default_rng(42)
    return rng.integers(30, 225, size=(200, 400, 3), dtype=np.uint8)


@pytest.fixture()
def portrait_image() -> np.ndarray:
    """A 400×200 portrait‑oriented test image."""
    rng = np.random.default_rng(99)
    return rng.integers(30, 225, size=(400, 200, 3), dtype=np.uint8)


@pytest.fixture()
def tmp_image_path(sample_rgb: np.ndarray, tmp_path: Path) -> Path:
    """Write *sample_rgb* to a temporary PNG and return the path."""
    p = tmp_path / "test_input.png"
    Image.fromarray(sample_rgb, mode="RGB").save(str(p))
    return p


# ──────────────────────────────────────────────
# load_image
# ──────────────────────────────────────────────

class TestLoadImage:
    def test_loads_png(self, tmp_image_path: Path) -> None:
        img = load_image(tmp_image_path)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    def test_returns_rgb(self, tmp_image_path: Path) -> None:
        """Verify the returned image is RGB."""
        # Write a known red pixel image.
        red_img = np.zeros((1, 1, 3), dtype=np.uint8)
        red_img[0, 0] = [255, 0, 0]  # Pure red in RGB
        p = tmp_image_path.parent / "red.png"
        Image.fromarray(red_img, mode="RGB").save(str(p))

        rgb = load_image(p)
        assert rgb[0, 0, 0] == 255  # R channel
        assert rgb[0, 0, 2] == 0    # B channel

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "no_such_file.png")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "image.gif"
        p.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(p)


# ──────────────────────────────────────────────
# save_image
# ──────────────────────────────────────────────

class TestSaveImage:
    def test_roundtrip_uint8(self, sample_rgb: np.ndarray, tmp_path: Path) -> None:
        out = tmp_path / "out.png"
        save_image(sample_rgb, out)
        reloaded = load_image(out)
        np.testing.assert_array_equal(reloaded, sample_rgb)

    def test_saves_float_image(self, tmp_path: Path) -> None:
        img_f = np.full((10, 10, 3), 0.5, dtype=np.float32)
        out = tmp_path / "float_out.png"
        save_image(img_f, out)
        assert out.exists()

    def test_creates_parent_dirs(self, sample_rgb: np.ndarray, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "deep.png"
        save_image(sample_rgb, out)
        assert out.exists()


# ──────────────────────────────────────────────
# resize_with_padding
# ──────────────────────────────────────────────

class TestResizeWithPadding:
    def test_output_shape_matches_target(self, sample_rgb: np.ndarray) -> None:
        target = (256, 256)
        resized, _ = resize_with_padding(sample_rgb, target_size=target)
        assert resized.shape[:2] == target

    def test_preserves_aspect_ratio(self, landscape_image: np.ndarray) -> None:
        """After resize, the scaled region should not be distorted."""
        target = (256, 256)
        _, meta = resize_with_padding(landscape_image, target_size=target)
        orig_h, orig_w = meta["original_size"]
        new_h, new_w = meta["scaled_size"]
        orig_ar = orig_w / orig_h
        new_ar = new_w / new_h
        assert abs(orig_ar - new_ar) < 0.02  # allow minor rounding

    def test_portrait_image_padding(self, portrait_image: np.ndarray) -> None:
        target = (256, 256)
        resized, meta = resize_with_padding(portrait_image, target_size=target)
        assert resized.shape[:2] == target
        # Portrait is taller than wide → should have left/right padding.
        assert meta["pad_left"] > 0
        assert meta["pad_top"] == 0

    def test_landscape_image_padding(self, landscape_image: np.ndarray) -> None:
        target = (256, 256)
        resized, meta = resize_with_padding(landscape_image, target_size=target)
        assert resized.shape[:2] == target
        # Landscape is wider than tall → should have top/bottom padding.
        assert meta["pad_top"] > 0
        assert meta["pad_left"] == 0

    def test_square_image_no_padding(self) -> None:
        square = np.full((100, 100, 3), 200, dtype=np.uint8)
        resized, meta = resize_with_padding(square, target_size=(256, 256))
        assert meta["pad_top"] == 0
        assert meta["pad_left"] == 0

    def test_metadata_keys(self, sample_rgb: np.ndarray) -> None:
        _, meta = resize_with_padding(sample_rgb)
        expected_keys = {
            "original_size", "scale", "scaled_size",
            "pad_top", "pad_left", "target_size",
        }
        assert set(meta.keys()) == expected_keys

    def test_pad_color(self, sample_rgb: np.ndarray) -> None:
        pad_color = (255, 0, 0)
        resized, meta = resize_with_padding(
            sample_rgb, target_size=(256, 256), pad_color=pad_color
        )
        # Top‑left corner should be padding if there is top padding.
        if meta["pad_top"] > 0:
            np.testing.assert_array_equal(resized[0, 0], pad_color)


# ──────────────────────────────────────────────
# normalize / denormalize
# ──────────────────────────────────────────────

class TestNormalization:
    def test_normalize_output_range(self, sample_rgb: np.ndarray) -> None:
        normed = normalize(sample_rgb)
        assert normed.dtype == np.float32
        # With ImageNet stats a uniform 128 image should be near zero.
        assert np.abs(normed.mean()) < 1.0

    def test_normalize_denormalize_roundtrip(self) -> None:
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        normed = normalize(img)
        recovered = denormalize(normed)
        expected = img.astype(np.float32) / 255.0
        np.testing.assert_allclose(recovered, expected, atol=1e-5)

    def test_normalize_already_float(self) -> None:
        img_f = np.full((10, 10, 3), 0.5, dtype=np.float32)
        normed = normalize(img_f)
        assert normed.dtype == np.float32


# ──────────────────────────────────────────────
# Tensor format conversions
# ──────────────────────────────────────────────

class TestTensorFormat:
    def test_to_tensor_shape(self) -> None:
        hwc = np.zeros((64, 128, 3), dtype=np.float32)
        chw = to_tensor_format(hwc)
        assert chw.shape == (3, 64, 128)

    def test_roundtrip(self) -> None:
        hwc = np.random.rand(64, 128, 3).astype(np.float32)
        recovered = from_tensor_format(to_tensor_format(hwc))
        np.testing.assert_array_equal(recovered, hwc)


# ──────────────────────────────────────────────
# Background utilities
# ──────────────────────────────────────────────

class TestBackgroundUtilities:
    def test_threshold_mask_shape(self, sample_rgb: np.ndarray) -> None:
        mask = estimate_background_mask(sample_rgb)
        assert mask.shape == sample_rgb.shape[:2]

    def test_dark_pixels_are_background(self) -> None:
        dark = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = estimate_background_mask(dark, threshold=15)
        assert mask.all()  # all dark → all background

    def test_bright_pixels_are_background(self) -> None:
        bright = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = estimate_background_mask(bright, threshold=15)
        assert mask.all()

    def test_midtone_pixels_are_foreground(self) -> None:
        mid = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = estimate_background_mask(mid, threshold=15)
        assert not mask.any()

    def test_unknown_method_raises(self, sample_rgb: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            estimate_background_mask(sample_rgb, method="magic")

    def test_apply_mask_replaces_pixels(self) -> None:
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:5, :] = 1  # top half is masked
        result = apply_mask(img, mask, fill_color=(255, 0, 0))

        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[9, 0], [100, 100, 100])

    def test_apply_mask_does_not_mutate_input(self) -> None:
        img = np.full((4, 4, 3), 50, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)
        _ = apply_mask(img, mask, fill_color=(0, 0, 0))
        assert (img == 50).all()


# ──────────────────────────────────────────────
# Full pipeline (preprocess)
# ──────────────────────────────────────────────

class TestPreprocess:
    def test_returns_expected_keys(self, tmp_image_path: Path) -> None:
        result = preprocess(tmp_image_path)
        assert set(result.keys()) == {
            "image", "image_chw", "original", "resized", "metadata",
        }

    def test_output_shape(self, tmp_image_path: Path) -> None:
        target = (128, 128)
        result = preprocess(tmp_image_path, target_size=target)
        assert result["image"].shape == (128, 128, 3)
        assert result["image_chw"].shape == (3, 128, 128)

    def test_original_is_unmodified(self, tmp_image_path: Path) -> None:
        result = preprocess(tmp_image_path)
        orig = load_image(tmp_image_path)
        np.testing.assert_array_equal(result["original"], orig)

    def test_with_background_removal(self, tmp_image_path: Path) -> None:
        # Should not crash.
        result = preprocess(tmp_image_path, remove_background=True)
        assert result["image"].shape[2] == 3

    def test_custom_normalization(self, tmp_image_path: Path) -> None:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = preprocess(tmp_image_path, mean=mean, std=std)
        meta = result["metadata"]
        # Extract just the content region (excluding padding).
        t, l = meta["pad_top"], meta["pad_left"]
        sh, sw = meta["scaled_size"]
        content = result["image"][t : t + sh, l : l + sw]
        # Uniform 128 image → (128/255 - 0.5) / 0.5 ≈ 0.004
        assert abs(content.mean()) < 0.1
