"""
Parallax — Image Preprocessing Module

Provides utilities for loading, resizing, padding, and normalizing images
for downstream ML pipelines (depth estimation, 3D reconstruction, etc.).

All functions operate on NumPy arrays in HWC / RGB format unless noted
otherwise.  The ``preprocess`` function chains the full pipeline and
returns a metadata dict so downstream modules can map predictions back
to original image coordinates.

The module uses **Pillow** for core image I/O and resizing, so there is
no hard dependency on OpenCV.  OpenCV (``cv2``) is imported lazily only
where it provides a clear advantage (e.g. advanced interpolation in
background estimation).  All public functions work with just NumPy +
Pillow installed.

Usage
-----
    from src.preprocessing import preprocess, DEFAULT_INPUT_SIZE

    result = preprocess("photo.jpg", target_size=DEFAULT_INPUT_SIZE)
    tensor_ready = result["image"]       # (H, W, 3) float32, normalized
    meta         = result["metadata"]    # scale, padding offsets, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# Constants & presets
# ──────────────────────────────────────────────

#: Default input size (H, W) used throughout Parallax.
DEFAULT_INPUT_SIZE: Tuple[int, int] = (256, 256)

#: ImageNet channel‑wise mean (RGB order).
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)

#: ImageNet channel‑wise standard deviation (RGB order).
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ──────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from *path* and return it as an RGB uint8 array.

    Parameters
    ----------
    path : str or Path
        Path to a JPEG, PNG, or other supported image file.

    Returns
    -------
    np.ndarray
        Image array with shape ``(H, W, 3)`` and dtype ``uint8``, in RGB
        channel order.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or the image cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        img = Image.open(str(path)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {path}") from exc

    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    """Save an RGB, RGBA, or grayscale uint8/float image to *path*.

    If *image* is float, it is clipped to [0, 1] and scaled to uint8
    automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype in (np.float32, np.float64):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode="L")
    elif image.ndim == 3 and image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode="RGBA")
    else:
        pil_img = Image.fromarray(image, mode="RGB")

    pil_img.save(str(path))
    return path


# ──────────────────────────────────────────────
# Resize & pad
# ──────────────────────────────────────────────

def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    resample: int = Image.BILINEAR,
) -> Tuple[np.ndarray, dict]:
    """Resize *image* to fit inside *target_size* and pad the remainder.

    The image is scaled so its largest dimension matches the corresponding
    target dimension, preserving the original aspect ratio.  The remaining
    space is filled with *pad_color*.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape ``(H, W, 3)``.
    target_size : (int, int)
        Desired output size as ``(height, width)``.
    pad_color : (int, int, int)
        RGB fill color for the padded region.
    resample : int
        PIL resampling filter (e.g. ``Image.BILINEAR``, ``Image.LANCZOS``).

    Returns
    -------
    resized : np.ndarray
        Resized and padded image with shape ``(target_h, target_w, 3)``.
    metadata : dict
        Transformation metadata with keys:

        - ``original_size`` – ``(H, W)`` of the input.
        - ``scale`` – float scale factor applied.
        - ``scaled_size`` – ``(H, W)`` after scaling, before padding.
        - ``pad_top``, ``pad_left`` – pixel offsets of the image within
          the padded canvas.
    """
    target_h, target_w = target_size
    src_h, src_w = image.shape[:2]

    # Scale factor: fit the image inside the target rectangle.
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    pil_img = Image.fromarray(image)
    scaled = pil_img.resize((new_w, new_h), resample=resample)

    # Center the scaled image on the canvas.
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
    canvas.paste(scaled, (pad_left, pad_top))

    metadata = {
        "original_size": (src_h, src_w),
        "scale": scale,
        "scaled_size": (new_h, new_w),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "target_size": (target_h, target_w),
    }
    return np.array(canvas, dtype=np.uint8), metadata


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

def normalize(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Normalize a uint8 or [0, 1] float image with channel‑wise *mean* and *std*.

    The image is first converted to float32 in [0, 1], then the standard
    ``(x - mean) / std`` normalization is applied per channel.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` image, uint8 or float32.
    mean, std : array‑like
        Per‑channel mean and standard deviation (length 3).

    Returns
    -------
    np.ndarray
        Normalized float32 image, same spatial shape.
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return (img - mean) / std


def denormalize(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Reverse ``normalize`` — return a [0, 1] float32 image."""
    return np.clip(image * std + mean, 0.0, 1.0).astype(np.float32)


def to_tensor_format(image: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, C)`` image to ``(C, H, W)`` for PyTorch."""
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def from_tensor_format(tensor: np.ndarray) -> np.ndarray:
    """Convert a ``(C, H, W)`` tensor back to ``(H, W, C)``."""
    return np.ascontiguousarray(tensor.transpose(1, 2, 0))


# ──────────────────────────────────────────────
# Background handling (placeholder stubs)
# ──────────────────────────────────────────────

def estimate_background_mask(
    image: np.ndarray,
    method: str = "threshold",
    *,
    threshold: int = 15,
) -> np.ndarray:
    """Return a binary mask where **background = 1** and **foreground = 0**.

    This is a *placeholder* implementation using simple heuristics.
    It will be replaced by a learned segmentation model in a later step.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` RGB uint8 image.
    method : str
        ``"threshold"`` — mark near‑black / near‑white pixels as background.
    threshold : int
        Intensity threshold used by the ``"threshold"`` method.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(H, W)``.
    """
    if method == "threshold":
        # Convert to grayscale via luminance weights.
        gray = np.dot(image[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)
        is_dark = gray < threshold
        is_bright = gray > (255 - threshold)
        return (is_dark | is_bright).astype(np.uint8)
    if method in ("segmentation", "grabcut", "auto"):
        from src.segmentation import segment_object
        seg_res = segment_object(image, method="saliency_grabcut" if method == "grabcut" else "auto")
        # Invert foreground binary mask (1 for background, 0 for foreground)
        return (1 - seg_res.binary_mask).astype(np.uint8)
    raise ValueError(f"Unknown background method: {method!r}")


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Replace masked (``mask == 1``) pixels with *fill_color*.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` image.
    mask : np.ndarray
        ``(H, W)`` binary mask — 1 = replace, 0 = keep.
    fill_color : tuple
        RGB color to fill masked pixels with.

    Returns
    -------
    np.ndarray
        Copy of *image* with masked pixels replaced.
    """
    result = image.copy()
    result[mask.astype(bool)] = fill_color
    return result


# ──────────────────────────────────────────────
# Full preprocessing pipeline
# ──────────────────────────────────────────────

def preprocess(
    path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    remove_background: bool = False,
) -> dict:
    """Run the full preprocessing pipeline on a single image.

    Steps
    -----
    1. Load the image from disk.
    2. (Optional) Estimate and remove background.
    3. Resize + pad to *target_size*, preserving aspect ratio.
    4. Normalize with *mean* / *std*.

    Parameters
    ----------
    path : str or Path
        Image file path.
    target_size : (int, int)
        Target ``(H, W)`` after resize + pad.
    pad_color : (int, int, int)
        Padding fill color.
    mean, std : array‑like
        Normalization statistics.
    remove_background : bool
        If ``True``, apply the placeholder background removal before
        resizing.

    Returns
    -------
    dict
        ``image``      — preprocessed ``(H, W, 3)`` float32 array.
        ``image_chw``  — same data in ``(C, H, W)`` tensor layout.
        ``original``   — original loaded RGB uint8 image.
        ``resized``    — resized uint8 image before normalization.
        ``metadata``   — resize / padding metadata dict.
    """
    original = load_image(path)
    working = original.copy()

    effective_pad_color = pad_color
    if pad_color == (0, 0, 0):
        # Auto-detect background color from image boundary if background is non-black
        border_pixels = np.concatenate([original[0, :], original[-1, :], original[:, 0], original[:, -1]], axis=0)
        auto_bg = tuple(int(round(x)) for x in np.median(border_pixels, axis=0))
        if np.mean(auto_bg) > 30:
            effective_pad_color = auto_bg

    if remove_background:
        bg_mask = estimate_background_mask(working)
        working = apply_mask(working, bg_mask, fill_color=effective_pad_color)

    resized, metadata = resize_with_padding(working, target_size, effective_pad_color)
    normalized = normalize(resized, mean, std)

    return {
        "image": normalized,
        "image_chw": to_tensor_format(normalized),
        "original": original,
        "resized": resized,
        "metadata": metadata,
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.preprocessing",
        description="Parallax image preprocessing — quick sanity check.",
    )
    parser.add_argument("image", type=str, help="Path to an input image.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=list(DEFAULT_INPUT_SIZE),
        metavar=("H", "W"),
        help="Target size (height width). Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the preprocessed image.",
    )
    parser.add_argument(
        "--remove-bg",
        action="store_true",
        help="Apply placeholder background removal.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for quick preprocessing tests."""
    args = _build_parser().parse_args(argv)
    target_size = tuple(args.size)

    print(f"Loading image:  {args.image}")
    result = preprocess(
        args.image,
        target_size=target_size,
        remove_background=args.remove_bg,
    )

    meta = result["metadata"]
    print(f"Original size:  {meta['original_size'][1]}×{meta['original_size'][0]}")
    print(f"Scale factor:   {meta['scale']:.4f}")
    print(f"Scaled size:    {meta['scaled_size'][1]}×{meta['scaled_size'][0]}")
    print(f"Padding (t, l): ({meta['pad_top']}, {meta['pad_left']})")
    print(f"Output shape:   {result['image'].shape}")
    print(f"Value range:    [{result['image'].min():.3f}, {result['image'].max():.3f}]")

    if args.output:
        vis = denormalize(result["image"])
        out_path = save_image(vis, args.output)
        print(f"Saved to:       {out_path}")

    print("✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
