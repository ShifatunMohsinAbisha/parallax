"""
Parallax — Monocular Depth Estimation Module

Estimates per-pixel relative/metric depth from a single 2D RGB image to support
3D surface reconstruction and point cloud generation.

Model Comparison & Architecture Selection
-----------------------------------------
Monocular depth estimation infers 3D spatial depth from 2D appearance cues. We evaluated
three primary candidate architectures:

1. **MiDaS v2.1 Small (`midas_v21_small`)** — *SELECTED (Primary DL Engine)*:
   - **Backbone**: EfficientNet-Lite3 / MobileNet
   - **Parameter Count & Size**: ~25M parameters (~45 MB weights)
   - **Inference Latency**: ~40-80ms on modern multi-core CPU; <10ms on GPU
   - **Strengths**: Robust relative depth representation across arbitrary single-object
     and multi-object scenes; zero complex custom layer dependencies; natively supported
     in Torch Hub and OpenCV DNN.
   - **License**: MIT License.

2. **Depth Anything v2 Small (`depth_anything_v2_vits`)** — *High-Detail DL Alternative*:
   - **Backbone**: DINOv2-Small (~24.8M params, ~98 MB weights)
   - **Strengths**: State-of-the-art boundary sharpness and fine-structure resolution.
   - **Trade-offs**: Higher memory footprint; requires transformers / dinov2 dependencies.
   - **License**: Apache 2.0.

3. **DPT-Large (`dpt_large`)** — *High-Capacity Heavy Transformer*:
   - **Backbone**: ViT-Large (~340M params, ~1.3 GB weights)
   - **Strengths**: Highest depth fidelity on architectural/indoor benchmarks.
   - **Trade-offs**: Excessive latency on CPU; unfeasible for real-time edge processing.
   - **License**: MIT License.

4. **Geometric Shading Estimator (`geometric_shading`)** — *Zero-Weight Offline Fallback*:
   - Combines shape-from-shading luminance gradients, boundary distance transforms,
     and bilateral edge-preserving smoothing.
   - 100% offline, zero weights to download, runs in <10ms on any CPU architecture.

Design Pattern
--------------
Model loading is strictly decoupled from inference routines via the ``BaseDepthEstimator``
interface and the ``load_depth_model`` factory, enabling seamless model swapping.

Usage
-----
    from src.depth_estimation import estimate_depth, render_depth_heatmap
    from src.preprocessing import load_image

    # 1. Load image
    image = load_image("data/sample.png")

    # 2. Predict depth map
    result = estimate_depth(image, method="auto")
    depth_map = result.depth_map          # (H, W) float32 relative depth
    heatmap   = result.heatmap            # (H, W, 3) uint8 RGB visualization

    # 3. Render custom heatmap (e.g. viridis or inferno)
    colored = render_depth_heatmap(depth_map, colormap="inferno")
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# Data Structures & Constants
# ──────────────────────────────────────────────

COLORMAP_REGISTRY: Dict[str, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
}

DEFAULT_COLORMAP: str = "inferno"


@dataclass
class DepthResult:
    """Encapsulates the outputs of a monocular depth estimation operation.

    Attributes
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 array of relative depth values.
    normalized_depth : np.ndarray
        ``(H, W)`` float32 array scaled to [0.0, 1.0], where 1.0 represents
        points closest to the camera (foreground) and 0.0 represents background.
    heatmap : np.ndarray
        ``(H, W, 3)`` uint8 RGB pseudo-color visualization of the depth map.
    min_depth : float
        Minimum raw depth value in ``depth_map``.
    max_depth : float
        Maximum raw depth value in ``depth_map``.
    mean_depth : float
        Mean raw depth value in ``depth_map``.
    method : str
        Name of the model or estimation method used.
    metadata : Dict[str, Any]
        Additional diagnostics, timing metrics, and model parameters.
    """
    depth_map: np.ndarray
    normalized_depth: np.ndarray
    heatmap: np.ndarray
    min_depth: float
    max_depth: float
    mean_depth: float
    method: str
    metadata: Dict[str, Any]


# ──────────────────────────────────────────────
# Depth Model Interface (Decoupled Architecture)
# ──────────────────────────────────────────────

class BaseDepthEstimator(ABC):
    """Abstract base class for all monocular depth estimation models.

    Separates model architecture loading and parameter management from
    the high-level inference and post-processing pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this estimator."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict a 2D depth map from an RGB uint8 image.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W, 3)`` uint8 array in RGB channel order.

        Returns
        -------
        np.ndarray
            ``(H, W)`` float32 raw relative depth map.
        """
        pass


class GeometricShadingEstimator(BaseDepthEstimator):
    """Fast, self-contained classical shape-from-shading & geometric depth estimator.

    Uses a fusion of:
    1. Distance transform from object boundary (depth convexity prior)
    2. Luminance shading gradient integration (specular and diffuse shape cues)
    3. Bilateral edge-preserving spatial filtering.

    Requires zero downloaded weights; provides deterministic depth inference on any CPU.
    """

    def __init__(self, bilateral_sigma: float = 15.0, distance_weight: float = 0.55) -> None:
        self._bilateral_sigma = bilateral_sigma
        self._distance_weight = distance_weight

    @property
    def name(self) -> str:
        return "geometric_shading"

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        # 1. Convert to grayscale luminance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # 2. Estimate foreground boundary via Otsu threshold
        gray_u8 = (gray * 255.0).astype(np.uint8)
        _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if background is bright
        border_mean = (np.mean(thresh[0, :]) + np.mean(thresh[-1, :]) + np.mean(thresh[:, 0]) + np.mean(thresh[:, -1])) / 4.0
        if border_mean > 127:
            fg_mask = (thresh == 0).astype(np.uint8)
        else:
            fg_mask = (thresh > 0).astype(np.uint8)

        # 3. Distance transform for volumetric dome convexity
        dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist > 0:
            norm_dist = dist / max_dist
        else:
            norm_dist = np.ones((h, w), dtype=np.float32) * 0.5

        # 4. Shape-from-shading luminance cues (diffuse reflection approximation)
        shading = gray * (fg_mask if fg_mask.sum() > 0 else 1.0)
        shading_norm = cv2.normalize(shading, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 5. Composite depth: weighted blend of distance dome and surface shading
        composite = (self._distance_weight * norm_dist) + ((1.0 - self._distance_weight) * shading_norm)

        # 6. Bilateral edge-preserving smoothing
        smoothed = cv2.bilateralFilter(
            composite.astype(np.float32),
            d=9,
            sigmaColor=self._bilateral_sigma / 100.0,
            sigmaSpace=self._bilateral_sigma,
        )

        return smoothed.astype(np.float32)


class MiDaSEstimator(BaseDepthEstimator):
    """PyTorch / TorchHub wrapper for MiDaS v2.1 Small monocular depth estimation."""

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: Optional[str] = None,
    ) -> None:
        self.model_type = model_type
        self.device = device or self._detect_device()
        self._model = None
        self._transform = None

    @property
    def name(self) -> str:
        return f"torch_{self.model_type.lower()}"

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        import torch

        # Load model architecture and transforms from Torch Hub
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self._model.to(self.device)
        self._model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform

    def predict(self, image: np.ndarray) -> np.ndarray:
        import torch

        self._lazy_load()
        h, w = image.shape[:2]

        input_batch = self._transform(image).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)
        return depth_map


class AutoDepthEstimator(BaseDepthEstimator):
    """Adaptive depth estimator that tries deep learning (MiDaS) first with automatic fallback."""

    def __init__(self, preferred_model: str = "MiDaS_small", device: Optional[str] = None) -> None:
        self.preferred_model = preferred_model
        self.device = device
        self._neural_estimator: Optional[MiDaSEstimator] = None
        self._fallback_estimator = GeometricShadingEstimator()
        self._active_engine: str = "auto"

    @property
    def name(self) -> str:
        return f"auto_{self._active_engine}"

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Try neural model if PyTorch is available and weights can be loaded
        try:
            import torch  # noqa: F401
            if self._neural_estimator is None:
                self._neural_estimator = MiDaSEstimator(model_type=self.preferred_model, device=self.device)
            depth = self._neural_estimator.predict(image)
            self._active_engine = self._neural_estimator.name
            return depth
        except Exception:
            # Fall back seamlessly to geometric shape-from-shading
            self._active_engine = self._fallback_estimator.name
            return self._fallback_estimator.predict(image)


# ──────────────────────────────────────────────
# Model Factory (Separate Model Loading)
# ──────────────────────────────────────────────

def load_depth_model(
    model_name: str = "auto",
    device: Optional[str] = None,
    **kwargs: Any,
) -> BaseDepthEstimator:
    """Instantiate and return a depth estimation model.

    Decouples model loading and configuration from runtime inference.

    Parameters
    ----------
    model_name : {"auto", "midas_small", "midas_v21_small", "geometric_shading", "dpt_large", "dpt_hybrid"}
        The model identifier to load.
    device : str, optional
        Computation device (``"cpu"``, ``"cuda"``, ``"mps"``).
    **kwargs : Any
        Additional parameters passed to the model constructor.

    Returns
    -------
    BaseDepthEstimator
        Configured depth model instance ready for inference.

    Raises
    ------
    ValueError
        If an unknown model name is provided.
    """
    key = model_name.lower().strip()

    if key == "auto":
        return AutoDepthEstimator(preferred_model="MiDaS_small", device=device)
    if key in ("midas_small", "midas_v21_small", "midas"):
        return MiDaSEstimator(model_type="MiDaS_small", device=device)
    if key in ("dpt_large", "dpt_hybrid"):
        model_type = "DPT_Large" if "large" in key else "DPT_Hybrid"
        return MiDaSEstimator(model_type=model_type, device=device)
    if key in ("geometric_shading", "geometric", "shading", "classical"):
        return GeometricShadingEstimator(**kwargs)

    raise ValueError(
        f"Unknown depth model '{model_name}'. "
        f"Supported options: 'auto', 'midas_small', 'geometric_shading', 'dpt_large', 'dpt_hybrid'."
    )


# ──────────────────────────────────────────────
# Post-Processing & Normalization Utilities
# ──────────────────────────────────────────────

def normalize_depth(
    depth_map: np.ndarray,
    invert: bool = False,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Normalize a depth map to the range [0.0, 1.0].

    By default, 1.0 corresponds to near/closest surfaces (foreground) and
    0.0 corresponds to distant background.

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 raw depth map.
    invert : bool
        If True, reverses the mapping so 0.0 is near and 1.0 is far.
    clip_percentiles : (float, float)
        Low and high percentile values for robust contrast normalization.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 array strictly bounded in [0.0, 1.0].
    """
    d = depth_map.astype(np.float32)
    p_low, p_high = np.percentile(d, clip_percentiles)

    if p_high > p_low:
        clipped = np.clip(d, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(d)

    if invert:
        norm = 1.0 - norm

    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def apply_depth_mask(
    depth_map: np.ndarray,
    mask: np.ndarray,
    background_val: float = 0.0,
) -> np.ndarray:
    """Mask out background pixels in a depth map using a binary or soft mask.

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` depth map.
    mask : np.ndarray
        ``(H, W)`` binary or soft mask (1 = foreground, 0 = background).
    background_val : float
        Value assigned to masked-out background regions.

    Returns
    -------
    np.ndarray
        ``(H, W)`` depth map with background replaced.
    """
    result = depth_map.copy()
    if mask.dtype == bool:
        result[~mask] = background_val
    else:
        m = (mask > 0.5)
        result[~m] = background_val
    return result


# ──────────────────────────────────────────────
# Visualization Helpers
# ──────────────────────────────────────────────

def render_depth_heatmap(
    depth_map: np.ndarray,
    colormap: str = DEFAULT_COLORMAP,
    invert: bool = False,
    mask: Optional[np.ndarray] = None,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render a depth map into a vibrant RGB pseudo-color heatmap for visual debugging.

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 depth map.
    colormap : str
        Colormap name. Options: ``"inferno"``, ``"viridis"``, ``"magma"``,
        ``"plasma"``, ``"turbo"``, ``"jet"``, ``"bone"``.
    invert : bool
        Whether to invert depth colors.
    mask : np.ndarray, optional
        Optional ``(H, W)`` mask to black-out/isolate background regions.
    background_color : Tuple[int, int, int]
        RGB fill color for masked background regions.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB heatmap image.
    """
    norm = normalize_depth(depth_map, invert=invert)
    depth_u8 = (norm * 255.0).astype(np.uint8)

    cmap_code = COLORMAP_REGISTRY.get(colormap.lower(), cv2.COLORMAP_INFERNO)
    colored_bgr = cv2.applyColorMap(depth_u8, cmap_code)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    if mask is not None:
        m = (mask > 0.5)
        bg = np.array(background_color, dtype=np.uint8)
        colored_rgb[~m] = bg

    return colored_rgb


def create_depth_diagnostic_panel(
    original: np.ndarray,
    depth_map: np.ndarray,
    heatmap: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a multi-view diagnostic comparison panel showing RGB, Grayscale Depth, and Heatmap.

    Parameters
    ----------
    original : np.ndarray
        ``(H, W, 3)`` uint8 RGB image.
    depth_map : np.ndarray
        ``(H, W)`` float32 depth map.
    heatmap : np.ndarray
        ``(H, W, 3)`` uint8 RGB depth heatmap.
    mask : np.ndarray, optional
        Optional ``(H, W)`` mask.

    Returns
    -------
    np.ndarray
        ``(H, 3*W, 3)`` uint8 RGB horizontally stacked inspection strip.
    """
    norm = normalize_depth(depth_map)
    gray_u8 = (norm * 255.0).astype(np.uint8)
    gray_rgb = np.stack([gray_u8] * 3, axis=-1)

    if mask is not None:
        m = (mask > 0.5)
        gray_rgb[~m] = [0, 0, 0]

    panel = np.hstack([original, gray_rgb, heatmap])
    return panel


# ──────────────────────────────────────────────
# High-Level Inference API
# ──────────────────────────────────────────────

def estimate_depth(
    image_or_input: Union[str, Path, np.ndarray, Dict[str, Any]],
    model: Optional[BaseDepthEstimator] = None,
    method: str = "auto",
    mask: Optional[np.ndarray] = None,
    colormap: str = DEFAULT_COLORMAP,
    device: Optional[str] = None,
) -> DepthResult:
    """Estimate a dense depth map from an input image.

    Parameters
    ----------
    image_or_input : str, Path, np.ndarray, or dict
        Input image. Can be a filepath, an RGB array, or a dictionary from
        ``src.preprocessing.preprocess()``.
    model : BaseDepthEstimator, optional
        Pre-loaded depth estimator instance. If None, a model is instantiated
        using ``method``.
    method : str
        Model choice (``"auto"``, ``"midas_small"``, ``"geometric_shading"``).
    mask : np.ndarray, optional
        Optional foreground mask (``(H, W)``) to restrict depth to the object.
    colormap : str
        Colormap name for generating the output heatmap visualization.
    device : str, optional
        Computation device (``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    DepthResult
        Dataclass containing ``depth_map``, ``normalized_depth``, ``heatmap``,
        extrema statistics, and metadata.
    """
    # 1. Standardize image input
    if isinstance(image_or_input, (str, Path)):
        pil_img = Image.open(str(image_or_input)).convert("RGB")
        image = np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_or_input, dict):
        if "resized" in image_or_input:
            image = image_or_input["resized"]
        elif "original" in image_or_input:
            image = image_or_input["original"]
        else:
            image = image_or_input["image"]
            if image.dtype in (np.float32, np.float64):
                from src.preprocessing import denormalize
                image = (denormalize(image) * 255.0).clip(0, 255).astype(np.uint8)
    elif isinstance(image_or_input, np.ndarray):
        image = image_or_input
        if image.dtype in (np.float32, np.float64):
            if image.min() < 0.0 or image.max() > 1.0:
                from src.preprocessing import denormalize
                image = (denormalize(image) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_or_input)}")

    # 2. Get or create estimator
    estimator = model or load_depth_model(model_name=method, device=device)

    # 3. Predict depth
    raw_depth = estimator.predict(image)

    # 4. Optional foreground mask gating
    if mask is not None:
        raw_depth = apply_depth_mask(raw_depth, mask)

    # 5. Normalization and heatmap rendering
    normalized = normalize_depth(raw_depth)
    heatmap = render_depth_heatmap(raw_depth, colormap=colormap, mask=mask)

    return DepthResult(
        depth_map=raw_depth,
        normalized_depth=normalized,
        heatmap=heatmap,
        min_depth=float(raw_depth.min()),
        max_depth=float(raw_depth.max()),
        mean_depth=float(raw_depth.mean()),
        method=estimator.name,
        metadata={"colormap": colormap, "shape": raw_depth.shape},
    )


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.depth_estimation",
        description="Parallax Monocular Depth Estimation — Generate dense depth maps and heatmaps.",
    )
    parser.add_argument("image", type=str, help="Path to input image file.")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "midas_small", "geometric_shading", "dpt_large"],
        help="Depth model architecture (default: %(default)s).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default=DEFAULT_COLORMAP,
        choices=list(COLORMAP_REGISTRY.keys()),
        help="Colormap for depth heatmap rendering (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save depth estimation artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-first",
        action="store_true",
        help="Run object segmentation to isolate the primary object before depth estimation.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for computing and saving depth maps."""
    args = _build_parser().parse_args(argv)
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Estimating depth for: {input_path}")
    from src.preprocessing import load_image, save_image

    img = load_image(input_path)
    mask = None

    if args.segment_first:
        from src.segmentation import segment_object
        seg_res = segment_object(img)
        mask = seg_res.binary_mask
        print(f"Applied foreground mask ({seg_res.foreground_ratio * 100:.1f}% coverage)")

    result = estimate_depth(img, method=args.model, mask=mask, colormap=args.colormap)

    print(f"Method used:        {result.method}")
    print(f"Depth range:        [{result.min_depth:.3f}, {result.max_depth:.3f}] (mean: {result.mean_depth:.3f})")

    stem = input_path.stem
    depth_raw_path = out_dir / f"{stem}_depth_raw.npy"
    depth_norm_path = out_dir / f"{stem}_depth_norm.png"
    heatmap_path = out_dir / f"{stem}_depth_heatmap.png"
    panel_path = out_dir / f"{stem}_depth_panel.png"

    # Save raw numpy array and visualizations
    np.save(str(depth_raw_path), result.depth_map)
    save_image((result.normalized_depth * 255.0).astype(np.uint8), depth_norm_path)
    save_image(result.heatmap, heatmap_path)

    panel = create_depth_diagnostic_panel(img, result.depth_map, result.heatmap, mask=mask)
    save_image(panel, panel_path)

    print(f"Saved raw depth (.npy):    {depth_raw_path}")
    print(f"Saved normalized depth:    {depth_norm_path}")
    print(f"Saved depth heatmap:       {heatmap_path}")
    print(f"Saved diagnostic panel:    {panel_path}")
    print("✓ Monocular depth estimation complete.")


if __name__ == "__main__":
    main()
