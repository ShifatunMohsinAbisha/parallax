"""
Parallax — Point Cloud Processing & Open3D Integration Module

Provides post-processing, outlier filtering, Open3D object conversion, and
multi-format export (.ply, .pcd) for 3D point clouds reconstructed from single images.

Open3D & Native Fallback Architecture
-------------------------------------
- **Open3D Backend**: When ``open3d`` is installed, uses native C++ accelerated
  ``open3d.geometry.PointCloud`` data structures, statistical outlier removal,
  and GUI/offscreen visualizers.
- **Native NumPy Engine**: When running in headless environments or Python versions
  where Open3D wheels are unavailable, provides pure NumPy implementations of
  statistical outlier removal, radius filtering, PCD v0.7 serialization, and 3D
  projection rendering.

PCD (Point Cloud Data) Format Support
-------------------------------------
Supports PCD v0.7 (Point Cloud Library format) with:
  - VERSION 0.7
  - FIELDS x y z rgb (or x y z r g b)
  - SIZE 4 4 4 4
  - TYPE F F F F (packed float/uint32 RGB)
  - DATA ascii / binary

Usage
-----
    from src.point_cloud import (
        create_point_cloud,
        clean_point_cloud,
        export_point_cloud_pcd,
        export_point_cloud_ply,
        render_point_cloud_views,
    )
    from src.geometry import depth_to_point_cloud

    # 1. Unproject geometry
    raw_pcd = depth_to_point_cloud(depth_map, intrinsics, rgb=image, mask=mask)

    # 2. Convert and clean (statistical outlier removal)
    cleaned_pcd, outlier_mask = clean_point_cloud(
        raw_pcd,
        nb_neighbors=20,
        std_ratio=2.0,
    )

    # 3. Export to PCD and PLY formats
    export_point_cloud_pcd(cleaned_pcd, "outputs/reconstruction.pcd")
    export_point_cloud_ply(cleaned_pcd, "outputs/reconstruction.ply")

    # 4. Render 3D isometric screenshot for headless inspection
    render_point_cloud_views(cleaned_pcd, "outputs/reconstruction_3d.png")
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.geometry import CameraIntrinsics, PointCloud

# ──────────────────────────────────────────────
# Open3D Lazy Import & Availability Detection
# ──────────────────────────────────────────────

try:
    import open3d as o3d
    HAS_OPEN3D: bool = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class PointCloudProcessingResult:
    """Encapsulates the result of point cloud processing and outlier filtering.

    Attributes
    ----------
    point_cloud : PointCloud
        Cleaned 3D point cloud.
    inlier_mask : np.ndarray
        ``(N,)`` boolean mask indicating which original points were retained.
    num_original : int
        Point count before filtering.
    num_retained : int
        Point count after filtering.
    outlier_ratio : float
        Fraction of points removed as noise (0.0 to 1.0).
    o3d_pcd : Optional[Any]
        Native ``open3d.geometry.PointCloud`` instance if Open3D is available.
    metadata : Dict[str, Any]
        Additional metrics, filter parameters, and bounding dimensions.
    """
    point_cloud: PointCloud
    inlier_mask: np.ndarray
    num_original: int
    num_retained: int
    outlier_ratio: float
    o3d_pcd: Optional[Any]
    metadata: Dict[str, Any]


# ──────────────────────────────────────────────
# Open3D Conversion Utilities
# ──────────────────────────────────────────────

def to_open3d_point_cloud(point_cloud: PointCloud) -> Any:
    """Convert a Parallax ``PointCloud`` into an ``open3d.geometry.PointCloud`` instance.

    Parameters
    ----------
    point_cloud : PointCloud
        Parallax point cloud dataclass.

    Returns
    -------
    open3d.geometry.PointCloud
        Populated Open3D point cloud with points, colors in [0, 1], and normals.

    Raises
    ------
    ImportError
        If ``open3d`` is not installed.
    """
    if not HAS_OPEN3D:
        raise ImportError(
            "Open3D is not installed in the current Python environment. "
            "Native Parallax NumPy point cloud routines are active."
        )

    pcd = o3d.geometry.PointCloud()
    if len(point_cloud.points) > 0:
        pcd.points = o3d.utility.Vector3dVector(point_cloud.points.astype(np.float64))

        if point_cloud.colors is not None and len(point_cloud.colors) == len(point_cloud.points):
            # Open3D expects colors normalized in range [0.0, 1.0]
            colors_norm = (point_cloud.colors.astype(np.float64) / 255.0).clip(0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors_norm)

        if point_cloud.normals is not None and len(point_cloud.normals) == len(point_cloud.points):
            pcd.normals = o3d.utility.Vector3dVector(point_cloud.normals.astype(np.float64))

    return pcd


def from_open3d_point_cloud(
    o3d_pcd: Any,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> PointCloud:
    """Convert an ``open3d.geometry.PointCloud`` instance back into a Parallax ``PointCloud``.

    Parameters
    ----------
    o3d_pcd : open3d.geometry.PointCloud
        Open3D point cloud instance.
    intrinsics : Optional[CameraIntrinsics]
        Camera intrinsic parameters.

    Returns
    -------
    PointCloud
        Parallax point cloud dataclass.
    """
    points = np.asarray(o3d_pcd.points, dtype=np.float32)

    colors = None
    if o3d_pcd.has_colors():
        colors_norm = np.asarray(o3d_pcd.colors, dtype=np.float32)
        colors = (colors_norm * 255.0).clip(0, 255).astype(np.uint8)

    normals = None
    if o3d_pcd.has_normals():
        normals = np.asarray(o3d_pcd.normals, dtype=np.float32)

    if intrinsics is None:
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=250.0, cy=250.0, width=500, height=500, fov_degrees=60.0
        )

    return PointCloud(
        points=points,
        colors=colors,
        normals=normals,
        intrinsics=intrinsics,
        metadata={"source": "open3d"},
    )


# ──────────────────────────────────────────────
# Point Cloud Filtering & Outlier Removal
# ──────────────────────────────────────────────

def remove_statistical_outliers_numpy(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    subsample_max: int = 15000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy implementation of Statistical Outlier Removal (SOR).

    Calculates the mean distance from each point to its k-nearest neighbors.
    Points with an average distance greater than (mean + std_ratio * std_dev)
    are classified as outliers and rejected.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` array of 3D spatial points.
    nb_neighbors : int
        Number of nearest neighbors to evaluate (k).
    std_ratio : float
        Standard deviation multiplier threshold.
    subsample_max : int
        Threshold for spatial chunking when point count is large.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(inlier_indices, inlier_mask)``
    """
    n_points = len(points)
    if n_points <= nb_neighbors:
        mask = np.ones(n_points, dtype=bool)
        return np.arange(n_points), mask

    # Compute k-nearest neighbor distances efficiently using spatial grid / chunking
    k = min(nb_neighbors, n_points - 1)
    mean_distances = np.zeros(n_points, dtype=np.float32)

    # Process in vectorized chunks to prevent high memory allocation
    chunk_size = 1024
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk = points[start_idx:end_idx]  # (C, 3)

        # Distance matrix from chunk to all points: (C, N)
        diff = chunk[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)  # (C, N)

        # Top k smallest distances (excluding distance to self which is 0)
        # partition finds the smallest k+1 elements without full sort
        part = np.partition(dists, kth=k + 1, axis=1)[:, 1 : k + 1]
        mean_distances[start_idx:end_idx] = part.mean(axis=1)

    # Statistical threshold: μ + α · σ
    dist_mean = np.mean(mean_distances)
    dist_std = np.std(mean_distances)
    threshold = dist_mean + (std_ratio * dist_std)

    inlier_mask = mean_distances <= threshold
    inlier_indices = np.where(inlier_mask)[0]

    return inlier_indices, inlier_mask


def remove_radius_outliers_numpy(
    points: np.ndarray,
    radius: float = 0.05,
    min_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy implementation of Radius Outlier Removal (ROR).

    Rejects points that have fewer than ``min_neighbors`` within a sphere
    of given ``radius``.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` array of 3D spatial points.
    radius : float
        Search sphere radius.
    min_neighbors : int
        Minimum neighbor count required for inlier status.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(inlier_indices, inlier_mask)``
    """
    n_points = len(points)
    if n_points <= min_neighbors:
        mask = np.ones(n_points, dtype=bool)
        return np.arange(n_points), mask

    neighbor_counts = np.zeros(n_points, dtype=np.int32)
    chunk_size = 1024

    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk = points[start_idx:end_idx]

        diff = chunk[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        # Count points within radius (excluding self)
        within_radius = (dists <= radius) & (dists > 0.0)
        neighbor_counts[start_idx:end_idx] = within_radius.sum(axis=1)

    inlier_mask = neighbor_counts >= min_neighbors
    inlier_indices = np.where(inlier_mask)[0]

    return inlier_indices, inlier_mask


def clean_point_cloud(
    point_cloud: PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    method: str = "statistical",
    radius: float = 0.05,
    min_neighbors: int = 5,
) -> PointCloudProcessingResult:
    """Filter noise and outliers from a point cloud using Open3D or NumPy fallback.

    Parameters
    ----------
    point_cloud : PointCloud
        Raw unprojected point cloud.
    nb_neighbors : int
        Number of neighbors for statistical outlier removal.
    std_ratio : float
        Standard deviation threshold multiplier.
    method : {"statistical", "radius"}
        Outlier removal algorithm.
    radius : float
        Radius for radius-based outlier removal.
    min_neighbors : int
        Minimum neighbor count for radius-based outlier removal.

    Returns
    -------
    PointCloudProcessingResult
        Dataclass containing the cleaned point cloud, inlier mask, and metrics.
    """
    n_orig = point_cloud.num_points
    if n_orig == 0:
        return PointCloudProcessingResult(
            point_cloud=point_cloud,
            inlier_mask=np.zeros(0, dtype=bool),
            num_original=0,
            num_retained=0,
            outlier_ratio=0.0,
            o3d_pcd=None,
            metadata={"filter": method},
        )

    o3d_instance = None

    # 1. Attempt Open3D C++ acceleration if available
    if HAS_OPEN3D:
        try:
            o3d_pcd = to_open3d_point_cloud(point_cloud)
            if method == "statistical":
                cl, ind = o3d_pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, std_ratio=std_ratio
                )
            else:
                cl, ind = o3d_pcd.remove_radius_outlier(
                    nb_points=min_neighbors, radius=radius
                )

            inlier_indices = np.asarray(ind, dtype=np.int64)
            inlier_mask = np.zeros(n_orig, dtype=bool)
            inlier_mask[inlier_indices] = True
            o3d_instance = cl
        except Exception:
            inlier_indices, inlier_mask = remove_statistical_outliers_numpy(
                point_cloud.points, nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
    else:
        # 2. Pure NumPy fallback
        if method == "statistical":
            inlier_indices, inlier_mask = remove_statistical_outliers_numpy(
                point_cloud.points, nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
        else:
            inlier_indices, inlier_mask = remove_radius_outliers_numpy(
                point_cloud.points, radius=radius, min_neighbors=min_neighbors
            )

    # 3. Assemble filtered PointCloud dataclass
    filtered_points = point_cloud.points[inlier_mask]
    filtered_colors = (
        point_cloud.colors[inlier_mask] if point_cloud.colors is not None else None
    )
    filtered_normals = (
        point_cloud.normals[inlier_mask] if point_cloud.normals is not None else None
    )

    n_retained = len(filtered_points)
    outlier_ratio = float(n_orig - n_retained) / float(n_orig) if n_orig > 0 else 0.0

    meta = {
        "filter_method": method,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "backend": "open3d" if (HAS_OPEN3D and o3d_instance is not None) else "numpy",
    }

    cleaned_pcd = PointCloud(
        points=filtered_points,
        colors=filtered_colors,
        normals=filtered_normals,
        intrinsics=point_cloud.intrinsics,
        metadata=meta,
    )

    return PointCloudProcessingResult(
        point_cloud=cleaned_pcd,
        inlier_mask=inlier_mask,
        num_original=n_orig,
        num_retained=n_retained,
        outlier_ratio=outlier_ratio,
        o3d_pcd=o3d_instance,
        metadata=meta,
    )


# ──────────────────────────────────────────────
# PCD Format Export (Point Cloud Data v0.7)
# ──────────────────────────────────────────────

def export_point_cloud_pcd(
    point_cloud: PointCloud,
    file_path: Union[str, Path],
    binary: bool = False,
) -> Path:
    """Serialize a PointCloud into standard Point Cloud Library (.pcd) v0.7 format.

    Supports XYZ coordinates and packed RGB color fields in ASCII and Binary modes.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to export.
    file_path : str or Path
        Destination .pcd file path.
    binary : bool
        If True, writes binary format. If False, writes standard ASCII format.

    Returns
    -------
    Path
        Path to the written PCD file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = point_cloud.num_points
    has_colors = point_cloud.colors is not None and len(point_cloud.colors) == n_points

    if has_colors:
        fields = "x y z rgb"
        sizes = "4 4 4 4"
        types = "F F F F"
        counts = "1 1 1 1"
    else:
        fields = "x y z"
        sizes = "4 4 4"
        types = "F F F"
        counts = "1 1 1"

    data_type = "binary" if binary else "ascii"

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        f"DATA {data_type}\n"
    )

    if binary:
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n_points):
                p = point_cloud.points[i]
                if has_colors:
                    r, g, b = point_cloud.colors[i]
                    # Pack RGB into 32-bit float according to PCL specification:
                    # uint32 rgb = ((uint32)r << 16 | (uint32)g << 8 | (uint32)b);
                    rgb_packed_int = (int(r) << 16) | (int(g) << 8) | int(b)
                    rgb_float = struct.unpack("f", struct.pack("I", rgb_packed_int))[0]
                    f.write(struct.pack("ffff", p[0], p[1], p[2], rgb_float))
                else:
                    f.write(struct.pack("fff", p[0], p[1], p[2]))
    else:
        with open(path, "w", encoding="ascii") as f:
            f.write(header)
            for i in range(n_points):
                p = point_cloud.points[i]
                if has_colors:
                    r, g, b = point_cloud.colors[i]
                    rgb_packed_int = (int(r) << 16) | (int(g) << 8) | int(b)
                    rgb_float = struct.unpack("f", struct.pack("I", rgb_packed_int))[0]
                    f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {rgb_float:.8e}\n")
                else:
                    f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

    return path


def export_point_cloud_ply(
    point_cloud: PointCloud,
    file_path: Union[str, Path],
    binary: bool = False,
) -> Path:
    """Export a PointCloud to .ply format (delegates to geometry.save_point_cloud_ply)."""
    from src.geometry import save_point_cloud_ply
    return save_point_cloud_ply(point_cloud, file_path, binary=binary)


# ──────────────────────────────────────────────
# 3D Visualizer & Projection Screenshot Helper
# ──────────────────────────────────────────────

def render_point_cloud_views(
    point_cloud: PointCloud,
    output_image_path: Union[str, Path],
    canvas_size: Tuple[int, int] = (600, 600),
    point_size: int = 2,
    background_color: Tuple[int, int, int] = (25, 28, 35),
) -> Path:
    """Render a multi-view 3D orthographic/isometric visualization of the point cloud.

    Generates a high-quality 2D projection screenshot suitable for headless
    environments without requiring an active X11/OpenGL display server.

    Parameters
    ----------
    point_cloud : PointCloud
        3D point cloud to visualize.
    output_image_path : str or Path
        Destination image path (.png).
    canvas_size : Tuple[int, int]
        Output image dimensions ``(width, height)``.
    point_size : int
        Radius of plotted 3D points.
    background_color : Tuple[int, int, int]
        Dark theme RGB background color.

    Returns
    -------
    Path
        Saved visualization image path.
    """
    path = Path(output_image_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    w_canvas, h_canvas = canvas_size
    canvas = np.full((h_canvas, w_canvas, 3), background_color, dtype=np.uint8)

    points = point_cloud.points
    colors = point_cloud.colors if point_cloud.colors is not None else np.full_like(points, 200, dtype=np.uint8)

    if len(points) > 0:
        # Center points at origin
        center = points.mean(axis=0)
        pts_centered = points - center

        # Apply isometric rotation matrix (Yaw ~30°, Pitch ~20°)
        yaw = np.radians(35.0)
        pitch = np.radians(25.0)

        r_yaw = np.array([
            [np.cos(yaw),  0.0, np.sin(yaw)],
            [0.0,          1.0, 0.0        ],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ])

        r_pitch = np.array([
            [1.0, 0.0,           0.0          ],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch),  np.cos(pitch)],
        ])

        rot = np.dot(r_pitch, r_yaw)
        pts_rot = np.dot(pts_centered, rot.T)

        # Scale to fit canvas with margin
        max_extent = np.max(np.abs(pts_rot[:, :2]))
        if max_extent > 0:
            scale = 0.42 * min(w_canvas, h_canvas) / max_extent
        else:
            scale = 1.0

        # Project to 2D pixel coordinates
        px = (pts_rot[:, 0] * scale + (w_canvas / 2.0)).astype(np.int32)
        py = (-pts_rot[:, 1] * scale + (h_canvas / 2.0)).astype(np.int32)  # Flip Y for screen

        # Sort by depth (painter's algorithm: draw far to near)
        depth_order = np.argsort(pts_rot[:, 2])

        # Draw points
        for idx in depth_order:
            x, y = px[idx], py[idx]
            if 0 <= x < w_canvas and 0 <= y < h_canvas:
                col = colors[idx]
                # Draw circular point disk
                for dy in range(-point_size, point_size + 1):
                    for dx in range(-point_size, point_size + 1):
                        if (dx * dx + dy * dy) <= (point_size * point_size):
                            qx, qy = x + dx, y + dy
                            if 0 <= qx < w_canvas and 0 <= qy < h_canvas:
                                canvas[qy, qx] = col

    # Save rendered projection screenshot
    Image.fromarray(canvas, mode="RGB").save(str(path))
    return path


def view_point_cloud_interactive(point_cloud: PointCloud) -> None:
    """Launch an interactive Open3D 3D viewer window if Open3D and a GUI display are available."""
    if not HAS_OPEN3D:
        print("Note: Open3D is not installed. Interactive viewer unavailable.", file=sys.stderr)
        return

    try:
        o3d_pcd = to_open3d_point_cloud(point_cloud)
        o3d.visualization.draw_geometries(
            [o3d_pcd],
            window_name="Parallax 3D Point Cloud Viewer",
            width=1024,
            height=768,
        )
    except Exception as exc:
        print(f"Interactive viewer failed (likely running headless): {exc}", file=sys.stderr)


# ──────────────────────────────────────────────
# Point Cloud Processing Pipeline Function
# ──────────────────────────────────────────────

def process_point_cloud(
    point_cloud: PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    export_ply_path: Optional[Union[str, Path]] = None,
    export_pcd_path: Optional[Union[str, Path]] = None,
    render_image_path: Optional[Union[str, Path]] = None,
) -> PointCloudProcessingResult:
    """Execute complete point cloud cleanup, statistical filtering, and multi-format export.

    Parameters
    ----------
    point_cloud : PointCloud
        Raw 3D point cloud from geometry back-projection.
    nb_neighbors : int
        Neighbor count for statistical outlier removal.
    std_ratio : float
        Outlier rejection standard deviation threshold.
    export_ply_path : Optional[str or Path]
        Optional destination path to export .ply file.
    export_pcd_path : Optional[str or Path]
        Optional destination path to export .pcd file.
    render_image_path : Optional[str or Path]
        Optional destination path to save a 3D isometric screenshot.

    Returns
    -------
    PointCloudProcessingResult
        Result containing cleaned point cloud, inlier statistics, and exported paths.
    """
    # 1. Clean point cloud
    clean_result = clean_point_cloud(
        point_cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    cleaned = clean_result.point_cloud

    # 2. Export PLY if requested
    if export_ply_path:
        export_point_cloud_ply(cleaned, export_ply_path)

    # 3. Export PCD if requested
    if export_pcd_path:
        export_point_cloud_pcd(cleaned, export_pcd_path)

    # 4. Render 3D isometric projection screenshot if requested
    if render_image_path:
        render_point_cloud_views(cleaned, render_image_path)

    return clean_result


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.point_cloud",
        description="Parallax Point Cloud Processing — Filter outliers and export .ply / .pcd formats.",
    )
    parser.add_argument("image", type=str, help="Path to input RGB image.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save 3D point clouds and visualizations (default: %(default)s).",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="Neighbor count for statistical outlier filtering (default: %(default)s).",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation multiplier threshold for outlier removal (default: %(default)s).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive Open3D viewer if GUI is available.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for point cloud processing."""
    args = _build_parser().parse_args(argv)
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing point cloud for: {input_path}")
    from src.preprocessing import load_image
    from src.segmentation import segment_object
    from src.depth_estimation import estimate_depth
    from src.geometry import depth_to_point_cloud

    img = load_image(input_path)
    seg_res = segment_object(img)
    depth_res = estimate_depth(img, mask=seg_res.binary_mask)
    raw_pcd = depth_to_point_cloud(depth_res.depth_map, rgb=img, mask=seg_res.binary_mask)

    stem = input_path.stem
    ply_path = out_dir / f"{stem}_clean.ply"
    pcd_path = out_dir / f"{stem}_clean.pcd"
    view_path = out_dir / f"{stem}_3d_view.png"

    result = process_point_cloud(
        raw_pcd,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        export_ply_path=ply_path,
        export_pcd_path=pcd_path,
        render_image_path=view_path,
    )

    print(f"Original points:  {result.num_original:,}")
    print(f"Cleaned points:   {result.num_retained:,} (removed {result.outlier_ratio * 100:.1f}% outliers)")
    print(f"Saved .ply cloud: {ply_path}")
    print(f"Saved .pcd cloud: {pcd_path}")
    print(f"Saved 3D render:  {view_path}")

    if args.interactive:
        view_point_cloud_interactive(result.point_cloud)

    print("✓ Point cloud processing complete.")


if __name__ == "__main__":
    main()
