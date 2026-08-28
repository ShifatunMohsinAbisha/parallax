"""
Parallax — 2D-to-3D Geometry & Back-Projection Module

Converts 2D pixel coordinates, monocular depth estimates, and camera intrinsics
into dense 3D point clouds using the classical pinhole camera model.

Pinhole Camera Model & Math Reference
======================================
The standard pinhole camera projection matrix K maps a 3D camera-frame point
P = [X, Y, Z]^T to a 2D homogeneous pixel coordinate p = [u, v, 1]^T:

               ┌               ┐ ┌   ┐
               │ f_x   0   c_x │ │ X │
       Z · p = │  0   f_y  c_y │ │ Y │ = K · P
               │  0    0    1  │ │ Z │
               └               ┘ └   ┘

Where:
  - f_x, f_y : Focal length in horizontal and vertical pixel units.
               When unknown, estimated from the horizontal Field of View (FOV θ):
                   f_x = f_y = W / (2 · tan(θ / 2))
  - c_x, c_y : Principal point (optical center on the image sensor plane):
                   c_x = (W - 1) / 2.0  (or W / 2.0)
                   c_y = (H - 1) / 2.0  (or H / 2.0)

Inverse Back-Projection (2D Depth → 3D Space):
----------------------------------------------
Given an observed pixel (u, v) with scalar depth Z = d(u, v) > 0, the 3D position
P = (X, Y, Z) in the camera coordinate frame is recovered by inverting K:

       P = Z · K^(-1) · [u, v, 1]^T

Explicit scalar coordinate equations:
       X = (u - c_x) · Z / f_x
       Y = (v - c_y) · Z / f_y  (or Y = -(v - c_y) · Z / f_y in OpenGL Y-up convention)
       Z = Z

Vectorized Meshgrid Implementation:
-----------------------------------
For an image of size (H, W), we construct 2D coordinate grids U and V:
       U, V = np.meshgrid(np.arange(W), np.arange(H))
       X_grid = (U - c_x) * Depth / f_x
       Y_grid = (V - c_y) * Depth / f_y
       Z_grid = Depth

Applying a binary segmentation mask M (where M(u, v) = 1 for the object and 0 for background):
       Masked_Points = [X_grid[M], Y_grid[M], Z_grid[M]]
       Masked_Colors = RGB_Image[M]

Usage
-----
    from src.geometry import (
        CameraIntrinsics,
        estimate_camera_intrinsics,
        depth_to_point_cloud,
        save_point_cloud_ply,
    )

    # 1. Estimate or specify camera intrinsics for a 256×256 image
    intrinsics = estimate_camera_intrinsics(image_size=(256, 256), fov_degrees=60.0)

    # 2. Back-project depth and mask to 3D point cloud
    point_cloud = depth_to_point_cloud(
        depth_map=depth_array,
        intrinsics=intrinsics,
        rgb=rgb_image,
        mask=object_mask,
    )

    # 3. Export to standard 3D PLY format
    save_point_cloud_ply(point_cloud, "outputs/reconstruction.ply")
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# Data Structures: Camera Intrinsics & Point Cloud
# ──────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Represents pinhole camera intrinsic parameters.

    Attributes
    ----------
    fx : float
        Focal length along the horizontal axis (in pixels).
    fy : float
        Focal length along the vertical axis (in pixels).
    cx : float
        Principal point x-coordinate (optical center, in pixels).
    cy : float
        Principal point y-coordinate (optical center, in pixels).
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fov_degrees : float
        Estimated horizontal Field of View in degrees.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fov_degrees: float

    def to_matrix(self) -> np.ndarray:
        """Return the 3×3 camera calibration matrix K.

        Returns
        -------
        np.ndarray
            3×3 float64 camera matrix K.
        """
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float64)

    def inv_matrix(self) -> np.ndarray:
        """Return the 3×3 inverse camera calibration matrix K^(-1).

        Returns
        -------
        np.ndarray
            3×3 float64 matrix K^(-1).
        """
        return np.linalg.inv(self.to_matrix())


@dataclass
class PointCloud:
    """Encapsulates a 3D point cloud with optional colors and surface normals.

    Attributes
    ----------
    points : np.ndarray
        ``(N, 3)`` float32 array of 3D spatial coordinates [X, Y, Z].
    colors : Optional[np.ndarray]
        ``(N, 3)`` uint8 array of RGB color values in range [0, 255].
    normals : Optional[np.ndarray]
        ``(N, 3)`` float32 array of unit normal vectors [Nx, Ny, Nz].
    intrinsics : CameraIntrinsics
        Camera intrinsic parameters used during unprojection.
    metadata : Dict[str, Any]
        Additional diagnostics, bounding box dimensions, and point count.
    """
    points: np.ndarray
    colors: Optional[np.ndarray]
    normals: Optional[np.ndarray]
    intrinsics: CameraIntrinsics
    metadata: Dict[str, Any]

    @property
    def num_points(self) -> int:
        """Return the total number of 3D points in the cloud."""
        return int(len(self.points))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the 3D axis-aligned bounding box (min_bounds, max_bounds)."""
        if len(self.points) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        """Return the 3D geometric centroid of the point cloud."""
        if len(self.points) == 0:
            return np.zeros(3, dtype=np.float32)
        return self.points.mean(axis=0)


# ──────────────────────────────────────────────
# Camera Intrinsics Estimation
# ──────────────────────────────────────────────

def estimate_camera_intrinsics(
    image_size: Tuple[int, int],
    fov_degrees: float = 60.0,
    focal_length_ratio: Optional[float] = None,
    principal_point: Optional[Tuple[float, float]] = None,
) -> CameraIntrinsics:
    """Compute pinhole camera intrinsics from image dimensions and FOV.

    Mathematical Formulation:
    -------------------------
    Given image dimensions (H, W) and horizontal Field of View θ_fov:
        f_x = W / (2 · tan(θ_fov / 2))
        f_y = f_x  (assuming square pixels)
        c_x = (W - 1) / 2.0
        c_y = (H - 1) / 2.0

    Parameters
    ----------
    image_size : Tuple[int, int]
        Image dimensions as ``(height, width)`` or ``(H, W)``.
    fov_degrees : float
        Horizontal Field of View in degrees (default: 60.0°, standard for consumer cameras).
    focal_length_ratio : Optional[float]
        Optional direct multiplier for focal length (e.g. 1.2 * max(W, H)).
        If specified, overrides ``fov_degrees``.
    principal_point : Optional[Tuple[float, float]]
        Optional explicit principal point ``(cx, cy)``. If None, defaults to image center.

    Returns
    -------
    CameraIntrinsics
        Configured intrinsic parameters dataclass.
    """
    height, width = image_size

    if focal_length_ratio is not None:
        fx = float(max(width, height) * focal_length_ratio)
        fy = fx
        # Compute equivalent FOV
        fov_rad = 2.0 * np.arctan(width / (2.0 * fx))
        fov_deg = float(np.degrees(fov_rad))
    else:
        fov_rad = np.radians(fov_degrees)
        fx = float(width / (2.0 * np.tan(fov_rad / 2.0)))
        fy = fx
        fov_deg = float(fov_degrees)

    if principal_point is not None:
        cx, cy = principal_point
    else:
        cx = float((width - 1) / 2.0)
        cy = float((height - 1) / 2.0)

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(width),
        height=int(height),
        fov_degrees=fov_deg,
    )


# ──────────────────────────────────────────────
# Pinhole Coordinate Projections
# ──────────────────────────────────────────────

def unproject_pixel(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
    y_up: bool = False,
) -> Tuple[float, float, float]:
    """Back-project a single 2D pixel (u, v) and depth Z into a 3D coordinate (X, Y, Z).

    Equations:
        X = (u - cx) · Z / fx
        Y = (v - cy) · Z / fy  (or -(v - cy) · Z / fy if y_up=True)
        Z = Z

    Parameters
    ----------
    u : float
        Horizontal pixel column coordinate [0, W - 1].
    v : float
        Vertical pixel row coordinate [0, H - 1].
    depth : float
        Estimated metric or relative depth along the camera optical z-axis (Z > 0).
    intrinsics : CameraIntrinsics
        Camera intrinsic parameters.
    y_up : bool
        If True, flips the Y-axis so positive Y points upward (OpenGL convention).
        If False, positive Y points downward along the image rows (OpenCV convention).

    Returns
    -------
    Tuple[float, float, float]
        3D point coordinates ``(X, Y, Z)``.
    """
    x = float((u - intrinsics.cx) * depth / intrinsics.fx)
    if y_up:
        y = float(-(v - intrinsics.cy) * depth / intrinsics.fy)
    else:
        y = float((v - intrinsics.cy) * depth / intrinsics.fy)
    z = float(depth)
    return (x, y, z)


def project_3d_to_pixel(
    point_3d: Union[Tuple[float, float, float], np.ndarray],
    intrinsics: CameraIntrinsics,
    y_up: bool = False,
) -> Tuple[float, float]:
    """Forward-project a 3D camera-frame point (X, Y, Z) to 2D pixel coordinates (u, v).

    Equations:
        u = (X · fx / Z) + cx
        v = (Y · fy / Z) + cy  (or -(Y · fy / Z) + cy if y_up=True)

    Parameters
    ----------
    point_3d : array-like
        3D point ``(X, Y, Z)`` with Z > 0.
    intrinsics : CameraIntrinsics
        Camera intrinsic parameters.
    y_up : bool
        Whether the 3D point uses OpenGL Y-up convention.

    Returns
    -------
    Tuple[float, float]
        2D pixel coordinates ``(u, v)``.
    """
    x, y, z = point_3d[0], point_3d[1], point_3d[2]
    if z <= 0:
        raise ValueError(f"Cannot project point with non-positive depth: Z={z}")

    u = float((x * intrinsics.fx / z) + intrinsics.cx)
    if y_up:
        v = float((-y * intrinsics.fy / z) + intrinsics.cy)
    else:
        v = float((y * intrinsics.fy / z) + intrinsics.cy)
    return (u, v)


# ──────────────────────────────────────────────
# Surface Normal Estimation from Depth Grid
# ──────────────────────────────────────────────

def estimate_surface_normals_from_depth(
    points_grid: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute local 3D surface normals from a dense unprojected points grid.

    Mathematical Formulation:
    -------------------------
    Using central finite differences across the structured 3D spatial grid P(v, u):
        dP/du = (P[v, u+1] - P[v, u-1]) / 2.0   (tangent vector in u)
        dP/dv = (P[v+1, u] - P[v-1, u]) / 2.0   (tangent vector in v)
        Normal N = normalize(dP/du × dP/dv)

    Parameters
    ----------
    points_grid : np.ndarray
        ``(H, W, 3)`` structured 3D points array.
    mask : Optional[np.ndarray]
        Optional ``(H, W)`` mask to select valid normal vectors.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` or ``(N, 3)`` array of unit surface normal vectors.
    """
    h, w = points_grid.shape[:2]

    # Compute spatial gradients using Sobel / finite differences
    dp_du = np.zeros_like(points_grid)
    dp_dv = np.zeros_like(points_grid)

    dp_du[:, 1:-1] = (points_grid[:, 2:] - points_grid[:, :-2]) * 0.5
    dp_du[:, 0] = points_grid[:, 1] - points_grid[:, 0]
    dp_du[:, -1] = points_grid[:, -1] - points_grid[:, -2]

    dp_dv[1:-1, :] = (points_grid[2:, :] - points_grid[:-2, :]) * 0.5
    dp_dv[0, :] = points_grid[1, :] - points_grid[0, :]
    dp_dv[-1, :] = points_grid[-1, :] - points_grid[-2, :]

    # Cross product: tangent_u × tangent_v
    normals = np.cross(dp_du, dp_dv)

    # Normalize vectors to unit length
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-8
    unit_normals = -normals / norm  # Invert so normal points toward camera (positive Z dot)

    if mask is not None:
        valid = (mask > 0)
        return unit_normals[valid].astype(np.float32)

    return unit_normals.astype(np.float32)


# ──────────────────────────────────────────────
# Dense 2D Depth → 3D Point Cloud Pipeline
# ──────────────────────────────────────────────

def depth_to_point_cloud(
    depth_map: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    rgb: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    fov_degrees: float = 60.0,
    min_depth: float = 1e-4,
    max_depth: float = 1e6,
    compute_normals: bool = True,
    y_up: bool = True,
) -> PointCloud:
    """Convert an RGB image, depth map, and optional segmentation mask into a 3D PointCloud.

    Mathematical Workflow:
    ----------------------
    1. Construct 2D grid of pixel coordinates (u, v).
    2. Back-project depth map Z = d(u, v) using pinhole camera equations:
           X(u, v) = (u - cx) · Z / fx
           Y(u, v) = -(v - cy) · Z / fy  (when y_up=True for OpenGL 3D space)
           Z(u, v) = Z
    3. Filter points using the segmentation mask M:
           Valid = (M(u, v) > 0) & (Z > min_depth) & (Z < max_depth)
    4. Extract corresponding RGB color values and compute surface normals.

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 array of relative or metric depth values.
    intrinsics : Optional[CameraIntrinsics]
        Camera intrinsic parameters. If None, automatically estimated from
        image dimensions and ``fov_degrees``.
    rgb : Optional[np.ndarray]
        Optional ``(H, W, 3)`` uint8 RGB image for coloring 3D points.
    mask : Optional[np.ndarray]
        Optional ``(H, W)`` binary or soft mask (1 = include object point, 0 = exclude background).
    fov_degrees : float
        Horizontal FOV in degrees (used when ``intrinsics`` is None).
    min_depth : float
        Minimum valid depth threshold.
    max_depth : float
        Maximum valid depth threshold.
    compute_normals : bool
        Whether to calculate 3D unit surface normals for all points.
    y_up : bool
        If True, orients Y-axis upward (OpenGL / standard 3D viewer convention).
        If False, orients Y-axis downward (OpenCV image coordinate convention).

    Returns
    -------
    PointCloud
        Dataclass containing 3D spatial coordinates, RGB colors, normals, and intrinsics.
    """
    h, w = depth_map.shape[:2]

    # 1. Resolve camera intrinsics
    if intrinsics is None:
        intrinsics = estimate_camera_intrinsics(image_size=(h, w), fov_degrees=fov_degrees)

    # 2. Vectorized 2D grid generation
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # 3. Vectorized pinhole unprojection equations
    depth_clamped = np.maximum(depth_map.astype(np.float32), 0.0)
    x_grid = (u_grid - intrinsics.cx) * depth_clamped / intrinsics.fx
    if y_up:
        y_grid = -(v_grid - intrinsics.cy) * depth_clamped / intrinsics.fy
    else:
        y_grid = (v_grid - intrinsics.cy) * depth_clamped / intrinsics.fy
    z_grid = depth_clamped

    points_grid = np.stack([x_grid, y_grid, z_grid], axis=-1)  # (H, W, 3)

    # 4. Determine valid point filter mask
    valid_mask = (depth_clamped > min_depth) & (depth_clamped <= max_depth)
    if mask is not None:
        if mask.shape[:2] != (h, w):
            raise ValueError(f"Mask shape {mask.shape[:2]} does not match depth shape {(h, w)}")
        valid_mask = valid_mask & (mask > 0.5)

    # 5. Extract unprojected 3D points
    points = points_grid[valid_mask].astype(np.float32)

    # 6. Extract RGB colors if provided
    colors = None
    if rgb is not None:
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"RGB image shape {rgb.shape[:2]} does not match depth shape {(h, w)}")
        if rgb.dtype in (np.float32, np.float64):
            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgb_u8 = rgb.astype(np.uint8)
        colors = rgb_u8[valid_mask]

    # 7. Compute surface normal vectors
    normals = None
    if compute_normals and len(points) > 0:
        normals = estimate_surface_normals_from_depth(points_grid, mask=valid_mask)

    metadata = {
        "total_pixels": h * w,
        "valid_points": len(points),
        "point_density": float(len(points)) / float(h * w) if (h * w) > 0 else 0.0,
        "y_up": y_up,
    }

    return PointCloud(
        points=points,
        colors=colors,
        normals=normals,
        intrinsics=intrinsics,
        metadata=metadata,
    )


# ──────────────────────────────────────────────
# Point Cloud File Exporters (PLY & OBJ)
# ──────────────────────────────────────────────

def save_point_cloud_ply(
    point_cloud: PointCloud,
    file_path: Union[str, Path],
    binary: bool = False,
) -> Path:
    """Save a PointCloud to standard Polygon File Format (PLY).

    Supports ASCII and Binary Little-Endian output formats with XYZ coordinates,
    RGB colors, and NxNyNz surface normals.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to serialize.
    file_path : str or Path
        Destination .ply file path.
    binary : bool
        If True, writes compact binary little-endian format. If False, writes standard ASCII.

    Returns
    -------
    Path
        Path to the saved PLY file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = point_cloud.num_points
    has_colors = point_cloud.colors is not None and len(point_cloud.colors) == n_points
    has_normals = point_cloud.normals is not None and len(point_cloud.normals) == n_points

    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        "comment Created by Parallax 3D Reconstruction",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_normals:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    if has_colors:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.append("end_header\n")
    header_str = "\n".join(header_lines)

    if binary:
        with open(path, "wb") as f:
            f.write(header_str.encode("ascii"))
            for i in range(n_points):
                # Write XYZ
                f.write(point_cloud.points[i].astype("<f4").tobytes())
                # Write Normals
                if has_normals:
                    f.write(point_cloud.normals[i].astype("<f4").tobytes())
                # Write Colors
                if has_colors:
                    f.write(point_cloud.colors[i].astype("u1").tobytes())
    else:
        with open(path, "w", encoding="ascii") as f:
            f.write(header_str)
            for i in range(n_points):
                p = point_cloud.points[i]
                line_parts = [f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}"]

                if has_normals:
                    norm = point_cloud.normals[i]
                    line_parts.append(f"{norm[0]:.5f} {norm[1]:.5f} {norm[2]:.5f}")

                if has_colors:
                    col = point_cloud.colors[i]
                    line_parts.append(f"{int(col[0])} {int(col[1])} {int(col[2])}")

                f.write(" ".join(line_parts) + "\n")

    return path


def save_point_cloud_obj(
    point_cloud: PointCloud,
    file_path: Union[str, Path],
) -> Path:
    """Save a PointCloud to Wavefront OBJ format.

    Parameters
    ----------
    point_cloud : PointCloud
        Point cloud to serialize.
    file_path : str or Path
        Destination .obj file path.

    Returns
    -------
    Path
        Path to the saved OBJ file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = point_cloud.num_points
    has_colors = point_cloud.colors is not None and len(point_cloud.colors) == n_points

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Parallax 3D Point Cloud Reconstruction\n")
        f.write(f"# Number of vertices: {n_points}\n")

        for i in range(n_points):
            p = point_cloud.points[i]
            if has_colors:
                # OBJ vertex colors in range [0.0, 1.0]
                c = point_cloud.colors[i].astype(np.float32) / 255.0
                f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            else:
                f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

    return path


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.geometry",
        description="Parallax 2D-to-3D Geometry — Unproject RGB + Depth + Mask into a 3D Point Cloud.",
    )
    parser.add_argument("image", type=str, help="Path to input RGB image file.")
    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="Optional path to precomputed .npy depth map. If omitted, depth is estimated on the fly.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional path to binary segmentation mask. If omitted, segmented automatically.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera horizontal Field of View in degrees (default: %(default)s°).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save 3D point cloud outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="ply",
        choices=["ply", "obj", "both"],
        help="Output 3D point cloud file format (default: %(default)s).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for running 2D-to-3D unprojection."""
    args = _build_parser().parse_args(argv)
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Back-projecting 3D geometry for: {input_path}")
    from src.preprocessing import load_image

    img = load_image(input_path)
    h, w = img.shape[:2]

    # 1. Load or compute mask
    if args.mask and Path(args.mask).exists():
        mask_img = Image.open(args.mask).convert("L")
        mask = (np.array(mask_img) > 127).astype(np.uint8)
    else:
        from src.segmentation import segment_object
        print("Segmenting foreground object...")
        seg_res = segment_object(img)
        mask = seg_res.binary_mask

    # 2. Load or compute depth
    if args.depth and Path(args.depth).exists():
        depth_map = np.load(args.depth).astype(np.float32)
    else:
        from src.depth_estimation import estimate_depth
        print("Estimating depth map...")
        depth_res = estimate_depth(img, mask=mask)
        depth_map = depth_res.depth_map

    # 3. Estimate camera intrinsics
    intrinsics = estimate_camera_intrinsics(image_size=(h, w), fov_degrees=args.fov)
    print(f"Camera Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f} (FOV: {intrinsics.fov_degrees}°)")

    # 4. Generate Point Cloud
    point_cloud = depth_to_point_cloud(
        depth_map=depth_map,
        intrinsics=intrinsics,
        rgb=img,
        mask=mask,
        compute_normals=True,
        y_up=True,
    )

    min_b, max_b = point_cloud.bounds
    center = point_cloud.center
    print(f"Generated {point_cloud.num_points:,} 3D points")
    print(f"3D Centroid:   X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
    print(f"3D Bounding Box: min=[{min_b[0]:.2f}, {min_b[1]:.2f}, {min_b[2]:.2f}] max=[{max_b[0]:.2f}, {max_b[1]:.2f}, {max_b[2]:.2f}]")

    # 5. Save outputs
    stem = input_path.stem
    if args.format in ("ply", "both"):
        ply_path = out_dir / f"{stem}_pointcloud.ply"
        save_point_cloud_ply(point_cloud, ply_path)
        print(f"Saved 3D PLY: {ply_path}")

    if args.format in ("obj", "both"):
        obj_path = out_dir / f"{stem}_pointcloud.obj"
        save_point_cloud_obj(point_cloud, obj_path)
        print(f"Saved 3D OBJ: {obj_path}")

    print("✓ 2D-to-3D back-projection complete.")


if __name__ == "__main__":
    main()
