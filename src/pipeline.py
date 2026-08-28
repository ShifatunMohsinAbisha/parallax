"""
Parallax — End-to-End Single-Image 3D Object Reconstruction Pipeline

Wires together all subsystems:
  1. Image Loading & Preprocessing (src/preprocessing.py)
  2. Foreground Object Segmentation (src/segmentation.py)
  3. Monocular Depth Estimation (src/depth_estimation.py)
  4. Pinhole Camera Back-Projection (src/geometry.py)
  5. Point Cloud Post-Processing, Filtering & 3D Export (src/point_cloud.py)

Pipeline Workflow Diagram
-------------------------
  [ Input Image (2D RGB) ]
             │
             ▼
  ┌───────────────────────┐
  │  1. Preprocessing     │  --> Aspect-ratio preserving resize & normalization
  └──────────┬────────────┘
             │
             ▼
  ┌───────────────────────┐
  │  2. Segmentation      │  --> Object binary mask & contour isolation
  └──────────┬────────────┘
             │
             ▼
  ┌───────────────────────┐
  │  3. Depth Estimation  │  --> Dense per-pixel depth map & heatmap
  └──────────┬────────────┘
             │
             ▼
  ┌───────────────────────┐
  │  4. Geometry Engine   │  --> Vectorized pinhole 3D back-projection & normals
  └──────────┬────────────┘
             │
             ▼
  ┌───────────────────────┐
  │  5. Point Cloud Post  │  --> Statistical outlier filtering, PLY & PCD export,
  └───────────────────────┘      and 3D multi-view visual rendering

Usage
-----
    from src.pipeline import run_pipeline

    result = run_pipeline(
        image_path="data/sample_vase.png",
        output_dir="outputs",
        fov_degrees=60.0,
        filter_outliers=True,
    )

    print(f"Reconstructed {result.point_cloud.num_points:,} 3D points.")
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.depth_estimation import DepthResult, estimate_depth
from src.geometry import CameraIntrinsics, PointCloud, depth_to_point_cloud, estimate_camera_intrinsics
from src.mesh_reconstruction import (
    MeshReconstructionResult,
    clean_mesh,
    export_mesh_all_formats,
    reconstruct_surface_mesh,
)
from src.point_cloud import (
    PointCloudProcessingResult,
    clean_point_cloud,
    export_point_cloud_pcd,
    export_point_cloud_ply,
    render_point_cloud_views,
)
from src.preprocessing import load_image, preprocess, save_image
from src.segmentation import SegmentationResult, create_side_by_side_panel, overlay_mask, segment_object

# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Encapsulates the complete outputs and diagnostics from all stages of Parallax.

    Attributes
    ----------
    point_cloud : PointCloud
        Final cleaned 3D point cloud.
    raw_point_cloud : PointCloud
        Initial unfiltered 3D point cloud before outlier rejection.
    mesh : Optional[MeshReconstructionResult]
        Reconstructed 3D polygonal surface mesh result.
    segmentation : SegmentationResult
        Object segmentation result with binary mask, soft mask, and metadata.
    depth : DepthResult
        Monocular depth estimation result with raw depth, normalized depth, and heatmap.
    intrinsics : CameraIntrinsics
        Camera intrinsic parameters used for back-projection.
    output_files : Dict[str, Path]
        Dictionary mapping artifact names to their saved file paths on disk.
    timing : Dict[str, float]
        Execution duration (in seconds) for each stage of the pipeline.
    metadata : Dict[str, Any]
        Additional diagnostics, 3D bounds, centroid, and configuration parameters.
    """
    point_cloud: PointCloud
    raw_point_cloud: PointCloud
    mesh: Optional[MeshReconstructionResult]
    segmentation: SegmentationResult
    depth: DepthResult
    intrinsics: CameraIntrinsics
    output_files: Dict[str, Path]
    timing: Dict[str, float]
    metadata: Dict[str, Any]


# ──────────────────────────────────────────────
# Master Overview Panel Assembler
# ──────────────────────────────────────────────

def create_pipeline_overview_panel(
    original_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    depth_heatmap: np.ndarray,
    point_cloud_view_path: Optional[Union[str, Path]],
    canvas_height: int = 320,
) -> np.ndarray:
    """Assemble a 4-panel visual summary of the complete 3D reconstruction pipeline.

    Layout: [ Input RGB | Segmented Object | Depth Heatmap | 3D Rendered Cloud ]

    Parameters
    ----------
    original_rgb : np.ndarray
        Original input image array.
    segmented_rgb : np.ndarray
        Isolated foreground object array.
    depth_heatmap : np.ndarray
        Rendered depth heatmap.
    point_cloud_view_path : Optional[str or Path]
        Path to the rendered 3D projection screenshot.
    canvas_height : int
        Standardized height for each panel tile.

    Returns
    -------
    np.ndarray
        Horizontally concatenated ``(H, 4*W, 3)`` uint8 RGB overview image.
    """
    def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        target_w = int(round(w * (target_h / h)))
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    tile1 = _resize_to_height(original_rgb, canvas_height)
    tile2 = _resize_to_height(segmented_rgb, canvas_height)
    tile3 = _resize_to_height(depth_heatmap, canvas_height)

    if point_cloud_view_path and Path(point_cloud_view_path).exists():
        pcd_img = Image.open(str(point_cloud_view_path)).convert("RGB")
        tile4 = _resize_to_height(np.array(pcd_img, dtype=np.uint8), canvas_height)
    else:
        tile4 = np.zeros_like(tile1)

    panel = np.hstack([tile1, tile2, tile3, tile4])
    return panel


# ──────────────────────────────────────────────
# End-to-End Pipeline Execution
# ──────────────────────────────────────────────

def run_pipeline(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    target_size: Tuple[int, int] = (256, 256),
    fov_degrees: float = 60.0,
    depth_method: str = "auto",
    segmentation_method: str = "auto",
    mesh_method: str = "auto",
    colormap: str = "inferno",
    filter_outliers: bool = True,
    clean_mesh_geometry: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    save_intermediate_artifacts: bool = True,
    y_up: bool = True,
) -> PipelineResult:
    """Execute the end-to-end 3D object reconstruction pipeline on a single image.

    Stages:
      1. Preprocessing (Resize with aspect ratio preservation + Normalization)
      2. Object Segmentation (Isolate primary foreground object)
      3. Depth Estimation (Generate dense relative depth map & heatmap)
      4. 2D-to-3D Geometry (Vectorized pinhole unprojection into 3D camera space)
      5. Point Cloud Post-Processing (Statistical noise filtering + PLY/PCD export)
      6. 3D Surface Mesh Reconstruction (Mesh generation, cleanup & OBJ/PLY/GLB export)

    Parameters
    ----------
    image_path : str or Path
        Path to the input 2D image.
    output_dir : str or Path
        Directory to save generated 3D meshes, point clouds, and visualizations.
    target_size : Tuple[int, int]
        Preprocessing target resolution (height, width).
    fov_degrees : float
        Camera Field of View in degrees for intrinsics estimation.
    depth_method : str
        Depth estimation algorithm (``"auto"``, ``"midas_small"``, ``"geometric_shading"``).
    segmentation_method : str
        Object segmentation algorithm (``"auto"``, ``"lraspp"``, ``"saliency_grabcut"``).
    mesh_method : str
        Surface mesh reconstruction algorithm (``"auto"``, ``"grid"``, ``"poisson"``, ``"ball_pivoting"``).
    colormap : str
        Colormap name for depth heatmap rendering (``"inferno"``, ``"viridis"``, etc.).
    filter_outliers : bool
        Whether to perform statistical outlier filtering on the point cloud.
    clean_mesh_geometry : bool
        Whether to apply automated mesh cleanup and repair.
    nb_neighbors : int
        Neighbor count for statistical outlier removal.
    std_ratio : float
        Standard deviation threshold multiplier for outlier removal.
    save_intermediate_artifacts : bool
        Whether to save step-by-step diagnostic masks, heatmaps, and panels.
    y_up : bool
        Whether to use OpenGL Y-up orientation for 3D coordinates.

    Returns
    -------
    PipelineResult
        Dataclass containing the final 3D point cloud, mesh, metrics, and saved paths.
    """
    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    saved_files: Dict[str, Path] = {}
    timing: Dict[str, float] = {}

    total_start = time.perf_counter()

    # ──────────────────────────────────────────────
    # Stage 1: Preprocessing
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    prep_result = preprocess(input_path, target_size=target_size)
    working_image = prep_result["resized"]
    h, w = working_image.shape[:2]
    timing["preprocessing"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Stage 2: Object Segmentation
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    seg_result = segment_object(working_image, method=segmentation_method)
    mask = seg_result.binary_mask
    timing["segmentation"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Stage 3: Monocular Depth Estimation
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    depth_result = estimate_depth(
        working_image,
        method=depth_method,
        mask=mask,
        colormap=colormap,
    )
    timing["depth_estimation"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Stage 4: 2D-to-3D Geometry & Unprojection
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    intrinsics = estimate_camera_intrinsics(image_size=(h, w), fov_degrees=fov_degrees)
    raw_pcd = depth_to_point_cloud(
        depth_map=depth_result.depth_map,
        intrinsics=intrinsics,
        rgb=working_image,
        mask=mask,
        compute_normals=True,
        y_up=y_up,
    )
    timing["geometry_unprojection"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Stage 5: Point Cloud Filtering & 3D Serialization
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    if filter_outliers and raw_pcd.num_points > nb_neighbors:
        clean_res = clean_point_cloud(
            raw_pcd,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        final_pcd = clean_res.point_cloud
    else:
        final_pcd = raw_pcd

    # Export standard 3D file formats
    ply_path = out_dir / f"{stem}_reconstruction.ply"
    pcd_path = out_dir / f"{stem}_reconstruction.pcd"
    export_point_cloud_ply(final_pcd, ply_path)
    export_point_cloud_pcd(final_pcd, pcd_path)
    saved_files["point_cloud_ply"] = ply_path
    saved_files["point_cloud_pcd"] = pcd_path

    # Render 3D isometric projection screenshot
    view_3d_path = out_dir / f"{stem}_3d_isometric.png"
    render_point_cloud_views(final_pcd, view_3d_path)
    saved_files["view_3d"] = view_3d_path
    timing["point_cloud_processing"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Stage 6: 3D Surface Mesh Reconstruction
    # ──────────────────────────────────────────────
    t0 = time.perf_counter()
    mesh_res = reconstruct_surface_mesh(
        final_pcd,
        method=mesh_method,
        clean=clean_mesh_geometry,
    )
    # Export mesh in OBJ, PLY, and GLB formats
    mesh_files = export_mesh_all_formats(mesh_res.mesh, out_dir, stem=f"{stem}_reconstruction")
    saved_files.update(mesh_files)
    mesh_res.output_files = mesh_files
    timing["mesh_reconstruction"] = time.perf_counter() - t0

    # ──────────────────────────────────────────────
    # Optional Intermediate & Overview Visualizations
    # ──────────────────────────────────────────────
    if save_intermediate_artifacts:
        # 1. Mask overlay
        overlay = overlay_mask(working_image, mask)
        overlay_path = out_dir / f"{stem}_stage2_mask_overlay.png"
        save_image(overlay, overlay_path)
        saved_files["stage2_mask_overlay"] = overlay_path

        # 2. Depth Heatmap
        heatmap_path = out_dir / f"{stem}_stage3_depth_heatmap.png"
        save_image(depth_result.heatmap, heatmap_path)
        saved_files["stage3_depth_heatmap"] = heatmap_path

        # 3. Master 4-panel overview
        overview_panel = create_pipeline_overview_panel(
            original_rgb=working_image,
            segmented_rgb=seg_result.masked_image,
            depth_heatmap=depth_result.heatmap,
            point_cloud_view_path=view_3d_path,
        )
        overview_path = out_dir / f"{stem}_pipeline_overview.png"
        save_image(overview_panel, overview_path)
        saved_files["pipeline_overview"] = overview_path

    timing["total_pipeline"] = time.perf_counter() - total_start

    min_b, max_b = final_pcd.bounds
    center = final_pcd.center

    metadata = {
        "image_size": (h, w),
        "fov_degrees": fov_degrees,
        "depth_method": depth_result.method,
        "segmentation_method": seg_result.method,
        "mesh_method": mesh_res.method,
        "raw_points": raw_pcd.num_points,
        "cleaned_points": final_pcd.num_points,
        "num_vertices": mesh_res.num_vertices,
        "num_faces": mesh_res.num_faces,
        "is_watertight": mesh_res.is_watertight,
        "centroid": center.tolist(),
        "bounds_min": min_b.tolist(),
        "bounds_max": max_b.tolist(),
        "foreground_ratio": seg_result.foreground_ratio,
    }

    return PipelineResult(
        point_cloud=final_pcd,
        raw_point_cloud=raw_pcd,
        mesh=mesh_res,
        segmentation=seg_result,
        depth=depth_result,
        intrinsics=intrinsics,
        output_files=saved_files,
        timing=timing,
        metadata=metadata,
    )


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Parallax Single-Image 3D Object Reconstruction Pipeline.",
    )
    parser.add_argument("image", type=str, help="Path to input 2D image file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save reconstructed 3D models and visuals (default: %(default)s).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Estimated horizontal camera FOV in degrees (default: %(default)s°).",
    )
    parser.add_argument(
        "--depth-method",
        type=str,
        default="auto",
        choices=["auto", "midas_small", "geometric_shading"],
        help="Depth estimation engine (default: %(default)s).",
    )
    parser.add_argument(
        "--segmentation-method",
        type=str,
        default="auto",
        choices=["auto", "lraspp", "saliency_grabcut"],
        help="Object segmentation engine (default: %(default)s).",
    )
    parser.add_argument(
        "--mesh-method",
        type=str,
        default="auto",
        choices=["auto", "grid", "poisson", "ball_pivoting"],
        help="3D surface mesh reconstruction engine (default: %(default)s).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        help="Colormap for depth visualization (default: %(default)s).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable statistical outlier filtering.",
    )
    parser.add_argument(
        "--no-clean-mesh",
        action="store_true",
        help="Disable automatic mesh repair.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for running the complete Parallax reconstruction pipeline."""
    args = _build_parser().parse_args(argv)
    input_path = Path(args.image)

    print("=" * 65)
    print("  PARALLAX — Single-Image 3D Object Reconstruction Pipeline  ")
    print("=" * 65)
    print(f"Input Image:         {input_path}")
    print(f"Output Directory:    {args.output_dir}")
    print(f"Camera FOV:          {args.fov}°")
    print(f"Depth Engine:        {args.depth_method}")
    print(f"Segmentation Engine: {args.segmentation_method}")
    print(f"Mesh Engine:         {args.mesh_method}")
    print("-" * 65)

    result = run_pipeline(
        image_path=input_path,
        output_dir=args.output_dir,
        fov_degrees=args.fov,
        depth_method=args.depth_method,
        segmentation_method=args.segmentation_method,
        mesh_method=args.mesh_method,
        colormap=args.colormap,
        filter_outliers=not args.no_filter,
        clean_mesh_geometry=not args.no_clean_mesh,
    )

    print("\n--- Pipeline Execution Summary ---")
    print(f"1. Preprocessing:        {result.timing['preprocessing']*1000:.1f} ms")
    print(f"2. Segmentation:         {result.timing['segmentation']*1000:.1f} ms ({result.segmentation.method})")
    print(f"3. Depth Estimation:     {result.timing['depth_estimation']*1000:.1f} ms ({result.depth.method})")
    print(f"4. Geometry Unproject:   {result.timing['geometry_unprojection']*1000:.1f} ms")
    print(f"5. Point Cloud Post:     {result.timing['point_cloud_processing']*1000:.1f} ms")
    print(f"6. Mesh Reconstruction:  {result.timing['mesh_reconstruction']*1000:.1f} ms ({result.metadata['mesh_method']})")
    print(f"Total Execution Time:    {result.timing['total_pipeline']*1000:.1f} ms ({result.timing['total_pipeline']:.2f} s)")

    print("\n--- 3D Reconstruction Metrics ---")
    print(f"Raw Points:              {result.raw_point_cloud.num_points:,}")
    print(f"Final Cleaned Points:    {result.point_cloud.num_points:,}")
    if result.mesh:
        print(f"Mesh Vertices:           {result.mesh.num_vertices:,}")
        print(f"Mesh Faces (Triangles):  {result.mesh.num_faces:,}")
        print(f"Watertight Manifold:     {result.mesh.is_watertight}")
    center = result.point_cloud.center
    print(f"3D Geometric Centroid:   X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")

    print("\n--- Generated Artifacts ---")
    for key, path in result.output_files.items():
        print(f"  • {key.ljust(24)}: {path}")

    print("=" * 65)
    print("✓ Parallax 3D reconstruction pipeline finished successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
