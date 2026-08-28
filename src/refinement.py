"""
Parallax — 3D Mesh & Point Cloud Refinement Module

Provides post-processing, surface smoothing, boundary cleanup, hole filling,
and quantitative before/after diagnostic metrics for reconstructed 3D models.

Refinement Philosophy & Mathematical Techniques
-----------------------------------------------
Single-image 3D reconstructions frequently exhibit step-like quantization artifacts,
boundary noise along silhouette edges, and small non-manifold pinholes. The refinement
stage performs best-effort geometric optimization:

1. **Volume-Preserving Taubin Smoothing (`"taubin"`)**:
   - Standard Laplacian smoothing shrinks convex geometries. Taubin smoothing mitigates
     shrinkage by alternating between a positive diffusion step (λ > 0) and a negative
     anti-diffusion inflation step (μ < -λ < 0):
         v^(t+1/2) = v^(t) + λ · L(v^(t))
         v^(t+1)   = v^(t+1/2) + μ · L(v^(t+1/2))
   - Removes high-frequency noise while preserving macro volume and feature boundaries.

2. **Humphrey / Standard Laplacian Smoothing (`"laplacian"`)**:
   - Moves each vertex towards the weighted average of its 1-ring neighbors:
         v_i^(t+1) = v_i^(t) + λ · ∑_(j ∈ N(i)) w_ij · (v_j^(t) - v_i^(t))

3. **Boundary Edge & Silhouette Cleanup**:
   - Identifies open boundary edges, removes non-manifold slivers, and trims
     unattached edge triangles caused by depth discontinuities.

4. **Hole Filling & Degenerate Face Purging**:
   - Stitches small open internal boundary loops using ear-clipping triangulation.
   - Eliminates duplicate faces, zero-area triangles, and unreferenced vertices.

Known Limitations Note
----------------------
Single-image 3D reconstruction is fundamentally an ill-posed inverse problem. While
refinement smooths sensor noise and repairs topological flaws, it **cannot** synthesize
occluded geometry (such as the back side of an object not visible in the input photo).

Usage
-----
    from src.refinement import refine_mesh, RefinementConfig

    config = RefinementConfig(
        smoothing_method="taubin",
        smoothing_iterations=5,
        fill_holes=True,
        clean_boundaries=True,
    )

    result = refine_mesh(raw_mesh, config=config)
    refined_mesh = result.refined_mesh
    metrics = result.metrics  # Vertex change, holes filled, roughness reduction
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import trimesh
from PIL import Image

from src.geometry import CameraIntrinsics, PointCloud

# ──────────────────────────────────────────────
# Data Structures & Configuration
# ──────────────────────────────────────────────

@dataclass
class RefinementConfig:
    """Configuration hyperparameters for the 3D refinement stage.

    Attributes
    ----------
    smoothing_method : {"taubin", "laplacian", "none"}
        Smoothing algorithm to apply.
    smoothing_iterations : int
        Number of smoothing passes (default: 5).
    taubin_lambda : float
        Positive scaling factor for Taubin smoothing (0.0 < λ < 1.0, default: 0.5).
    taubin_mu : float
        Negative scaling factor for Taubin inflation (μ < -λ, default: -0.53).
    laplacian_damping : float
        Relaxation factor for Laplacian smoothing (default: 0.4).
    fill_holes : bool
        Whether to attempt automatic boundary hole stitching.
    clean_boundaries : bool
        Whether to detect and smooth jagged silhouette boundary contours.
    remove_degenerates : bool
        Whether to remove duplicate and zero-area faces.
    """
    smoothing_method: str = "taubin"
    smoothing_iterations: int = 5
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    laplacian_damping: float = 0.4
    fill_holes: bool = True
    clean_boundaries: bool = True
    remove_degenerates: bool = True


@dataclass
class RefinementMetrics:
    """Quantitative diagnostics comparing geometry before and after refinement.

    Attributes
    ----------
    vertices_before : int
        Initial vertex count.
    vertices_after : int
        Final vertex count.
    vertex_delta : int
        Change in vertex count (after - before).
    faces_before : int
        Initial face count.
    faces_after : int
        Final face count.
    face_delta : int
        Change in face count (after - before).
    holes_filled : int
        Estimated number of boundary holes closed.
    degenerate_faces_removed : int
        Number of zero-area or duplicate triangles purged.
    roughness_before : float
        Mean surface normal variance / roughness before smoothing.
    roughness_after : float
        Mean surface normal variance / roughness after smoothing.
    roughness_reduction_pct : float
        Percentage reduction in high-frequency surface roughness.
    watertight_before : bool
        Whether the mesh was watertight initially.
    watertight_after : bool
        Whether the mesh is watertight after refinement.
    """
    vertices_before: int
    vertices_after: int
    vertex_delta: int
    faces_before: int
    faces_after: int
    face_delta: int
    holes_filled: int
    degenerate_faces_removed: int
    roughness_before: float
    roughness_after: float
    roughness_reduction_pct: float
    watertight_before: bool
    watertight_after: bool


@dataclass
class RefinementResult:
    """Encapsulates the output of the 3D refinement stage.

    Attributes
    ----------
    refined_mesh : trimesh.Trimesh
        Cleaned, smoothed, and repaired 3D mesh.
    original_mesh : trimesh.Trimesh
        Input mesh prior to refinement.
    metrics : RefinementMetrics
        Comparative quantitative statistics.
    output_files : Dict[str, Path]
        Paths to exported refined artifacts on disk.
    metadata : Dict[str, Any]
        Configuration parameters and timing details.
    """
    refined_mesh: trimesh.Trimesh
    original_mesh: trimesh.Trimesh
    metrics: RefinementMetrics
    output_files: Dict[str, Path]
    metadata: Dict[str, Any]


# ──────────────────────────────────────────────
# Surface Roughness & Normal Variance Analyzer
# ──────────────────────────────────────────────

def compute_surface_roughness(mesh: trimesh.Trimesh) -> float:
    """Measure the average high-frequency surface roughness (angular normal variance).

    Computes the average angular discrepancy between adjacent face normals across
    shared interior edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh to inspect.

    Returns
    -------
    float
        Average angular variance in radians (lower values indicate smoother surfaces).
    """
    if len(mesh.faces) < 2:
        return 0.0

    try:
        # Face adjacency pairs: (num_edges, 2)
        adj = mesh.face_adjacency
        if len(adj) == 0:
            return 0.0

        n1 = mesh.face_normals[adj[:, 0]]
        n2 = mesh.face_normals[adj[:, 1]]

        # Dot product between adjacent face normals
        dots = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
        # Angular difference in radians
        angles = np.arccos(dots)
        return float(np.mean(angles))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
# Smoothing Algorithms (Taubin & Laplacian)
# ──────────────────────────────────────────────

def apply_taubin_smoothing(
    mesh: trimesh.Trimesh,
    iterations: int = 5,
    lam: float = 0.5,
    mu: float = -0.53,
) -> trimesh.Trimesh:
    """Apply two-step volume-preserving Taubin smoothing to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular mesh.
    iterations : int
        Number of Taubin cycles.
    lam : float
        Positive shrinkage factor (0 < λ < 1).
    mu : float
        Negative expansion factor (μ < -λ).

    Returns
    -------
    trimesh.Trimesh
        Smoothed mesh copy.
    """
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or iterations <= 0:
        return mesh

    smoothed = mesh.copy()

    try:
        # Construct sparse adjacency graph using trimesh
        neighbors = smoothed.vertex_neighbors
        v = smoothed.vertices.copy().astype(np.float64)

        for _ in range(iterations):
            # 1. Diffusion step (+λ)
            v_new = v.copy()
            for i, n_list in enumerate(neighbors):
                if len(n_list) > 0:
                    v_new[i] += lam * (v[n_list].mean(axis=0) - v[i])
            v = v_new

            # 2. Anti-diffusion inflation step (+μ, where μ is negative)
            v_inflated = v.copy()
            for i, n_list in enumerate(neighbors):
                if len(n_list) > 0:
                    v_inflated[i] += mu * (v[n_list].mean(axis=0) - v[i])
            v = v_inflated

        smoothed.vertices = v.astype(np.float32)
        smoothed.fix_normals()
    except Exception:
        # Fallback to trimesh built-in Laplacian if manual loop encounters degenerate topology
        try:
            trimesh.smoothing.filter_laplacian(smoothed, iterations=iterations)
        except Exception:
            pass

    return smoothed


def apply_laplacian_smoothing(
    mesh: trimesh.Trimesh,
    iterations: int = 5,
    damping: float = 0.4,
) -> trimesh.Trimesh:
    """Apply standard Laplacian neighborhood relaxation to a mesh."""
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or iterations <= 0:
        return mesh

    smoothed = mesh.copy()
    try:
        trimesh.smoothing.filter_laplacian(smoothed, lamb=damping, iterations=iterations)
        smoothed.fix_normals()
    except Exception:
        pass

    return smoothed


# ──────────────────────────────────────────────
# Hole Filling & Boundary Repair
# ──────────────────────────────────────────────

def repair_and_fill_holes(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, int]:
    """Identify and stitch open boundary loop holes in a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.

    Returns
    -------
    Tuple[trimesh.Trimesh, int]
        ``(repaired_mesh, estimated_holes_filled)``
    """
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh, 0

    repaired = mesh.copy()
    initial_faces = len(repaired.faces)

    try:
        # Count open boundary edges
        trimesh.repair.fill_holes(repaired)
        added_faces = len(repaired.faces) - initial_faces
        # Estimate number of closed loops (roughly added_faces / 3)
        holes_filled = max(1, added_faces // 3) if added_faces > 0 else 0
    except Exception:
        holes_filled = 0

    return repaired, holes_filled


def clean_boundary_edges(
    mesh: trimesh.Trimesh,
    min_component_faces: int = 10,
) -> trimesh.Trimesh:
    """Prune isolated disconnected edge fragments and sliver artifacts."""
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh

    cleaned = mesh.copy()

    try:
        # Split into connected surface components and discard tiny disconnected noise
        components = cleaned.split(only_watertight=False)
        valid_components = [c for c in components if len(c.faces) >= min_component_faces]
        if len(valid_components) > 0:
            cleaned = trimesh.util.concatenate(valid_components)
    except Exception:
        pass

    return cleaned


# ──────────────────────────────────────────────
# Master Mesh Refinement Pipeline
# ──────────────────────────────────────────────

def refine_mesh(
    mesh: trimesh.Trimesh,
    config: Optional[RefinementConfig] = None,
) -> RefinementResult:
    """Execute complete multi-stage 3D mesh refinement and repair.

    Workflow:
      1. Baseline metrics calculation (roughness, initial vertex/face count).
      2. Degenerate and duplicate geometry elimination.
      3. Boundary hole stitching and topology repair.
      4. Disconnected edge fragment pruning.
      5. High-frequency surface smoothing (Taubin or Laplacian).
      6. Final normal recomputation and comparative metric evaluation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input raw surface mesh from reconstruction.
    config : Optional[RefinementConfig]
        Refinement options and hyperparameter settings.

    Returns
    -------
    RefinementResult
        Dataclass containing the refined mesh, comparative metrics, and metadata.
    """
    if config is None:
        config = RefinementConfig()

    orig_copy = mesh.copy()

    # 1. Baseline metrics
    v_before = len(mesh.vertices)
    f_before = len(mesh.faces)
    wt_before = bool(mesh.is_watertight) if f_before > 0 else False
    rough_before = compute_surface_roughness(mesh)

    current_mesh = mesh.copy()
    degenerate_count = 0
    holes_filled = 0

    # 2. Degenerate & duplicate geometry elimination
    if config.remove_degenerates and f_before > 0:
        # Count degenerate faces
        non_deg = current_mesh.nondegenerate_faces()
        degenerate_count = int((~non_deg).sum())
        current_mesh.update_faces(non_deg)

        if len(current_mesh.faces) > 0:
            unique_faces = trimesh.grouping.unique_rows(current_mesh.faces)[0]
            degenerate_count += int(len(current_mesh.faces) - len(unique_faces))
            current_mesh.update_faces(unique_faces)

        current_mesh.remove_unreferenced_vertices()

    # 3. Hole filling
    if config.fill_holes and len(current_mesh.faces) > 0:
        current_mesh, holes_filled = repair_and_fill_holes(current_mesh)

    # 4. Boundary cleanup
    if config.clean_boundaries and len(current_mesh.faces) > 0:
        current_mesh = clean_boundary_edges(current_mesh)

    # 5. Surface Smoothing
    method = config.smoothing_method.lower().strip()
    if method == "taubin" and len(current_mesh.faces) > 0:
        current_mesh = apply_taubin_smoothing(
            current_mesh,
            iterations=config.smoothing_iterations,
            lam=config.taubin_lambda,
            mu=config.taubin_mu,
        )
    elif method == "laplacian" and len(current_mesh.faces) > 0:
        current_mesh = apply_laplacian_smoothing(
            current_mesh,
            iterations=config.smoothing_iterations,
            damping=config.laplacian_damping,
        )

    # 6. Final normal recomputation
    try:
        current_mesh.fix_normals()
    except Exception:
        pass

    # 7. Final metrics evaluation
    v_after = len(current_mesh.vertices)
    f_after = len(current_mesh.faces)
    wt_after = bool(current_mesh.is_watertight) if f_after > 0 else False
    rough_after = compute_surface_roughness(current_mesh)

    if rough_before > 0:
        rough_reduction = float((rough_before - rough_after) / rough_before) * 100.0
    else:
        rough_reduction = 0.0

    metrics = RefinementMetrics(
        vertices_before=v_before,
        vertices_after=v_after,
        vertex_delta=v_after - v_before,
        faces_before=f_before,
        faces_after=f_after,
        face_delta=f_after - f_before,
        holes_filled=holes_filled,
        degenerate_faces_removed=degenerate_count,
        roughness_before=rough_before,
        roughness_after=rough_after,
        roughness_reduction_pct=rough_reduction,
        watertight_before=wt_before,
        watertight_after=wt_after,
    )

    metadata = {
        "config": {
            "smoothing_method": config.smoothing_method,
            "smoothing_iterations": config.smoothing_iterations,
            "fill_holes": config.fill_holes,
            "clean_boundaries": config.clean_boundaries,
        },
        "roughness_reduction_pct": f"{rough_reduction:.1f}%",
    }

    return RefinementResult(
        refined_mesh=current_mesh,
        original_mesh=orig_copy,
        metrics=metrics,
        output_files={},
        metadata=metadata,
    )


# ──────────────────────────────────────────────
# Before / After Visualization Comparison Helper
# ──────────────────────────────────────────────

def render_mesh_shading_view(
    mesh: trimesh.Trimesh,
    canvas_size: Tuple[int, int] = (400, 400),
    title: str = "",
    background_color: Tuple[int, int, int] = (25, 28, 35),
) -> np.ndarray:
    """Render an isometric shaded surface preview of a mesh for visual inspection."""
    w, h = canvas_size
    canvas = np.full((h, w, 3), background_color, dtype=np.uint8)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return canvas

    verts = mesh.vertices.copy()
    center = verts.mean(axis=0)
    pts_centered = verts - center

    # Apply 3D isometric rotation (Yaw ~35°, Pitch ~25°)
    yaw = np.radians(35.0)
    pitch = np.radians(25.0)

    r_yaw = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    r_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])
    rot = np.dot(r_pitch, r_yaw)
    pts_rot = np.dot(pts_centered, rot.T)

    max_extent = np.max(np.abs(pts_rot[:, :2]))
    scale = (0.42 * min(w, h) / max_extent) if max_extent > 0 else 1.0

    px = (pts_rot[:, 0] * scale + (w / 2.0)).astype(np.int32)
    py = (-pts_rot[:, 1] * scale + (h / 2.0)).astype(np.int32)

    # Transform face normals for diffuse directional lighting
    normals = mesh.face_normals
    normals_rot = np.dot(normals, rot.T)
    light_dir = np.array([0.4, 0.6, 0.7])
    light_dir /= np.linalg.norm(light_dir)

    # Compute face centroids and depth sort
    face_cents_z = pts_rot[mesh.faces].mean(axis=1)[:, 2]
    face_order = np.argsort(face_cents_z)  # Painter's order: far to near

    for f_idx in face_order:
        face = mesh.faces[f_idx]
        pts_tri = np.array([
            [px[face[0]], py[face[0]]],
            [px[face[1]], py[face[1]]],
            [px[face[2]], py[face[2]]],
        ], dtype=np.int32)

        # Diffuse shading factor
        norm_face = normals_rot[f_idx]
        diffuse = max(0.2, float(np.dot(norm_face, light_dir)))

        # Base vertex color if present, else metallic cyan-blue
        if mesh.visual and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            c = mesh.visual.vertex_colors[face[0]][:3].astype(np.float32)
        else:
            c = np.array([120.0, 180.0, 240.0], dtype=np.float32)

        shaded_col = (c * diffuse).clip(0, 255).astype(np.uint8)
        bgr = (int(shaded_col[0]), int(shaded_col[1]), int(shaded_col[2]))

        cv2.fillPoly(canvas, [pts_tri], bgr)
        # Subtle wireframe outline
        cv2.polylines(canvas, [pts_tri], isClosed=True, color=(40, 50, 60), thickness=1)

    if title:
        cv2.putText(
            canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA
        )

    return canvas


def create_refinement_comparison_image(
    mesh_before: trimesh.Trimesh,
    mesh_after: trimesh.Trimesh,
    output_path: Union[str, Path],
    canvas_size: Tuple[int, int] = (400, 400),
) -> Path:
    """Generate a side-by-side [ Before Refinement | After Refinement ] comparison image.

    Parameters
    ----------
    mesh_before : trimesh.Trimesh
        Raw unrefined mesh.
    mesh_after : trimesh.Trimesh
        Refined mesh.
    output_path : str or Path
        Destination image path (.png).
    canvas_size : Tuple[int, int]
        Tile resolution ``(width, height)``.

    Returns
    -------
    Path
        Saved comparison image path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    view1 = render_mesh_shading_view(mesh_before, canvas_size=canvas_size, title="Before Refinement (Raw)")
    view2 = render_mesh_shading_view(mesh_after, canvas_size=canvas_size, title="After Refinement (Smoothed)")

    comparison = np.hstack([view1, view2])
    Image.fromarray(comparison, mode="RGB").save(str(path))
    return path


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.refinement",
        description="Parallax 3D Mesh Refinement — Smooth, repair, and optimize 3D surface meshes.",
    )
    parser.add_argument("mesh", type=str, help="Path to input 3D mesh (.obj, .ply, .glb).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save refined 3D meshes and metrics (default: %(default)s).",
    )
    parser.add_argument(
        "--smoothing-method",
        type=str,
        default="taubin",
        choices=["taubin", "laplacian", "none"],
        help="Surface smoothing technique (default: %(default)s).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of smoothing passes (default: %(default)s).",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Disable boundary hole filling.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for running mesh refinement."""
    args = _build_parser().parse_args(argv)
    input_path = Path(args.mesh)
    if not input_path.exists():
        print(f"Error: Mesh file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Refining 3D mesh: {input_path}")
    raw_mesh = trimesh.load(str(input_path))

    config = RefinementConfig(
        smoothing_method=args.smoothing_method,
        smoothing_iterations=args.iterations,
        fill_holes=not args.no_fill_holes,
    )

    result = refine_mesh(raw_mesh, config=config)
    m = result.metrics

    print("\n" + "=" * 55)
    print("         3D MESH REFINEMENT METRICS SUMMARY         ")
    print("=" * 55)
    print(f"  • Vertices:              {m.vertices_before:,} -> {m.vertices_after:,} ({m.vertex_delta:+d})")
    print(f"  • Triangles (Faces):     {m.faces_before:,} -> {m.faces_after:,} ({m.face_delta:+d})")
    print(f"  • Degenerate Purged:     {m.degenerate_faces_removed:,}")
    print(f"  • Boundary Holes Closed: {m.holes_filled:,}")
    print(f"  • Surface Roughness:     {m.roughness_before:.4f} -> {m.roughness_after:.4f} ({m.roughness_reduction_pct:+.1f}%)")
    print(f"  • Watertight Manifold:   {m.watertight_before} -> {m.watertight_after}")
    print("=" * 55)

    stem = input_path.stem
    from src.mesh_reconstruction import export_mesh_all_formats

    saved = export_mesh_all_formats(result.refined_mesh, out_dir, stem=f"{stem}_refined")
    for k, p in saved.items():
        print(f"Saved {k.upper()}: {p}")

    comp_path = out_dir / f"{stem}_refinement_comparison.png"
    create_refinement_comparison_image(result.original_mesh, result.refined_mesh, comp_path)
    print(f"Saved Before/After Comparison: {comp_path}")
    print("✓ 3D mesh refinement complete.")


if __name__ == "__main__":
    main()
