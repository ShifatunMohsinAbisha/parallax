"""Unit tests for src.refinement (3D mesh & surface refinement, smoothing, hole repair)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.refinement import (
    RefinementConfig,
    RefinementMetrics,
    RefinementResult,
    apply_laplacian_smoothing,
    apply_taubin_smoothing,
    clean_boundary_edges,
    compute_surface_roughness,
    create_refinement_comparison_image,
    refine_mesh,
    repair_and_fill_holes,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def flat_plane_mesh() -> trimesh.Trimesh:
    """Create a clean 5×5 flat grid mesh (zero roughness)."""
    y, x = np.mgrid[:5, :5]
    z = np.zeros_like(x, dtype=np.float32)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    faces = []
    for r in range(4):
        for c in range(4):
            i0 = r * 5 + c
            i1 = i0 + 1
            i2 = (r + 1) * 5 + c
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)


@pytest.fixture()
def noisy_surface_mesh() -> trimesh.Trimesh:
    """Create a 10×10 grid with intentional high-frequency Z perturbations."""
    np.random.seed(42)
    y, x = np.mgrid[:10, :10]
    z = 0.2 * np.sin(x * 2.0) + 0.15 * np.random.randn(*x.shape)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    faces = []
    for r in range(9):
        for c in range(9):
            i0 = r * 10 + c
            i1 = i0 + 1
            i2 = (r + 1) * 10 + c
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)


@pytest.fixture()
def mesh_with_hole() -> trimesh.Trimesh:
    """Create a sphere with a missing face."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # Remove 2 adjacent faces to create an open hole
    faces = sphere.faces[:-2].copy()
    return trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)


# ──────────────────────────────────────────────
# Roughness & Metric Tests
# ──────────────────────────────────────────────

class TestSurfaceRoughness:
    def test_flat_plane_has_zero_roughness(self, flat_plane_mesh: trimesh.Trimesh) -> None:
        flat_plane_mesh.fix_normals()
        roughness = compute_surface_roughness(flat_plane_mesh)
        assert roughness == pytest.approx(0.0, abs=1e-5)

    def test_noisy_surface_has_positive_roughness(self, noisy_surface_mesh: trimesh.Trimesh) -> None:
        noisy_surface_mesh.fix_normals()
        roughness = compute_surface_roughness(noisy_surface_mesh)
        assert roughness > 0.1


# ──────────────────────────────────────────────
# Smoothing Tests (Taubin & Laplacian)
# ──────────────────────────────────────────────

class TestSmoothingAlgorithms:
    def test_taubin_smoothing_reduces_roughness(
        self, noisy_surface_mesh: trimesh.Trimesh
    ) -> None:
        noisy_surface_mesh.fix_normals()
        r_before = compute_surface_roughness(noisy_surface_mesh)

        smoothed = apply_taubin_smoothing(noisy_surface_mesh, iterations=8)
        r_after = compute_surface_roughness(smoothed)

        assert r_after < r_before
        assert len(smoothed.vertices) == len(noisy_surface_mesh.vertices)

    def test_laplacian_smoothing_reduces_roughness(
        self, noisy_surface_mesh: trimesh.Trimesh
    ) -> None:
        noisy_surface_mesh.fix_normals()
        r_before = compute_surface_roughness(noisy_surface_mesh)

        smoothed = apply_laplacian_smoothing(noisy_surface_mesh, iterations=5, damping=0.3)
        r_after = compute_surface_roughness(smoothed)

        assert r_after < r_before

    def test_smoothing_handles_empty_mesh(self) -> None:
        empty = trimesh.Trimesh()
        res = apply_taubin_smoothing(empty, iterations=5)
        assert len(res.vertices) == 0


# ──────────────────────────────────────────────
# Hole Filling & Boundary Repair Tests
# ──────────────────────────────────────────────

class TestHoleFillingAndBoundaryRepair:
    def test_repair_and_fill_holes(self, mesh_with_hole: trimesh.Trimesh) -> None:
        assert not mesh_with_hole.is_watertight

        repaired, holes_filled = repair_and_fill_holes(mesh_with_hole)
        assert len(repaired.faces) >= len(mesh_with_hole.faces)

    def test_clean_boundary_edges_prunes_sliver_component(self) -> None:
        # Main square mesh
        m1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        # Tiny disconnected 2-triangle noise component
        v_noise = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]], dtype=np.float32)
        f_noise = np.array([[0, 1, 2]], dtype=np.int64)
        m_noise = trimesh.Trimesh(vertices=v_noise, faces=f_noise)

        combined = trimesh.util.concatenate([m1, m_noise])
        assert len(combined.faces) == len(m1.faces) + 1

        cleaned = clean_boundary_edges(combined, min_component_faces=4)
        assert len(cleaned.faces) == len(m1.faces)


# ──────────────────────────────────────────────
# Master Refine Mesh Pipeline Tests
# ──────────────────────────────────────────────

class TestRefineMeshPipeline:
    def test_refine_mesh_returns_result_dataclass(
        self, noisy_surface_mesh: trimesh.Trimesh
    ) -> None:
        config = RefinementConfig(
            smoothing_method="taubin",
            smoothing_iterations=4,
            fill_holes=True,
            clean_boundaries=True,
        )
        res = refine_mesh(noisy_surface_mesh, config=config)

        assert isinstance(res, RefinementResult)
        assert isinstance(res.metrics, RefinementMetrics)
        assert res.metrics.roughness_after <= res.metrics.roughness_before
        assert res.metrics.roughness_reduction_pct >= 0.0

    def test_refine_mesh_laplacian(self, noisy_surface_mesh: trimesh.Trimesh) -> None:
        config = RefinementConfig(smoothing_method="laplacian", smoothing_iterations=3)
        res = refine_mesh(noisy_surface_mesh, config=config)
        assert res.metrics.roughness_after < res.metrics.roughness_before

    def test_create_refinement_comparison_image(
        self, noisy_surface_mesh: trimesh.Trimesh, tmp_path: Path
    ) -> None:
        smoothed = apply_taubin_smoothing(noisy_surface_mesh, iterations=4)
        out_img = tmp_path / "comparison.png"
        saved = create_refinement_comparison_image(noisy_surface_mesh, smoothed, out_img)

        assert saved.exists()
        assert saved.stat().st_size > 0
