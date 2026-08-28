"""Unit tests for src.evaluation (Quantitative 3D metrics, Chamfer distance, Point-to-Mesh distance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.evaluation import (
    align_point_clouds,
    compute_chamfer_distance,
    compute_point_to_mesh_distance,
    create_primitive_mesh,
    generate_all_primitive_meshes,
    render_mesh_to_image,
    sample_mesh_surface,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def random_point_cloud() -> np.ndarray:
    """Create a deterministic random 3D point cloud."""
    np.random.seed(42)
    return np.random.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)


@pytest.fixture()
def unit_cube_mesh() -> trimesh.Trimesh:
    """Create a 1x1x1 unit cube mesh."""
    return create_primitive_mesh("cube", scale=1.0)


@pytest.fixture()
def unit_sphere_mesh() -> trimesh.Trimesh:
    """Create a unit sphere mesh."""
    return create_primitive_mesh("sphere", scale=1.0)


# ──────────────────────────────────────────────
# Chamfer Distance Tests
# ──────────────────────────────────────────────

class TestChamferDistance:
    def test_chamfer_distance_identical_point_clouds_is_zero(
        self, random_point_cloud: np.ndarray
    ) -> None:
        cd_l2, cd_l1 = compute_chamfer_distance(random_point_cloud, random_point_cloud)

        assert cd_l2 == pytest.approx(0.0, abs=1e-6)
        assert cd_l1 == pytest.approx(0.0, abs=1e-6)

    def test_chamfer_distance_increases_with_shift(self) -> None:
        # Create a grid where points are well separated (spacing = 1.0)
        grid_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)

        shift_small = grid_points + np.array([0.1, 0.0, 0.0], dtype=np.float32)
        shift_large = grid_points + np.array([0.5, 0.0, 0.0], dtype=np.float32)

        cd_l2_small, cd_l1_small = compute_chamfer_distance(grid_points, shift_small)
        cd_l2_large, cd_l1_large = compute_chamfer_distance(grid_points, shift_large)

        # Distance should be positive and strictly monotonically increasing with shift magnitude
        assert cd_l2_small > 0.0
        assert cd_l2_large > cd_l2_small
        assert cd_l1_large > cd_l1_small

        # Exact analytic verification for pure translation delta = 0.1 on isolated points:
        assert cd_l2_small == pytest.approx(2.0 * (0.1 ** 2), rel=1e-3)
        assert cd_l1_small == pytest.approx(0.1, rel=1e-3)

    def test_chamfer_distance_empty_clouds(self) -> None:
        empty = np.zeros((0, 3), dtype=np.float32)
        pts = np.ones((10, 3), dtype=np.float32)
        cd_l2, _ = compute_chamfer_distance(empty, pts)
        assert np.isinf(cd_l2)


# ──────────────────────────────────────────────
# Point-to-Mesh Distance Tests
# ──────────────────────────────────────────────

class TestPointToMeshDistance:
    def test_points_on_surface_have_near_zero_distance(
        self, unit_cube_mesh: trimesh.Trimesh
    ) -> None:
        # Sample points strictly on the surface of the cube
        surface_points = sample_mesh_surface(unit_cube_mesh, num_samples=1000)
        mean_dist, p95_dist = compute_point_to_mesh_distance(
            surface_points, unit_cube_mesh, dense_surface_samples=20000
        )

        # Distance should be negligible (< 0.05)
        assert mean_dist < 0.05
        assert p95_dist < 0.08

    def test_point_to_mesh_distance_increases_with_offset(
        self, unit_sphere_mesh: trimesh.Trimesh
    ) -> None:
        surface_pts = sample_mesh_surface(unit_sphere_mesh, num_samples=500)
        # Shift outward along radius
        shifted_pts = surface_pts * 1.5

        mean_dist_on_surface, _ = compute_point_to_mesh_distance(surface_pts, unit_sphere_mesh)
        mean_dist_shifted, _ = compute_point_to_mesh_distance(shifted_pts, unit_sphere_mesh)

        assert mean_dist_shifted > mean_dist_on_surface


# ──────────────────────────────────────────────
# Primitive Mesh Generation & Rendering Tests
# ──────────────────────────────────────────────

class TestPrimitivesAndRendering:
    def test_generate_all_primitive_meshes(self) -> None:
        primitives = generate_all_primitive_meshes(scale=1.0)
        assert set(primitives.keys()) == {"cube", "sphere", "cylinder", "cone"}

        for name, mesh in primitives.items():
            assert isinstance(mesh, trimesh.Trimesh)
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0
            assert mesh.is_watertight

    def test_render_mesh_to_image_output_format(
        self, unit_cube_mesh: trimesh.Trimesh
    ) -> None:
        img = render_mesh_to_image(unit_cube_mesh, image_size=(128, 128))
        assert isinstance(img, np.ndarray)
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8
        # Ensure image is not purely background (some pixels changed by rendered mesh)
        assert not np.all(img == 240)

    def test_align_point_clouds(self) -> None:
        p1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        p2 = np.array([[10.0, 10.0, 10.0], [14.0, 10.0, 10.0]], dtype=np.float32)

        a1, a2 = align_point_clouds(p1, p2)
        assert np.allclose(a1.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(a2.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-5)
