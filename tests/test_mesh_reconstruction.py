"""Unit tests for src.mesh_reconstruction (3D surface mesh generation, repair, and export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.geometry import CameraIntrinsics, PointCloud, depth_to_point_cloud
from src.mesh_reconstruction import (
    MeshReconstructionResult,
    clean_mesh,
    export_mesh,
    export_mesh_all_formats,
    reconstruct_grid_mesh,
    reconstruct_surface_mesh,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def synthetic_point_cloud() -> PointCloud:
    """Create a structured 10×10 depth grid unprojected into a PointCloud."""
    y, x = np.mgrid[:10, :10]
    # Smooth, gently curved depth surface (adjacent steps < 0.05)
    depth = 2.0 + 0.1 * ((x - 4.5) ** 2 + (y - 4.5) ** 2) / 40.0
    depth = depth.astype(np.float32)

    rgb = np.full((10, 10, 3), 180, dtype=np.uint8)
    intrinsics = CameraIntrinsics(
        fx=10.0, fy=10.0, cx=4.5, cy=4.5, width=10, height=10, fov_degrees=60.0
    )

    return depth_to_point_cloud(depth, intrinsics=intrinsics, rgb=rgb)


@pytest.fixture()
def mesh_with_degenerates() -> trimesh.Trimesh:
    """Create a simple quad mesh with an intentional duplicate face and a degenerate face."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],  # Isolated unreferenced vertex
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 2],  # Duplicate face
        [0, 0, 1],  # Degenerate zero-area face (collinear vertices)
    ], dtype=np.int64)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ──────────────────────────────────────────────
# Mesh Reconstruction Tests
# ──────────────────────────────────────────────

class TestMeshReconstruction:
    def test_reconstruct_grid_mesh(self, synthetic_point_cloud: PointCloud) -> None:
        mesh = reconstruct_grid_mesh(synthetic_point_cloud)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == 100
        assert len(mesh.faces) > 0
        # 10×10 grid has 9×9 = 81 quads = 162 triangles
        assert len(mesh.faces) == 162

    def test_reconstruct_grid_mesh_empty(self) -> None:
        empty_pcd = PointCloud(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=None,
            normals=None,
            intrinsics=CameraIntrinsics(10, 10, 5, 5, 10, 10, 60),
            metadata={},
        )
        mesh = reconstruct_grid_mesh(empty_pcd)
        assert len(mesh.vertices) == 0
        assert len(mesh.faces) == 0

    def test_reconstruct_surface_mesh_auto(self, synthetic_point_cloud: PointCloud) -> None:
        res = reconstruct_surface_mesh(synthetic_point_cloud, method="auto", clean=True)

        assert isinstance(res, MeshReconstructionResult)
        assert res.num_vertices > 0
        assert res.num_faces > 0
        assert "grid" in res.method or "poisson" in res.method

    def test_reconstruct_surface_mesh_invalid_method_raises(
        self, synthetic_point_cloud: PointCloud
    ) -> None:
        with pytest.raises(ValueError, match="Unknown reconstruction method"):
            reconstruct_surface_mesh(synthetic_point_cloud, method="non_existent_method")


# ──────────────────────────────────────────────
# Mesh Cleanup & Repair Tests
# ──────────────────────────────────────────────

class TestMeshCleanup:
    def test_clean_mesh_removes_degenerates_and_unreferenced(
        self, mesh_with_degenerates: trimesh.Trimesh
    ) -> None:
        assert len(mesh_with_degenerates.faces) == 4
        assert len(mesh_with_degenerates.vertices) == 5

        cleaned = clean_mesh(mesh_with_degenerates, remove_degenerate=True)

        # Duplicate and degenerate faces removed (2 valid faces left)
        assert len(cleaned.faces) == 2
        # Isolated 5th vertex removed (4 vertices left)
        assert len(cleaned.vertices) == 4


# ──────────────────────────────────────────────
# Multi-Format Mesh Exporters (.obj, .ply, .glb)
# ──────────────────────────────────────────────

class TestMeshExporters:
    def test_export_mesh_obj(
        self, synthetic_point_cloud: PointCloud, tmp_path: Path
    ) -> None:
        res = reconstruct_surface_mesh(synthetic_point_cloud, method="grid")
        obj_file = tmp_path / "model.obj"
        export_mesh(res.mesh, obj_file, file_type="obj")

        assert obj_file.exists()
        content = obj_file.read_text(encoding="utf-8")
        assert "v " in content
        assert "f " in content

    def test_export_mesh_ply(
        self, synthetic_point_cloud: PointCloud, tmp_path: Path
    ) -> None:
        res = reconstruct_surface_mesh(synthetic_point_cloud, method="grid")
        ply_file = tmp_path / "model.ply"
        export_mesh(res.mesh, ply_file, file_type="ply")

        assert ply_file.exists()
        assert ply_file.stat().st_size > 0

    def test_export_mesh_glb(
        self, synthetic_point_cloud: PointCloud, tmp_path: Path
    ) -> None:
        res = reconstruct_surface_mesh(synthetic_point_cloud, method="grid")
        glb_file = tmp_path / "model.glb"
        export_mesh(res.mesh, glb_file, file_type="glb")

        assert glb_file.exists()
        # Binary glTF starts with magic bytes "glTF"
        with open(glb_file, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF"

    def test_export_mesh_all_formats(
        self, synthetic_point_cloud: PointCloud, tmp_path: Path
    ) -> None:
        res = reconstruct_surface_mesh(synthetic_point_cloud, method="grid")
        exported = export_mesh_all_formats(res.mesh, tmp_path, stem="test_reconstruction")

        assert "mesh_obj" in exported and exported["mesh_obj"].exists()
        assert "mesh_ply" in exported and exported["mesh_ply"].exists()
        assert "mesh_glb" in exported and exported["mesh_glb"].exists()
