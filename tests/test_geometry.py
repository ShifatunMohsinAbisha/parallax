"""Unit tests for src.geometry (Pinhole camera model & 2D-to-3D back-projection)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.geometry import (
    CameraIntrinsics,
    PointCloud,
    depth_to_point_cloud,
    estimate_camera_intrinsics,
    estimate_surface_normals_from_depth,
    project_3d_to_pixel,
    save_point_cloud_obj,
    save_point_cloud_ply,
    unproject_pixel,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def fixed_intrinsics() -> CameraIntrinsics:
    """Create exact, known camera intrinsics for deterministic mathematical testing."""
    return CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        fov_degrees=60.0,
    )


# ──────────────────────────────────────────────
# Camera Intrinsics Tests
# ──────────────────────────────────────────────

class TestCameraIntrinsics:
    def test_estimate_camera_intrinsics_defaults(self) -> None:
        intrinsics = estimate_camera_intrinsics(image_size=(480, 640), fov_degrees=60.0)
        assert intrinsics.width == 640
        assert intrinsics.height == 480
        assert intrinsics.cx == (640 - 1) / 2.0
        assert intrinsics.cy == (480 - 1) / 2.0
        # For FOV 60 deg, fx = 640 / (2 * tan(30 deg)) = 640 / (2 * 0.57735) ≈ 554.256
        expected_fx = 640.0 / (2.0 * np.tan(np.radians(30.0)))
        assert abs(intrinsics.fx - expected_fx) < 1e-3
        assert intrinsics.fx == intrinsics.fy

    def test_intrinsics_to_matrix_and_inverse(self, fixed_intrinsics: CameraIntrinsics) -> None:
        K = fixed_intrinsics.to_matrix()
        K_inv = fixed_intrinsics.inv_matrix()

        assert K.shape == (3, 3)
        assert K[0, 0] == 100.0  # fx
        assert K[1, 1] == 100.0  # fy
        assert K[0, 2] == 50.0   # cx
        assert K[1, 2] == 50.0   # cy

        # Verify K * K_inv = Identity
        identity = np.dot(K, K_inv)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-6)


# ──────────────────────────────────────────────
# Pinhole Unprojection & Forward Projection Tests
# ──────────────────────────────────────────────

class TestPinholeProjections:
    def test_unproject_optical_center_is_on_optical_axis(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        # Optical center pixel (u=50, v=50) at depth Z=2.5 should unproject to (0, 0, 2.5)
        point = unproject_pixel(50.0, 50.0, depth=2.5, intrinsics=fixed_intrinsics)
        assert point == (0.0, 0.0, 2.5)

    def test_unproject_known_offset_pixels(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        # Pixel at (u=100, v=50) with Z=2.0:
        # X = (100 - 50) * 2.0 / 100 = 1.0, Y = 0.0, Z = 2.0
        p_right = unproject_pixel(100.0, 50.0, depth=2.0, intrinsics=fixed_intrinsics)
        assert p_right == (1.0, 0.0, 2.0)

        # Pixel at (u=50, v=100) with Z=2.0:
        # Y-down: Y = (100 - 50) * 2.0 / 100 = 1.0
        p_down = unproject_pixel(50.0, 100.0, depth=2.0, intrinsics=fixed_intrinsics, y_up=False)
        assert p_down == (0.0, 1.0, 2.0)

        # Y-up: Y = -(100 - 50) * 2.0 / 100 = -1.0
        p_up = unproject_pixel(50.0, 100.0, depth=2.0, intrinsics=fixed_intrinsics, y_up=True)
        assert p_up == (0.0, -1.0, 2.0)

    def test_forward_and_inverse_projection_roundtrip(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        # Project 3D -> 2D -> 3D
        orig_3d = (1.25, -0.75, 3.5)
        u, v = project_3d_to_pixel(orig_3d, fixed_intrinsics, y_up=True)
        recovered_3d = unproject_pixel(u, v, depth=orig_3d[2], intrinsics=fixed_intrinsics, y_up=True)

        np.testing.assert_allclose(recovered_3d, orig_3d, atol=1e-5)


# ──────────────────────────────────────────────
# Dense Point Cloud Unprojection with Synthetic Depth
# ──────────────────────────────────────────────

class TestSyntheticDepthPointCloud:
    def test_known_flat_plane_unprojection(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        """Create a synthetic 10×10 flat depth plane at constant Z=3.0 with known 3D points."""
        depth_plane = np.full((10, 10), 3.0, dtype=np.float32)
        rgb_grid = np.full((10, 10, 3), 200, dtype=np.uint8)

        # Set custom intrinsics matching the 10×10 resolution
        intrinsics_10 = CameraIntrinsics(
            fx=10.0, fy=10.0, cx=4.5, cy=4.5, width=10, height=10, fov_degrees=60.0
        )

        pcd = depth_to_point_cloud(
            depth_map=depth_plane,
            intrinsics=intrinsics_10,
            rgb=rgb_grid,
            compute_normals=True,
            y_up=True,
        )

        assert pcd.num_points == 100
        assert pcd.points.shape == (100, 3)
        # All Z values must exactly equal 3.0
        np.testing.assert_allclose(pcd.points[:, 2], 3.0, atol=1e-5)

        # Check center pixel (u=4, v=4) unprojected position:
        # X = (4 - 4.5) * 3.0 / 10.0 = -0.15
        # Y = -(4 - 4.5) * 3.0 / 10.0 = +0.15
        # Z = 3.0
        center_pt = pcd.points[4 * 10 + 4]  # row 4, col 4
        np.testing.assert_allclose(center_pt, [-0.15, 0.15, 3.0], atol=1e-5)

        # Check surface normal of flat plane (should face camera, normal = [0, 0, 1])
        if pcd.normals is not None:
            # Interior normals should be oriented along positive Z
            interior_normals = pcd.normals.reshape(10, 10, 3)[2:8, 2:8]
            assert np.all(interior_normals[:, :, 2] > 0.9)

    def test_masked_object_unprojection(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        """Verify only masked object pixels are included in the point cloud."""
        depth = np.full((10, 10), 2.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.uint8)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)

        # Mask exactly 4 center pixels
        mask[4:6, 4:6] = 1
        rgb[4:6, 4:6] = [255, 0, 128]  # Distinct magenta color

        intrinsics_10 = CameraIntrinsics(
            fx=10.0, fy=10.0, cx=4.5, cy=4.5, width=10, height=10, fov_degrees=60.0
        )

        pcd = depth_to_point_cloud(
            depth_map=depth,
            intrinsics=intrinsics_10,
            rgb=rgb,
            mask=mask,
        )

        # Only the 4 masked points should exist
        assert pcd.num_points == 4
        assert pcd.colors is not None
        assert pcd.colors.shape == (4, 3)
        np.testing.assert_array_equal(pcd.colors, np.tile([255, 0, 128], (4, 1)))

    def test_point_cloud_bounds_and_center(
        self, fixed_intrinsics: CameraIntrinsics
    ) -> None:
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        pcd = depth_to_point_cloud(depth, intrinsics=fixed_intrinsics)

        min_b, max_b = pcd.bounds
        center = pcd.center

        assert abs(center[0]) < 0.1
        assert abs(center[1]) < 0.1
        assert abs(center[2] - 4.0) < 1e-4


# ──────────────────────────────────────────────
# PLY & OBJ Serialization Tests
# ──────────────────────────────────────────────

class TestPointCloudExporters:
    def test_save_point_cloud_ply_ascii(
        self, fixed_intrinsics: CameraIntrinsics, tmp_path: Path
    ) -> None:
        depth = np.full((10, 10), 1.0, dtype=np.float32)
        rgb = np.full((10, 10, 3), 255, dtype=np.uint8)
        pcd = depth_to_point_cloud(depth, intrinsics=fixed_intrinsics, rgb=rgb)

        ply_file = tmp_path / "test.ply"
        save_point_cloud_ply(pcd, ply_file, binary=False)

        assert ply_file.exists()
        content = ply_file.read_text(encoding="ascii")
        assert "ply" in content
        assert "element vertex 100" in content
        assert "property float x" in content
        assert "property uchar red" in content
        assert "end_header" in content

    def test_save_point_cloud_ply_binary(
        self, fixed_intrinsics: CameraIntrinsics, tmp_path: Path
    ) -> None:
        depth = np.full((10, 10), 1.0, dtype=np.float32)
        pcd = depth_to_point_cloud(depth, intrinsics=fixed_intrinsics)

        ply_file = tmp_path / "test_bin.ply"
        save_point_cloud_ply(pcd, ply_file, binary=True)

        assert ply_file.exists()
        assert ply_file.stat().st_size > 0

    def test_save_point_cloud_obj(
        self, fixed_intrinsics: CameraIntrinsics, tmp_path: Path
    ) -> None:
        depth = np.full((5, 5), 2.0, dtype=np.float32)
        pcd = depth_to_point_cloud(depth, intrinsics=fixed_intrinsics)

        obj_file = tmp_path / "test.obj"
        save_point_cloud_obj(pcd, obj_file)

        assert obj_file.exists()
        lines = obj_file.read_text(encoding="utf-8").strip().splitlines()
        vertex_lines = [l for l in lines if l.startswith("v ")]
        assert len(vertex_lines) == 25
