"""Unit tests for src.point_cloud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.geometry import CameraIntrinsics, PointCloud
from src.point_cloud import (
    PointCloudProcessingResult,
    clean_point_cloud,
    export_point_cloud_pcd,
    export_point_cloud_ply,
    remove_radius_outliers_numpy,
    remove_statistical_outliers_numpy,
    render_point_cloud_views,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def sample_cloud_with_outlier() -> PointCloud:
    """Create a dense cluster of points around origin plus 1 far distant outlier."""
    # 50 points clustered tightly around origin
    rng = np.random.default_rng(42)
    cluster = rng.normal(loc=0.0, scale=0.05, size=(50, 3)).astype(np.float32)
    # 1 distant outlier at (10.0, 10.0, 10.0)
    outlier = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    points = np.vstack([cluster, outlier])

    colors = np.full((len(points), 3), 200, dtype=np.uint8)
    colors[-1] = [255, 0, 0]  # Red outlier

    intrinsics = CameraIntrinsics(
        fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, fov_degrees=60.0
    )

    return PointCloud(
        points=points,
        colors=colors,
        normals=None,
        intrinsics=intrinsics,
        metadata={},
    )


# ──────────────────────────────────────────────
# Outlier Removal Tests
# ──────────────────────────────────────────────

class TestOutlierRemoval:
    def test_statistical_outlier_removal_numpy_filters_distant_point(
        self, sample_cloud_with_outlier: PointCloud
    ) -> None:
        pts = sample_cloud_with_outlier.points
        indices, mask = remove_statistical_outliers_numpy(pts, nb_neighbors=10, std_ratio=1.5)

        assert len(indices) == 50
        assert mask[-1] == False  # Outlier index 50 rejected
        assert mask[0] == True    # Inlier cluster preserved

    def test_radius_outlier_removal_numpy_filters_solitary_point(
        self, sample_cloud_with_outlier: PointCloud
    ) -> None:
        pts = sample_cloud_with_outlier.points
        indices, mask = remove_radius_outliers_numpy(pts, radius=0.2, min_neighbors=5)

        assert mask[-1] == False  # Outlier rejected
        assert mask[:40].all()    # Core cluster inliers kept

    def test_clean_point_cloud_empty(self) -> None:
        empty_pcd = PointCloud(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=None,
            normals=None,
            intrinsics=CameraIntrinsics(100, 100, 50, 50, 100, 100, 60),
            metadata={},
        )
        res = clean_point_cloud(empty_pcd)
        assert isinstance(res, PointCloudProcessingResult)
        assert res.num_original == 0
        assert res.num_retained == 0

    def test_clean_point_cloud_returns_result_dataclass(
        self, sample_cloud_with_outlier: PointCloud
    ) -> None:
        res = clean_point_cloud(sample_cloud_with_outlier, nb_neighbors=10, std_ratio=1.5)

        assert isinstance(res, PointCloudProcessingResult)
        assert res.num_original == 51
        assert res.num_retained == 50
        assert res.outlier_ratio > 0.0
        assert res.point_cloud.num_points == 50


# ──────────────────────────────────────────────
# PCD & PLY Exporter Tests
# ──────────────────────────────────────────────

class TestPointCloudExports:
    def test_export_point_cloud_pcd_ascii(
        self, sample_cloud_with_outlier: PointCloud, tmp_path: Path
    ) -> None:
        pcd_path = tmp_path / "test.pcd"
        export_point_cloud_pcd(sample_cloud_with_outlier, pcd_path, binary=False)

        assert pcd_path.exists()
        content = pcd_path.read_text(encoding="ascii")
        assert "VERSION 0.7" in content
        assert "FIELDS x y z rgb" in content
        assert "POINTS 51" in content
        assert "DATA ascii" in content

    def test_export_point_cloud_pcd_binary(
        self, sample_cloud_with_outlier: PointCloud, tmp_path: Path
    ) -> None:
        pcd_path = tmp_path / "test_bin.pcd"
        export_point_cloud_pcd(sample_cloud_with_outlier, pcd_path, binary=True)

        assert pcd_path.exists()
        assert pcd_path.stat().st_size > 0

    def test_export_point_cloud_ply(
        self, sample_cloud_with_outlier: PointCloud, tmp_path: Path
    ) -> None:
        ply_path = tmp_path / "test.ply"
        export_point_cloud_ply(sample_cloud_with_outlier, ply_path)

        assert ply_path.exists()
        assert "ply" in ply_path.read_text(encoding="ascii")


# ──────────────────────────────────────────────
# Visualization Screenshot Tests
# ──────────────────────────────────────────────

class TestPointCloudVisualization:
    def test_render_point_cloud_views_creates_image(
        self, sample_cloud_with_outlier: PointCloud, tmp_path: Path
    ) -> None:
        out_img = tmp_path / "render_3d.png"
        res_path = render_point_cloud_views(
            sample_cloud_with_outlier,
            out_img,
            canvas_size=(300, 300),
            point_size=2,
        )

        assert res_path.exists()
        from PIL import Image
        img = Image.open(str(res_path))
        assert img.size == (300, 300)
        assert img.mode == "RGB"
