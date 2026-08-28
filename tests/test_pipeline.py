"""Unit tests for src.pipeline (End-to-End single-image 3D reconstruction)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.pipeline import (
    PipelineResult,
    create_pipeline_overview_panel,
    run_pipeline,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def synthetic_toy_image_file(tmp_path: Path) -> Path:
    """Create a 128×128 synthetic test image with a distinct centered foreground object."""
    img = np.full((128, 128, 3), 235, dtype=np.uint8)
    # Shaded object in center
    cv2.circle(img, (64, 64), 35, (40, 140, 220), -1)
    cv2.circle(img, (55, 55), 18, (80, 180, 255), -1)
    cv2.circle(img, (50, 50), 6, (255, 255, 255), -1)

    p = tmp_path / "synthetic_object.png"
    Image.fromarray(img, mode="RGB").save(str(p))
    return p


# ──────────────────────────────────────────────
# Pipeline Integration Tests
# ──────────────────────────────────────────────

class TestEndToEndPipeline:
    def test_run_pipeline_end_to_end(
        self, synthetic_toy_image_file: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "pipeline_outputs"
        result = run_pipeline(
            image_path=synthetic_toy_image_file,
            output_dir=out_dir,
            target_size=(128, 128),
            fov_degrees=60.0,
            depth_method="geometric_shading",
            segmentation_method="saliency_grabcut",
            filter_outliers=True,
            save_intermediate_artifacts=True,
        )

        assert isinstance(result, PipelineResult)
        assert result.point_cloud.num_points > 0
        assert result.raw_point_cloud.num_points >= result.point_cloud.num_points
        assert result.intrinsics.width == 128
        assert result.intrinsics.height == 128

        # Verify output files exist
        assert "point_cloud_ply" in result.output_files
        assert "point_cloud_pcd" in result.output_files
        assert "view_3d" in result.output_files
        assert "pipeline_overview" in result.output_files

        for name, file_path in result.output_files.items():
            assert Path(file_path).exists(), f"Expected artifact {name} at {file_path} to exist"

        # Verify timing entries
        expected_stages = {
            "preprocessing",
            "segmentation",
            "depth_estimation",
            "geometry_unprojection",
            "point_cloud_processing",
            "total_pipeline",
        }
        assert expected_stages.issubset(set(result.timing.keys()))
        assert result.timing["total_pipeline"] > 0.0

    def test_create_pipeline_overview_panel(self, tmp_path: Path) -> None:
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        seg = np.zeros((100, 100, 3), dtype=np.uint8)
        depth_hm = np.zeros((100, 100, 3), dtype=np.uint8)

        dummy_3d = tmp_path / "dummy_3d.png"
        Image.fromarray(np.full((100, 100, 3), 50, dtype=np.uint8)).save(str(dummy_3d))

        panel = create_pipeline_overview_panel(
            rgb, seg, depth_hm, point_cloud_view_path=dummy_3d, canvas_height=100
        )
        assert panel.shape == (100, 400, 3)
        assert panel.dtype == np.uint8
