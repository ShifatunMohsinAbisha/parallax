"""Unit tests for src.visualize (Interactive 3D HTML/Three.js web viewer generator)."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from src.visualize import generate_html_viewer, main


@pytest.fixture()
def sample_glb_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic .glb file for testing viewer generation."""
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    glb_path = tmp_path / "test_model.glb"
    mesh.export(str(glb_path), file_type="glb")
    return glb_path


class TestVisualizeHTMLViewer:
    def test_generate_html_viewer_embedded(self, sample_glb_file: Path, tmp_path: Path) -> None:
        out_html = tmp_path / "test_viewer.html"
        generated = generate_html_viewer(
            mesh_path=sample_glb_file,
            output_html_path=out_html,
            title="Custom Test Viewer",
            embed_model=True,
        )

        assert generated.exists()
        content = generated.read_text(encoding="utf-8")
        assert "Custom Test Viewer" in content
        assert "three.min.js" in content
        assert "OrbitControls.js" in content
        assert "GLTFLoader.js" in content
        assert "data:model/gltf-binary;base64," in content
        assert "THREE.OrbitControls" in content

    def test_generate_html_viewer_relative_path(
        self, sample_glb_file: Path, tmp_path: Path
    ) -> None:
        out_html = tmp_path / "relative_viewer.html"
        generated = generate_html_viewer(
            mesh_path=sample_glb_file,
            output_html_path=out_html,
            embed_model=False,
        )

        assert generated.exists()
        content = generated.read_text(encoding="utf-8")
        assert "data:model/gltf-binary" not in content
        assert sample_glb_file.name in content

    def test_generate_html_viewer_default_output_path(
        self, sample_glb_file: Path
    ) -> None:
        generated = generate_html_viewer(mesh_path=sample_glb_file)
        expected_path = sample_glb_file.parent / "viewer.html"
        assert generated == expected_path
        assert generated.exists()

    def test_generate_html_viewer_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_html_viewer(tmp_path / "non_existent.glb")

    def test_cli_visualize(self, sample_glb_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        out_html = tmp_path / "cli_viewer.html"
        main([str(sample_glb_file), "--output", str(out_html), "--title", "CLI Test"])

        assert out_html.exists()
        captured = capsys.readouterr()
        assert "Interactive 3D Web Viewer Generated" in captured.out
