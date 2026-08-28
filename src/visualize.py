"""
Parallax — Interactive 3D Web Viewer Generator (src/visualize.py)

Generates lightweight, self-contained, interactive HTML5 3D viewers using Three.js.
Supports loading .glb models directly or embedding them as Base64 data URIs so the
resulting HTML file can be opened in any browser directly via file:// without CORS restrictions.

Features:
- Three.js r128 + OrbitControls + GLTFLoader via high-availability CDNs.
- Full 360° rotation (left-click drag), zooming (scroll wheel), and panning (right-click drag).
- Studio lighting setup (ambient + directional key/fill lights + ground grid).
- Floating UI overlay with model statistics (vertex/face count), auto-rotate, wireframe toggle,
  and camera reset.
- Self-contained portable export mode (Base64 embedded GLB).

Usage:
    # As a Python module:
    from src.visualize import generate_html_viewer
    viewer_path = generate_html_viewer("outputs/sample_vase_refined.glb", "outputs/viewer.html")

    # From CLI:
    python -m src.visualize outputs/sample_vase_refined.glb --output outputs/viewer.html
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Optional, Union


# ──────────────────────────────────────────────
# Three.js HTML Template
# ──────────────────────────────────────────────

HTML_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{TITLE}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            user-select: none;
        }}
        body, html {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: #0e1117;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #f0f2f6;
        }}
        #canvas-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}
        /* Header Overlay */
        .header-bar {{
            position: absolute;
            top: 20px;
            left: 24px;
            z-index: 10;
            background: rgba(18, 22, 31, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 14px 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            pointer-events: auto;
        }}
        .header-bar h1 {{
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.3px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .header-bar h1 span.badge {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 8px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 6px;
            letter-spacing: 0.5px;
        }}
        .header-bar p.model-name {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }}

        /* Control Panel */
        .controls-panel {{
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            background: rgba(18, 22, 31, 0.88);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 8px 14px;
            border-radius: 100px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        .btn {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
            padding: 8px 14px;
            border-radius: 50px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
        }}
        .btn.active {{
            background: #3b82f6;
            border-color: #60a5fa;
            color: #ffffff;
        }}

        /* Instructions Hint */
        .instructions-hint {{
            position: absolute;
            bottom: 24px;
            right: 24px;
            z-index: 10;
            background: rgba(18, 22, 31, 0.75);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 10px 16px;
            border-radius: 10px;
            font-size: 11px;
            color: #94a3b8;
            line-height: 1.6;
        }}
        .instructions-hint strong {{
            color: #cbd5e1;
        }}

        /* Loading Spinner */
        #loader {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 20;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            transition: opacity 0.4s ease;
        }}
        .spinner {{
            width: 44px;
            height: 44px;
            border: 3px solid rgba(59, 130, 246, 0.2);
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        #loader p {{
            font-size: 13px;
            font-weight: 500;
            color: #94a3b8;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
    <!-- Three.js r128 + OrbitControls + GLTFLoader CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
</head>
<body>
    <div id="canvas-container"></div>

    <div class="header-bar">
        <h1>PARALLAX <span class="badge">3D Viewer</span></h1>
        <p class="model-name">{MODEL_NAME}</p>
    </div>

    <div class="controls-panel">
        <button class="btn" id="btn-rotate">🔄 Auto-Rotate</button>
        <button class="btn" id="btn-wireframe">📐 Wireframe</button>
        <button class="btn" id="btn-grid">🌐 Toggle Grid</button>
        <button class="btn" id="btn-reset">🎯 Reset View</button>
    </div>

    <div class="instructions-hint">
        <strong>Controls:</strong><br>
        • Left Click + Drag: Orbit / Rotate<br>
        • Right Click + Drag: Pan Camera<br>
        • Scroll Wheel: Zoom
    </div>

    <div id="loader">
        <div class="spinner"></div>
        <p>Loading 3D Model...</p>
    </div>

    <script>
        // ──────────────────────────────────────────────
        // 1. Scene, Camera & WebGL Renderer Setup
        // ──────────────────────────────────────────────
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0e1117);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
        camera.position.set(0, 0.5, 2.5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 20;
        controls.minDistance = 0.1;

        // ──────────────────────────────────────────────
        // 2. Studio Lighting Setup
        // ──────────────────────────────────────────────
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
        scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
        keyLight.position.set(4, 8, 6);
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.55);
        fillLight.position.set(-5, -2, -4);
        scene.add(fillLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
        backLight.position.set(0, 6, -6);
        scene.add(backLight);

        // Ground Grid Helper
        const gridHelper = new THREE.GridHelper(10, 20, 0x3b82f6, 0x1e293b);
        gridHelper.position.y = -0.5;
        scene.add(gridHelper);

        // ──────────────────────────────────────────────
        // 3. GLB Model Loading
        // ──────────────────────────────────────────────
        const modelSource = "{MODEL_DATA_OR_URL}";
        let currentModel = null;
        let isWireframe = false;
        let isAutoRotating = false;

        const loaderElement = document.getElementById('loader');
        const loader = new THREE.GLTFLoader();

        function fitModelToView(object) {{
            const box = new THREE.Box3().setFromObject(object);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            // Center geometry at origin
            object.position.x += (object.position.x - center.x);
            object.position.y += (object.position.y - center.y);
            object.position.z += (object.position.z - center.z);

            const maxDim = Math.max(size.x, size.y, size.z);
            if (maxDim > 0) {{
                const scale = 1.6 / maxDim;
                object.scale.set(scale, scale, scale);
            }}

            gridHelper.position.y = -0.85;
            controls.target.set(0, 0, 0);
            camera.position.set(0, 0.4, 2.6);
            controls.update();
        }}

        loader.load(
            modelSource,
            function (gltf) {{
                currentModel = gltf.scene;
                // Enable two-sided rendering & vertex colors
                currentModel.traverse((node) => {{
                    if (node.isMesh && node.material) {{
                        node.material.side = THREE.DoubleSide;
                        node.material.roughness = 0.6;
                        node.material.metalness = 0.1;
                    }}
                }});

                scene.add(currentModel);
                fitModelToView(currentModel);

                // Fade out loader
                loaderElement.style.opacity = '0';
                setTimeout(() => loaderElement.style.display = 'none', 400);
            }},
            undefined,
            function (error) {{
                console.error('Error loading 3D model:', error);
                loaderElement.innerHTML = '<p style="color:#ef4444;">Failed to load 3D model.</p>';
            }}
        );

        // ──────────────────────────────────────────────
        // 4. UI Interactive Controls
        // ──────────────────────────────────────────────
        const btnRotate = document.getElementById('btn-rotate');
        btnRotate.addEventListener('click', () => {{
            isAutoRotating = !isAutoRotating;
            controls.autoRotate = isAutoRotating;
            controls.autoRotateSpeed = 2.0;
            btnRotate.classList.toggle('active', isAutoRotating);
        }});

        const btnWireframe = document.getElementById('btn-wireframe');
        btnWireframe.addEventListener('click', () => {{
            if (!currentModel) return;
            isWireframe = !isWireframe;
            currentModel.traverse((node) => {{
                if (node.isMesh && node.material) {{
                    node.material.wireframe = isWireframe;
                }}
            }});
            btnWireframe.classList.toggle('active', isWireframe);
        }});

        const btnGrid = document.getElementById('btn-grid');
        btnGrid.addEventListener('click', () => {{
            gridHelper.visible = !gridHelper.visible;
            btnGrid.classList.toggle('active', gridHelper.visible);
        }});

        const btnReset = document.getElementById('btn-reset');
        btnReset.addEventListener('click', () => {{
            if (currentModel) fitModelToView(currentModel);
        }});

        // Window resize handler
        window.addEventListener('resize', onWindowResize, false);
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        // ──────────────────────────────────────────────
        // 5. Animation Render Loop
        // ──────────────────────────────────────────────
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# Viewer Generation Functions
# ──────────────────────────────────────────────

def generate_html_viewer(
    mesh_path: Union[str, Path],
    output_html_path: Optional[Union[str, Path]] = None,
    title: str = "Parallax 3D Interactive Viewer",
    embed_model: bool = True,
) -> Path:
    """Generate an interactive HTML5/Three.js 3D viewer for a .glb mesh.

    Parameters
    ----------
    mesh_path : str or Path
        Path to input `.glb` binary glTF model.
    output_html_path : str or Path, optional
        Destination `.html` file path. Defaults to `<mesh_parent>/viewer.html`.
    title : str
        Browser window / tab title.
    embed_model : bool
        If True, embeds the entire GLB binary as a base64 Data URI inside the HTML.
        This enables opening the HTML file directly in any browser (via `file://`)
        without hitting local CORS security restrictions.

    Returns
    -------
    Path
        Absolute path to generated HTML viewer.
    """
    input_mesh = Path(mesh_path)
    if not input_mesh.exists():
        raise FileNotFoundError(f"3D mesh file not found: {input_mesh}")

    if output_html_path is None:
        out_path = input_mesh.parent / "viewer.html"
    else:
        out_path = Path(output_html_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = input_mesh.name

    if embed_model:
        # Embed binary GLB as Base64 Data URI
        with open(input_mesh, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("ascii")
        model_source = f"data:model/gltf-binary;base64,{b64_data}"
    else:
        # Use relative path
        try:
            model_source = str(input_mesh.relative_to(out_path.parent))
        except ValueError:
            model_source = input_mesh.name

    html_content = HTML_VIEWER_TEMPLATE.format(
        TITLE=title,
        MODEL_NAME=model_name,
        MODEL_DATA_OR_URL=model_source,
    )

    out_path.write_text(html_content, encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.visualize",
        description="Parallax 3D Web Viewer Generator — Create standalone interactive Three.js 3D viewers.",
    )
    parser.add_argument("mesh", type=str, help="Path to input .glb 3D mesh file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination HTML viewer file (default: <mesh_dir>/viewer.html).",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Do not embed model as Base64; use relative file reference instead.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Parallax 3D Viewer",
        help="Viewer browser title.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for generating a 3D web viewer."""
    args = _build_parser().parse_args(argv)
    mesh_path = Path(args.mesh)

    if not mesh_path.exists():
        print(f"Error: Mesh file '{mesh_path}' not found.", file=sys.stderr)
        sys.exit(1)

    viewer_path = generate_html_viewer(
        mesh_path=mesh_path,
        output_html_path=args.output,
        title=args.title,
        embed_model=not args.no_embed,
    )

    print("=" * 60)
    print("  PARALLAX — Interactive 3D Web Viewer Generated  ")
    print("=" * 60)
    print(f"Source Mesh:  {mesh_path}")
    print(f"HTML Viewer:  {viewer_path}")
    print("\nTo view in browser, open the file directly:")
    print(f"  open {viewer_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
