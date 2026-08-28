# Parallax

**Single-image 3D object reconstruction using machine learning and computer vision.**

Parallax takes a single 2D photograph as input and reconstructs a full 3D representation of the object(s) in the scene. The project combines deep learning–based depth estimation, shape prediction, and mesh generation techniques to infer 3D geometry from monocular images — bridging the gap between flat photographs and volumetric understanding.

## Pipeline Architecture

```
                       ┌─────────────────────────┐
                       │  Input Image (2D RGB)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   1. Preprocessing      │ (src/preprocessing.py)
                       │   - Aspect-ratio resize │
                       │   - ImageNet normalize  │
                       └────────────┬────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │   2. Object Segmentation  │ │   3. Depth Estimation     │
       │   (src/segmentation.py)   │ │   (src/depth_estimation.py) │
       │   - LRASPP MobileNetV3 /  │ │   - MiDaS v2.1 Small /      │
       │     Saliency+GrabCut      │ │     Geometric Shading       │
       │   - Binary / Soft Mask    │ │   - Dense Depth Map         │
       └─────────────┬─────────────┘ └─────────────┬─────────────┘
                     └──────────────┬──────────────┘
                                    │ (Masked Depth + RGB + Intrinsics K)
                                    ▼
                       ┌─────────────────────────┐
                       │   4. Geometry Engine    │ (src/geometry.py)
                       │   - Pinhole unproject   │
                       │   - Surface normals     │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   5. Point Cloud Post   │ (src/point_cloud.py)
                       │   - Statistical filter  │ (Open3D / NumPy SOR)
                       │   - Color mapping       │
                       └────────────┬────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                 Reconstructed 3D Output                 │
       │   • 3D Point Cloud (.ply, .pcd, .obj)                   │
       │   • 3D Isometric Projection Screenshot (.png)           │
       │   • 4-Panel End-to-End Diagnostic Overview              │
       └─────────────────────────────────────────────────────────┘
```

## Project Structure

```
Parallax/
├── src/
│   ├── preprocessing.py    # Aspect-preserving resizing, normalization, image I/O
│   ├── segmentation.py     # LRASPP MobileNetV3 & Saliency GrabCut object isolation
│   ├── depth_estimation.py # MiDaS Small & Geometric monocular depth estimation
│   ├── geometry.py         # Pinhole camera back-projection & surface normal computation
│   ├── point_cloud.py      # Open3D & NumPy statistical outlier filtering, PCD/PLY export
│   ├── pipeline.py         # Unified end-to-end reconstruction CLI & orchestrator
│   ├── data/               # Dataset loaders & custom datasets
│   ├── models/             # Neural network model definitions
│   ├── training/           # Model training and fine-tuning loops
│   └── inference/          # Batch inference and reconstruction scripts
├── tests/
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_depth_estimation.py
│   ├── test_geometry.py
│   ├── test_point_cloud.py
│   ├── test_mesh_reconstruction.py
│   └── test_pipeline.py
├── notebooks/              # Jupyter exploration & visualization notebooks
├── data/                   # Input datasets (git-ignored)
├── models/                 # Model weights and checkpoints (git-ignored)
├── outputs/                # Reconstructions, 3D meshes, and renders (git-ignored)
├── requirements.txt        # Python dependencies
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ShifatunMohsinAbisha/parallax.git
cd parallax

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete end-to-end 3D reconstruction pipeline on an image
python -m src.pipeline path/to/image.png --output-dir outputs --fov 60.0
```

## Models Used

Parallax is designed for modular, lightweight, high-performance execution across both CPU and GPU environments.

### 1. Object Segmentation

| Model / Method | Backbone / Technique | Size / Params | Pretraining | License | Primary Use Case |
|---|---|---|---|---|---|
| **`lraspp_mobilenet_v3_large`** *(Primary DL)* | MobileNetV3 + Lite RASPP | ~13 MB / 3.2M params | Pascal VOC / COCO (20 classes) | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Fast, real-time object segmentation on CPU/edge devices |
| **`deeplabv3_mobilenet_v3_large`** *(High-Accuracy DL)* | MobileNetV3 + ASPP | ~42 MB / 11M params | Pascal VOC / COCO (20 classes) | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | High-fidelity boundary segmentation |
| **`saliency_grabcut`** *(Offline / Classical)* | Spatial color saliency + iterative Graph Cuts | 0 MB (Built-in) | N/A (Algorithmic) | [Apache 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Self-contained, zero-download offline fallback |

### 2. Monocular Depth Estimation

We researched and benchmarked candidate architectures for monocular depth estimation:

| Candidate Model | Backbone / Technique | Size / Params | Latency (CPU) | License | Selection Status |
|---|---|---|---|---|---|
| **MiDaS v2.1 Small (`midas_v21_small`)** | EfficientNet-Lite3 / MobileNet | ~45 MB / ~25M params | ~40–80 ms | [MIT License](https://github.com/isl-org/MiDaS/blob/master/LICENSE) | **SELECTED (Primary DL Model)** |
| **Depth Anything v2 Small (`depth_anything_v2_vits`)** | DINOv2-Small | ~98 MB / ~24.8M params | ~120–220 ms | [Apache 2.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE) | Alternative (High boundary detail) |
| **DPT-Large (`dpt_large`)** | ViT-Large Transformer | ~1.3 GB / ~340M params | >1200 ms | [MIT License](https://github.com/isl-org/MiDaS/blob/master/LICENSE) | Benchmark only (Heavy compute) |
| **Geometric Shading (`geometric_shading`)** | Distance Transform + Shape-from-Shading | 0 MB (Built-in) | <10 ms | [Apache 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) | **SELECTED (Offline Fallback)** |

### 3. 3D Surface Mesh Reconstruction

Parallax supports multiple surface reconstruction algorithms to transform unstructured point clouds into continuous polygonal 3D models:

| Reconstruction Method | Core Technique | Strengths | Trade-offs | License / Engine | Recommended For |
|---|---|---|---|---|---|
| **Structured Grid Triangulation (`grid`)** | Monocular quad-pair triangulation with depth discontinuity thresholding | Ultra-fast (<5 ms), deterministic topology, preserves fine UV alignment | Single-view front-face manifold | [MIT License](https://github.com/mikedh/trimesh/blob/main/LICENSE.md) (Trimesh) | **Default single-image 3D generation** |
| **Poisson Surface Reconstruction (`poisson`)** | Solves $\Delta \chi = \nabla \cdot \mathbf{V}$ over an adaptive octree | Watertight, smooth organic surfaces, robust to sensor noise | Requires normal vectors; smooths sharp creases | [MIT License](https://github.com/isl-org/Open3D/blob/main/LICENSE) (Open3D) | Organic models, watertight 3D printing |
| **Ball-Pivoting Algorithm (`ball_pivoting`)** | Rolls virtual sphere across point triplets to form Delaunay-like triangles | Exact coordinate interpolation; retains sharp boundary edges | Sensitive to ball radius selection and point sparsity | [MIT License](https://github.com/isl-org/Open3D/blob/main/LICENSE) (Open3D) | Uniform point clouds, mechanical parts |

#### Mesh Cleanup & Export Capabilities
- **Automated Geometry Repair**: Eliminates degenerate (zero-area) faces, removes duplicate triangles and unreferenced vertices, repairs inconsistent surface normal orientations, and stitches small boundary holes.
- **Export Formats**:
  - **`.glb`** (Binary glTF 2.0) — Standard 3D web format with embedded vertex colors and material properties.
  - **`.obj`** (Wavefront) — Universally supported format for Blender, Unity, Unreal Engine, and CAD tools.
  - **`.ply`** (Polygon File Format) — Point- and mesh-level geometry storage with surface normals.

### Model Selection Rationale

- **MiDaS v2.1 Small**: Selected as the primary deep learning depth estimator due to its optimal balance between relative depth fidelity and CPU inference speed. Its lightweight footprint (~45 MB) and compatibility with Torch Hub and ONNX runtimes make it ideal for accessible 3D reconstruction without discrete GPUs.
- **Structured Grid & Trimesh**: Provides sub-10ms mesh generation on CPU while guaranteeing that reconstructed vertices and colors correspond exactly to monocular pixels.
- **Decoupled Architecture**: Model loading is separated from inference logic via `BaseDepthEstimator` and `load_depth_model()`, allowing users to swap between neural models and offline classical estimators with zero code changes.
- **Offline / Edge Fallback**: The built-in `GeometricShadingEstimator` and `GridTriangulation` enable instant, zero-weight 3D reconstruction in air-gapped or network-constrained setups.

## License

TBD


