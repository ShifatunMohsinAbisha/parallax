# Parallax

**Single-Image 3D Object Reconstruction using Deep Learning and Computer Vision.**

Parallax is an end-to-end computer vision and machine learning framework designed to reconstruct full 3D representations (point clouds, surface meshes, and interactive 3D web visualizations) from a single 2D photograph. The project investigates a central research question in modern spatial vision:

> *"How accurately can machine learning and geometric reasoning infer 3D structure from a single 2D observation?"*

By bridging monocular neural depth estimation, deep semantic segmentation, pinhole projective camera calibration, computational surface reconstruction, and iterative geometric refinement, Parallax transforms flat photographs into navigable 3D models.

---

## 1. Pipeline Overview

Parallax processes monocular RGB images through an 8-stage modular computer vision pipeline:

```
                       ┌─────────────────────────┐
                       │  Input Image (2D RGB)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   1. Preprocessing      │ (src/preprocessing.py)
                       │   - Aspect-ratio pad    │
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
                       ┌─────────────────────────┐
                       │ 6. Mesh Reconstruction  │ (src/mesh_reconstruction.py)
                       │ - Grid / Poisson / BPA  │
                       │ - Multi-format export   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  7. Refinement & Smooth │ (src/refinement.py)
                       │ - Taubin / Laplacian    │
                       │ - Hole repair & pruning │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ 8. Interactive Viewer   │ (src/visualize.py)
                       │ - Three.js WebGL engine │
                       │ - Zero-CORS HTML5 app   │
                       └────────────┬────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                 Reconstructed 3D Output                 │
       │   • 3D Surface Meshes (.glb, .obj, .ply)                │
       │   • Refined 3D Meshes (_refined.glb, .obj, .ply)        │
       │   • 3D Point Clouds (.ply, .pcd)                        │
       │   • Interactive WebGL 3D Viewer (viewer.html)           │
       │   • Before/After Refinement Comparison Image (.png)     │
       │   • 3D Isometric Projection Screenshot (.png)           │
       │   • 4-Panel End-to-End Diagnostic Overview (.png)       │
       └─────────────────────────────────────────────────────────┘
```

---

## 2. Setup Instructions

### Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Operating System**: macOS (Apple Silicon / Intel), Linux, or Windows.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ShifatunMohsinAbisha/parallax.git
cd parallax

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Usage Examples

### Running the Complete End-to-End Pipeline
Execute all 8 stages on an input photograph in a single command:
```bash
python -m src.pipeline path/to/image.png --output-dir outputs --generate-viewer
```

### Opening the Interactive 3D Web Viewer
Open the generated WebGL viewer directly in your browser:
```bash
# macOS
open outputs/viewer.html

# Linux
xdg-open outputs/viewer.html

# Windows
start outputs/viewer.html
```

### Running Individual Stages & Utilities

#### Preprocessing
```bash
python -m src.preprocessing path/to/image.png --output-dir outputs
```

#### Object Segmentation
```bash
python -m src.segmentation path/to/image.png --output-dir outputs --method auto
```

#### Monocular Depth Estimation
```bash
python -m src.depth_estimation path/to/image.png --output-dir outputs --method auto --colormap inferno
```

#### Point Cloud Post-Processing & Geometry
```bash
python -m src.point_cloud outputs/reconstruction.ply --output-dir outputs --nb-neighbors 20 --std-ratio 2.0
```

#### 3D Mesh Refinement
```bash
python -m src.refinement outputs/sample_vase_reconstruction.glb --output-dir outputs --smoothing-method taubin --iterations 5
```

#### Standalone 3D HTML Viewer Generation
```bash
python -m src.visualize outputs/sample_vase_refined.glb --output outputs/viewer.html
```

#### Quantitative Evaluation Benchmark
```bash
python -m src.evaluation --output-dir outputs --eval-data-dir data/eval_shapes
```

#### Running the Unit Test Suite
```bash
pytest tests/ -v
```

---

## 4. Technologies Used

- **Deep Learning**: [PyTorch](https://pytorch.org/) (Tensor computation & Autograd), [TorchVision](https://pytorch.org/vision/) (Pretrained segmentation & deep backbones).
- **Computer Vision & Image I/O**: [OpenCV](https://opencv.org/) (Spatial filters, morphological operations, Graph Cuts), [Pillow](https://python-pillow.org/) (Image formats).
- **3D Geometry & Mesh Processing**: [Trimesh](https://trimesh.org/) (Polygonal mesh manipulation, GLB/OBJ/PLY export, ray-casting), [Open3D](http://www.open3d.org/) (Point cloud processing, Poisson surface reconstruction, Ball-Pivoting).
- **Scientific Computing**: [NumPy](https://numpy.org/) (Vectorized linear algebra & unprojection), [SciPy](https://scipy.org/) (Spatial KD-trees, nearest-neighbor searches).
- **Interactive 3D Web Visualization**: [Three.js](https://threejs.org/) (r128 WebGL renderer, OrbitControls, GLTFLoader).
- **Testing & Quality Assurance**: [pytest](https://pytest.org/) (Comprehensive unit test suite).

---

## 5. Models & Algorithms Used

Parallax is designed for modular, decoupled execution with automatic zero-download offline fallbacks:

### Object Segmentation Models

| Model / Algorithm | Technique / Backbone | Weights Size | License | Key Characteristics |
|---|---|---|---|---|
| **`lraspp_mobilenet_v3_large`** *(Primary DL)* | MobileNetV3 + Lite Reduced Atrous Spatial Pyramid Pooling | ~13 MB | BSD 3-Clause | Fast real-time CPU/GPU semantic object isolation |
| **`deeplabv3_mobilenet_v3_large`** *(High-Accuracy DL)* | MobileNetV3 + Atrous Spatial Pyramid Pooling | ~42 MB | BSD 3-Clause | High boundary precision |
| **`saliency_grabcut`** *(Offline Fallback)* | Center-biased color saliency + iterative Graph Cuts | 0 MB (Built-in) | Apache 2.0 | Zero-dependency, network-independent offline segmentation |

### Monocular Depth Estimation Models

| Model / Algorithm | Technique / Backbone | Weights Size | License | Key Characteristics |
|---|---|---|---|---|
| **`midas_v21_small`** *(Primary DL)* | EfficientNet-Lite3 / MobileNet | ~45 MB | MIT License | Optimal speed/fidelity trade-off for monocular depth |
| **`depth_anything_v2_vits`** *(High-Detail DL)* | DINOv2-Small Transformer | ~98 MB | Apache 2.0 | Fine edge delineation |
| **`geometric_shading`** *(Offline Fallback)* | Distance Transform + Monocular Shape-from-Shading | 0 MB (Built-in) | Apache 2.0 | Fast, zero-weight algorithmic depth generator |

### 3D Surface Reconstruction & Refinement Algorithms

| Stage / Algorithm | Mathematical Core | Strengths & Trade-offs | Recommended Use Case |
|---|---|---|---|
| **Structured Grid Triangulation (`grid`)** | Quad-pair triangulation with depth discontinuity thresholding | Sub-5ms execution, exact UV pixel correspondence; front-manifold topology | Default single-image 3D generation |
| **Poisson Surface Reconstruction (`poisson`)** | Solves $\Delta \chi = \nabla \cdot \mathbf{V}$ on an adaptive octree | Smooth, watertight organic geometry; smooths sharp creases | Watertight 3D printing & organic objects |
| **Ball-Pivoting Algorithm (`ball_pivoting`)** | Rolls virtual sphere across point triplets | Preserves sharp angular geometry; sensitive to point sparsity | Mechanical parts and uniform point clouds |
| **Taubin Refinement (`taubin`)** | Two-step $\lambda-\mu$ diffusion/anti-diffusion filter | Attenuates surface roughness (+23.1% reduction) without volume shrinkage | Post-processing mesh smoothing |
| **Laplacian Smoothing (`laplacian`)** | 1-ring barycentric neighborhood relaxation | Rapid local normal variance reduction | Baseline surface denoising |

---

## 6. Quantitative Evaluation & Benchmark

### Evaluation Methodology
Because massive real-world 3D scanner repositories (e.g. ShapeNet) introduce metric ambiguity and prohibitive download hurdles, Parallax employs a **synthetic geometric primitive benchmark** (Cube, Sphere, Cylinder, Cone) with known exact analytical geometry.

Each primitive is rendered into a calibrated 2D image via the pinhole camera model, reconstructed end-to-end through the 8-stage Parallax pipeline, and evaluated against the ground-truth 3D model using two standard metrics:

1. **Symmetric Chamfer Distance (CD)**:
   $$\text{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\|_2^2 + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|q - p\|_2^2$$
2. **Point-to-Mesh Surface Distance (P2M)**:
   $$d_{\text{P2M}}(P, M_{\text{gt}}) = \frac{1}{|P|} \sum_{p \in P} \min_{s \in \text{Surface}(M_{\text{gt}})} \|p - s\|_2$$

### Quantitative Benchmark Results

| Synthetic Primitive | Chamfer Distance (L2) | Chamfer Distance (L1) | Point-to-Mesh Dist | GT Verts / Faces | Reconstructed Points | Reconstructed Faces | Runtime (s) |
|---|---|---|---|---|---|---|---|
| **Cube** | `0.08486` | `0.15078` | `0.13713` | 8 / 12 | 17,334 | 33,690 | 8.09s |
| **Sphere** | `0.38235` | `0.38325` | `0.36557` | 642 / 1,280 | 8,469 | 16,562 | 2.84s |
| **Cylinder** | `0.23838` | `0.30227` | `0.31808` | 98 / 192 | 11,493 | 22,451 | 4.17s |
| **Cone** | `0.18252` | `0.25427` | `0.22835` | 50 / 96 | 6,412 | 12,507 | 1.88s |
| **AVERAGE / MEAN** | **`0.22203`** | **`0.27264`** | **`0.26228`** | — | — | — | **4.25s** |

### Interpretation of Results
- **Planar & Compact Geometries (Cube / Cone)**: Exhibit the lowest Chamfer Distance (`0.08486` and `0.18252`), demonstrating that sharp, high-contrast silhouette boundaries are accurately segmented and back-projected.
- **Curved & Continuous Surfaces (Sphere / Cylinder)**: Depth curvature is smoothly preserved, though monocular single-view perspective foreshortening slightly broadens the boundary Chamfer distance.
- **Front-Manifold Surface Accuracy**: Across all primitives, the reconstructed front-facing surface achieves low average point-to-mesh deviation (`~0.26` units on a unit bounding sphere).

---

## 7. Known Limitations of Single-Image 3D Reconstruction

Single-image 3D reconstruction is fundamentally an **ill-posed inverse problem**. Users should consider the following constraints:

1. **Occluded Geometry (The Single-View / "Backside" Problem)**:
   - A single monocular photograph contains zero visual or geometric information regarding surfaces facing away from the camera viewpoint.
   - Parallax reconstructs a high-fidelity **2.5D surface manifold (front-facing relief / shell)**. Reconstructing true 360° solid volumes without multi-view observations requires generative hallucination.
2. **Scale & Metric Ambiguity**:
   - Monocular depth networks infer relative, affine-invariant depth maps rather than metric millimeters, unless calibrated with physical camera intrinsics and known reference scale markers.
3. **Specular & Textureless Surfaces**:
   - Regions with specular reflections or uniform flat texture rely on learned semantic priors rather than photometric shading gradients, which can introduce localized depth distortions.

---

## 8. Project Structure

```
Parallax/
├── src/
│   ├── preprocessing.py       # Aspect-preserving resizing, normalization, image I/O
│   ├── segmentation.py        # LRASPP MobileNetV3 & Saliency GrabCut object isolation
│   ├── depth_estimation.py    # MiDaS Small & Geometric monocular depth estimation
│   ├── geometry.py            # Pinhole camera back-projection & surface normal computation
│   ├── point_cloud.py         # Open3D & NumPy statistical outlier filtering, PCD/PLY export
│   ├── mesh_reconstruction.py # 3D surface mesh generation (Grid, Poisson, BPA) & export
│   ├── refinement.py          # Mesh smoothing (Taubin/Laplacian), hole filling & boundary cleanup
│   ├── visualize.py           # Standalone Three.js interactive 3D web viewer generator
│   ├── evaluation.py          # Quantitative benchmark (Chamfer & Point-to-Mesh metrics)
│   ├── pipeline.py            # Unified end-to-end 8-stage reconstruction CLI & orchestrator
│   ├── data/                  # Dataset loaders & custom datasets
│   ├── models/                # Neural network model definitions
│   ├── training/              # Model training and fine-tuning loops
│   └── inference/             # Batch inference and reconstruction scripts
├── tests/
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_depth_estimation.py
│   ├── test_geometry.py
│   ├── test_point_cloud.py
│   ├── test_mesh_reconstruction.py
│   ├── test_refinement.py
│   ├── test_visualize.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
├── notebooks/                 # Jupyter exploration & visualization notebooks
├── data/
│   └── eval_shapes/           # Rendered synthetic evaluation primitives
├── outputs/                   # Reconstructions, 3D meshes, viewers, and evaluation reports
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md
```

---

## 9. Credits & Acknowledgments

- **MiDaS v2.1 Small**: Intel Intelligent Systems Lab (ISL) — [MIT License](https://github.com/isl-org/MiDaS/blob/master/LICENSE).
- **LRASPP MobileNetV3**: PyTorch TorchVision Team — [BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE).
- **Trimesh**: Michael Dawson-Haggerty & Trimesh Contributors — [MIT License](https://github.com/mikedh/trimesh/blob/main/LICENSE.md).
- **Three.js**: Ricardo Cabello (Mr.doob) & Three.js Contributors — [MIT License](https://github.com/mrdoob/three.js/blob/dev/LICENSE).
- **Open3D**: Intel Visual Computing Lab & Open3D Contributors — [MIT License](https://github.com/isl-org/Open3D/blob/main/LICENSE).

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
