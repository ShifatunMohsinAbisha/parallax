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
       ┌─────────────────────────────────────────────────────────┐
       │                 Reconstructed 3D Output                 │
       │   • 3D Surface Meshes (.glb, .obj, .ply)                │
       │   • Refined 3D Meshes (_refined.glb, .obj, .ply)        │
       │   • 3D Point Clouds (.ply, .pcd)                        │
       │   • Before/After Refinement Comparison Image (.png)     │
       │   • 3D Isometric Projection Screenshot (.png)           │
       │   • 4-Panel End-to-End Diagnostic Overview (.png)       │
       └─────────────────────────────────────────────────────────┘
```

## Project Structure

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
├── data/                      # Input datasets (git-ignored)
├── models/                    # Model weights and checkpoints (git-ignored)
├── outputs/                   # Reconstructions, 3D meshes, and renders (git-ignored)
├── requirements.txt           # Python dependencies
└── README.md
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/ShifatunMohsinAbisha/parallax.git
cd parallax

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full end-to-end single-image 3D reconstruction pipeline
python -m src.pipeline path/to/image.png --output-dir outputs --generate-viewer

# 5. Open the interactive 3D viewer in your browser
open outputs/viewer.html       # On macOS
# On Linux: xdg-open outputs/viewer.html
# On Windows: start outputs/viewer.html
```

### Interactive 3D Web Viewer

You can also generate an interactive Three.js 3D web viewer directly from any `.glb` model using `src/visualize.py`:

```bash
# Generate a self-contained HTML viewer for any 3D mesh
python -m src.visualize outputs/sample_vase_refined.glb --output outputs/viewer.html

# View directly in your browser:
open outputs/viewer.html
```

#### Viewer Controls:
- **Left-Click + Drag**: Orbit and rotate around the 3D object.
- **Right-Click + Drag** (or Shift + Drag): Pan the camera.
- **Scroll Wheel**: Smooth zoom in / out.
- **Floating Controls**: Toggle auto-rotation, inspect wireframe topology, toggle ground grid, and reset camera view.

### Pipeline CLI Options

```bash
# Full command options
python -m src.pipeline path/to/image.png \
    --output-dir outputs \
    --fov 60.0 \
    --depth-method auto \
    --segmentation-method auto \
    --mesh-method auto \
    --smoothing-method taubin \
    --smoothing-iterations 5 \
    --generate-viewer
```

## Quantitative Evaluation & Benchmark

### Evaluation Methodology

Because massive 3D scanner datasets (such as ShapeNet) require extensive GPU clusters and heavy storage downloads, Parallax employs a **synthetic geometric primitive benchmark** (Cube, Sphere, Cylinder, Cone) with known exact analytical geometry.

This approach provides a controlled, reproducible, high-precision quantitative evaluation:
1. Each 3D primitive is rendered to a calibrated 2D image via the pinhole camera projection model.
2. The image is passed through the full Parallax 8-stage reconstruction pipeline.
3. The reconstructed 3D point cloud / mesh is aligned and evaluated against the ground-truth 3D model.

### Mathematical Formulations

#### 1. Symmetric Chamfer Distance (CD)
$$\text{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\|_2^2 + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|q - p\|_2^2$$

Where $P$ is the reconstructed point cloud and $Q$ is the ground-truth surface sample set.

#### 2. Point-to-Mesh Surface Distance (P2M)
$$d_{\text{P2M}}(P, M_{\text{gt}}) = \frac{1}{|P|} \sum_{p \in P} \min_{s \in \text{Surface}(M_{\text{gt}})} \|p - s\|_2$$

### Benchmark Results Summary

| Synthetic Primitive | Chamfer Distance (L2) | Chamfer Distance (L1) | Point-to-Mesh Dist | GT Verts / Faces | Reconstructed Points | Reconstructed Faces | Runtime (s) |
|---|---|---|---|---|---|---|---|
| **Cube** | `0.08486` | `0.15078` | `0.13713` | 8 / 12 | 17,334 | 33,690 | 8.09s |
| **Sphere** | `0.38235` | `0.38325` | `0.36557` | 642 / 1,280 | 8,469 | 16,562 | 2.84s |
| **Cylinder** | `0.23838` | `0.30227` | `0.31808` | 98 / 192 | 11,493 | 22,451 | 4.17s |
| **Cone** | `0.18252` | `0.25427` | `0.22835` | 50 / 96 | 6,412 | 12,507 | 1.88s |
| **AVERAGE / MEAN** | **`0.22203`** | **`0.27264`** | **`0.26228`** | — | — | — | **4.25s** |

To reproduce this benchmark locally:
```bash
python -m src.evaluation --output-dir outputs --eval-data-dir data/eval_shapes
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

### 4. 3D Mesh Refinement & Surface Smoothing

The refinement stage operates as a post-processing pass to optimize visual fidelity and surface topology:

- **Taubin Smoothing (`"taubin"`)**: Two-step volume-preserving filter alternating between positive diffusion ($\lambda > 0$) and negative expansion ($\mu < -\lambda < 0$). Attenuates high-frequency depth quantization ripples without causing volumetric shrinkage.
- **Laplacian Smoothing (`"laplacian"`)**: Neighborhood barycentric relaxation to minimize local normal variance.
- **Hole Filling & Repair**: Closes small open boundary loops with Delaunay ear-clipping triangulation.
- **Boundary & Fragment Pruning**: Trims detached high-discontinuity silhouette triangles.

---

## Known Limitations of Single-Image 3D Reconstruction

Single-image 3D reconstruction is an **ill-posed inverse problem**. Refinement performs best-effort geometric enhancement, but users should be aware of fundamental constraints:

1. **Occluded Geometry (The "Backside" Problem)**:
   - A single 2D photograph contains zero visual or geometric information about surfaces occluded from the camera viewpoint (e.g. the back or underside of an object).
   - Single-image pipelines reconstruct high-fidelity **2.5D surface manifolds (bas-reliefs / front-facing shells)** rather than true 360° scanned solid volumes. Attempting to force closed watertight meshes without multi-view observations involves heuristic hallucination.

2. **Scale & Metric Depth Ambiguity**:
   - Monocular depth networks infer **relative depth** (affine scale and shift invariant) rather than absolute physical millimeters, unless calibrated with known object dimensions or metric camera sensors.

3. **Specular Highlights & Textureless Surfaces**:
   - Monocular depth estimators rely on shading gradients, edge contours, and learned semantic priors. Uniformly reflective, transparent, or textureless surfaces can produce localized depth distortions that geometric smoothing attenuates but cannot completely resolve.

---

## License

TBD
