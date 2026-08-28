# Parallax

**Single-image 3D object reconstruction using machine learning and computer vision.**

Parallax takes a single 2D photograph as input and reconstructs a full 3D representation of the object(s) in the scene. The project combines deep learning–based depth estimation, shape prediction, and mesh generation techniques to infer 3D geometry from monocular images — bridging the gap between flat photographs and volumetric understanding.

## Pipeline Diagram

<!-- TODO: Add pipeline diagram here -->
<!-- Suggested sections: Input Image → Feature Extraction → Depth/Normal Estimation → 3D Representation → Mesh Output -->

```
[ placeholder — pipeline diagram coming soon ]
```

## Project Structure

```
Parallax/
├── src/                # Source code (models, data loaders, training, inference)
│   ├── __init__.py
│   ├── data/           # Dataset loading and preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training loops and utilities
│   └── inference/      # Inference and reconstruction scripts
├── notebooks/          # Jupyter notebooks for exploration and visualization
├── data/               # Datasets (excluded from version control)
├── models/             # Saved model weights and checkpoints (excluded from VC)
├── tests/              # Unit and integration tests
├── outputs/            # Reconstruction outputs, renders, logs (excluded from VC)
├── requirements.txt    # Python dependencies
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

### Model Selection Rationale

- **MiDaS v2.1 Small**: Selected as the primary deep learning depth estimator due to its optimal balance between relative depth fidelity and CPU inference speed. Its lightweight footprint (~45 MB) and compatibility with Torch Hub and ONNX runtimes make it ideal for accessible 3D reconstruction without discrete GPUs.
- **Decoupled Architecture**: Model loading is separated from inference logic via `BaseDepthEstimator` and `load_depth_model()`, allowing users to swap between neural models and offline classical estimators with zero code changes.
- **Offline / Edge Fallback**: The built-in `GeometricShadingEstimator` enables instant, zero-weight monocular depth generation in air-gapped or network-constrained setups.

## License

TBD

