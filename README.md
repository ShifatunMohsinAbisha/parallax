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

Parallax is designed for lightweight, high-performance execution across both CPU and GPU environments. The segmentation subsystem employs the following models and methods:

| Component | Model / Method | Backbone / Technique | Size / Params | Pretraining | License | Primary Use Case |
|---|---|---|---|---|---|---|
| **Object Segmentation (Primary DL)** | `lraspp_mobilenet_v3_large` | MobileNetV3 + Lite RASPP | ~13 MB / 3.2M params | Pascal VOC / COCO (20 classes) | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Fast, real-time object segmentation on CPU/edge devices |
| **Object Segmentation (High Accuracy DL)** | `deeplabv3_mobilenet_v3_large` | MobileNetV3 + ASPP | ~42 MB / 11M params | Pascal VOC / COCO (20 classes) | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | High-fidelity boundary segmentation |
| **Object Segmentation (Offline / Classical)** | Saliency + GrabCut | Spatial color saliency + iterative Graph Cuts | 0 MB (Built-in) | N/A (Algorithmic) | [Apache 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Self-contained, zero-download offline fallback |

### Model Selection Rationale

- **CPU & Resource Optimization**: Single-image 3D pipelines often suffer from heavyweight segmentation dependencies. LRASPP MobileNetV3-Large provides crisp semantic boundaries with sub-50ms CPU execution times and minimal memory consumption.
- **Zero-Download Resilience**: When operating in air-gapped or network-restricted environments, the built-in Saliency + GrabCut engine guarantees immediate object isolation without requiring external checkpoint downloads.

## License

TBD
