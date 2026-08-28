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

## License

TBD
