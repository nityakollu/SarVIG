# SARViG - SAR Vision Graph Neural Network

A Vision Graph Neural Network (ViG) implementation for SAR (Synthetic Aperture Radar) image classification, patch classification, self-supervised learning, graph extraction, graph visualization, and graph analysis. 

## Setup Instructions

### 1. Environment Setup (Python 3.11 REQUIRED)

You MUST use Python 3.11. Create a virtual environment:

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv311

# Activate it
venv311\Scripts\activate

# Install PyTorch with CUDA 12.1
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install timm version 0.5.4 (IMPORTANT: NOT the version in requirements.txt)
python -m pip install timm==0.5.4

# Install other dependencies
python -m pip install PyYAML torchprofile
```

## Supported Tasks

SARViG currently supports:

| Task | Description |
|--------|--------|
| `image_classification` | Standard image-level classification |
| `patch_classification` | Patch-level classification using ViG graph representations |
| `self_supervised_train` | Self-supervised pretraining / feature learning |
| `graph_extraction` | Extract KNN graph structures from ViG blocks |
| `graph_visualization` | Visualize graph structure and neighborhood relationships |
| `neighbor_metrics` | Compute graph neighborhood metrics across blocks and patch sizes |

### 2. Dataset Structure

#### Image Classification

```
data/
├── train/
│   └── class0/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    └── class0/
        ├── image1.jpg
        └── image2.jpg
```

#### Patch Classification Fine-Tuning (`dataset_ft`)

Used for patch classification training and validation.

```
dataset_ft/
├── train/
│   ├── SAR/
│   │   ├── image1.png
│   │   └── image2.png
│   └── ...
├── train_masks/
│   ├── boundary/
│   ├── ice/
│   ├── no_data/
│   ├── ocean/
│   └── small_ice/
├── val/
│   ├── SAR/
│   │   ├── image1.png
│   │   └── image2.png
│   └── ...
└── val_masks/
    ├── boundary/
    ├── ice/
    ├── no_data/
    ├── ocean/
    └── small_ice/
```

#### Patch Classification Evaluation (`dataset_eval`)

Used to evaluate a trained patch classification model.

```
dataset_eval/
├── val/
│   └── SAR/
│       ├── image1.png
│       └── image2.png
└── val_masks/
    ├── boundary/
    ├── ice/
    ├── no_data/
    ├── ocean/
    └── small_ice/
```

#### Self-Supervised Training (`unlabeled_dataset`)

Used for self-supervised pretraining and feature learning. No labels or masks are required.

```
unlabeled_dataset/
└── train/
    ├── image1.png
    ├── image2.png
    ├── image3.png
    └── ...
```

#### Graph Analysis

```
dataset_graph/
├── ogSAR/
│   └── image_name.jpg
└── bmSAR/
    └── image_name.png
```

### 3. Run with a YAML Config

```bash
python main.py --config configs/patch_eval.yaml
```

Print the command that would be run, without executing it:

```bash
python main.py --config configs/patch_eval.yaml --dry-run
```

### Classification Run Modes

| `run_mode`   | Effect |
|--------------|--------|
| `train`      | Training only (`--train-only`, `--skip-val`) |
| `train_eval` | Train and run validation each epoch |
| `eval`       | Validation / test only (`--evaluate`) |

### Graph Run Modes

| `run_mode` | Effect |
|------------|----------|
| `extract` | Extract graph structures from selected ViG blocks |
| `visualize` | Create graph visualizations |
| `metrics` | Compute neighborhood metrics |

### Example Tasks

#### Patch Classification Evaluation

```bash
python main.py --config configs/patch_eval.yaml
```

#### Self-Supervised Training

```bash
python main.py --config configs/self_supervised.yaml
```

#### Graph Extraction

```bash
python main.py --config configs/graph_extract.yaml
```

#### Graph Visualization

```bash
python main.py --config configs/graph_vis.yaml
```

#### Neighborhood Metrics

```bash
python main.py --config configs/neighbor_metrics.yaml
```

### Project Workflows

#### Classification Workflow

```
Image Classification
        or
Patch Classification
        or
Self-Supervised Training
```

#### Graph Analysis Workflow

```
Graph Extraction
        ↓
Graph Visualization
        ↓
Neighborhood Metrics
```

### 4. Run Training (manual `train.py`)
Most users should use `main.py` with a YAML configuration. Direct `train.py`
execution is still supported for debugging and development.

```bash
python train.py data/ --model vig_ti_224_gelu -b 2 --num-classes 1 --epochs 5 --workers 0
```
### 5. Key Parameters

```bash
python train.py <DATA_PATH> [OPTIONS]

Required:
  DATA_PATH              Path to data folder (contains train/ and val/)
  --model MODEL          ViG model architecture
  -b, --batch-size N     Batch size (MUST SET: default 32 will error)
  --num-classes N        Number of classes (MUST SET: default 1000 will error)

Optional:
  --epochs N             Number of epochs (default: 200)
  --lr LR                Learning rate (default: 0.01)
  --opt OPTIMIZER        Optimizer: sgd, adam, adamw, etc. (default: sgd)
  --workers N            Number of data loading workers (default: 4, use 0 for Windows)
  --output PATH          Output directory for checkpoints (default: ./output)
```

## Quick Start (Copy-Paste)

```cmd
REM Navigate to project
cd C:\Users\ThanhNam\Desktop\Grad-Class\CS675\Group-Project\SARViG

REM Activate environment
venv311\Scripts\activate

REM Patch evaluation
python main.py --config configs\patch_eval.yaml

REM Self-supervised training
python main.py --config configs\self_supervised.yaml

REM Graph extraction
python main.py --config configs\graph_extract.yaml

REM Graph visualization
python main.py --config configs\graph_vis.yaml

REM Neighborhood metrics
python main.py --config configs\neighbor_metrics.yaml
```

---
