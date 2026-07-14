# SARViG - SAR Vision Graph Neural Network

A Vision Graph Neural Network (ViG) implementation for SAR (Synthetic Aperture Radar) image classification, patch classification, self-supervised learning, graph extraction, graph visualization, and graph analysis. 

## Setup Instructions

### Environment Setup (Python 3.11 REQUIRED)

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

### Run with a YAML Config

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

### Run Training (manual `train.py`)
Most users should use `main.py` with a YAML configuration. Direct `train.py`
execution is still supported for debugging and development.

```bash
python train.py data/ --model vig_ti_224_gelu -b 2 --num-classes 1 --epochs 5 --workers 0
```
### Key Parameters

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
