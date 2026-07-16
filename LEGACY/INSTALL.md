# Installation Instructions

SpectraFormer requires Python 3.10 or later.

## Quick Start with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that ensures reproducible installations via lock files.

### Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### CPU-only Installation (for inference)

```bash
git clone https://github.com/pietronvll/SpectraFormer.git
cd SpectraFormer
uv sync
```

### GPU Installation (for training)

For NVIDIA GPU support with CUDA 12:

```bash
git clone https://github.com/pietronvll/SpectraFormer.git
cd SpectraFormer
uv sync --extra cuda12 --extra train
```

> **Note**: You must have NVIDIA drivers installed (version >= 525.60.13 for CUDA 12).
> See [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

## Alternative: pip Installation

If you prefer pip/conda:

### Step 1: Create Environment

```bash
conda create --name spectraformer python=3.11
conda activate spectraformer
```

### Step 2: Install JAX

**CPU-only:**

```bash
pip install "jax[cpu]"
```

**GPU (CUDA 12):**

```bash
pip install "jax[cuda12]"
```

### Step 3: Install SpectraFormer

**CPU inference only:**

```bash
pip install -e .
```

**GPU training:**

```bash
pip install -e ".[cuda12,train]"
```

## Usage

### Inference CLI

After installation, use the `spectraformer-unmix` command:

```bash
# Basic usage (CPU)
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/my_spectra.nc \
    --output data/unmixed_spectra.nc

# With GPU
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/my_spectra.nc \
    --output data/unmixed_spectra.nc \
    --device gpu
```

### Training

Training uses the YAML configuration files in `configs/`:

```bash
# Basic usage
python train_script.py --model-tag min70_highf --material SiC-high-f

# Single GPU
python train_script.py --model-tag min70_highf --material my_data --regime single-gpu

# Multi-GPU (default)
python train_script.py --model-tag min70_highf --material my_data --regime multi-gpu

# Show all options
python train_script.py --help
```

Arguments:

- `--model-tag`: Must match `configs/configs_{model_tag}.yaml`
- `--material`: Data directory name under `data/parsed_data_spatial/`
- `--regime`: `single-gpu` or `multi-gpu` (default)

## Troubleshooting

### TensorStore CA Certificates (Linux)

If you encounter certificate errors with TensorStore/Orbax checkpoints, set the CA bundle path:

```bash
export TENSORSTORE_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# or
export TENSORSTORE_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
```

Add this line to your `.bashrc` or `.zshrc` to make it permanent.

### JAX Device Selection

To force CPU even when GPU is available:

```bash
JAX_PLATFORMS=cpu spectraformer-unmix --checkpoint ... --input ... --output ...
```

Or in Python:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax  # import after setting environment variable
```
