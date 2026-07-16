---
title: Installation
---

# Installation

This guide covers how to install SpectraFormer for both inference and training.

## Requirements

- Python 3.11
- CUDA 12 (optional, for GPU acceleration)

## Quick Install

```bash
# Clone the repository (--depth 1 for faster download)
git clone --depth 1 https://github.com/pietronvll/SpectraFormer.git
cd SpectraFormer

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## GPU Support

For GPU-accelerated inference and training with CUDA 12:

```bash
uv sync --extra cuda12
```

## Optional Dependencies

SpectraFormer provides optional dependency groups for different use cases:

### Training

Install additional dependencies needed for model training:

```bash
uv sync --extra train
```

This includes:

- `tensorboardX` for logging
- `gpustat` for GPU monitoring

### Dashboard

Install the Streamlit dashboard for interactive visualization:

```bash
uv sync --extra dashboard
```

### Development

Install development tools:

```bash
uv sync --extra dev
```

### All Dependencies

Install everything:

```bash
uv sync --extra all
```

## Verify Installation

After installation, verify that SpectraFormer is correctly installed:

```bash
# Check if the CLI is available
spectraformer-unmix --help
```

For GPU users, verify JAX can see your GPU:

```python
import jax
print(jax.devices())
# Should show your GPU(s), e.g., [cuda(id=0)]
```

## Troubleshooting

### JAX Platform Errors

If you encounter JAX platform errors, you can explicitly set the platform:

```bash
# Force CPU
JAX_PLATFORMS=cpu spectraformer-unmix ...

# Force GPU
JAX_PLATFORMS=cuda spectraformer-unmix ...
```

### CUDA Version Mismatch

Ensure your CUDA version matches JAX's requirements. For CUDA 12 support:

```bash
uv sync --extra cuda12
```

### Python Version

SpectraFormer requires Python 3.11. Check your version:

```bash
python --version
```
