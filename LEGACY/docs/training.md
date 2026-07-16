---
title: Training
---

# Training Your Own SpectraFormer

This guide covers how to train a custom SpectraFormer model on your own Raman spectra data.

## Prerequisites

Install the training dependencies:

```bash
uv sync --extra train
```

This includes TensorboardX for logging and gpustat for GPU monitoring.

## Data Preparation

### 1. Prepare Your Data

Your raw Raman spectra should be in `.txt` format. Use the data parser script to convert them to NetCDF format:

```bash
python data_parser_script.py
```

Edit the script to point to your data directory before running.

### 2. Data Structure

Parsed data should be placed under `data/parsed_data_spatial/` with your material name:

```bash
data/
└── parsed_data_spatial/
    └── your-material/
        └── experiment_1/
            └── spectra.nc
```

The NetCDF files should contain:

- `wave_number` dimension for the spectral axis
- Intensity values as the data variable
- Optional spatial dimensions (e.g., `x`, `y` for hyperspectral maps)

## Configuration

### Create a Config File

Create a YAML configuration file in `configs/`:

```yaml
# configs/configs_my_model.yaml
tag: "spectraformer:my_model"

# Model architecture
embedding_dim: 64    # Embedding dimension (64 for min, 256 for base)
num_heads: 8         # Number of attention heads
num_layers: 2        # Number of transformer layers

# Training parameters
learning_rate: 1e-3
learning_rate_decay: "Multiple cosine decay cycles"
warmup_coeff: 1
warmup_steps: 2250
decay_steps: 4500
num_cycles: 15
decline_coeff: 0.9

num_epochs: 350
batch_size: 24       # Must be divisible by number of GPUs
dropout_rate: 0.2
root_rng_seed: 0

# Dataset
train_dataset: "your_dataset_name"

# Logging
log_every_epochs: 1

# Masking configuration (wavenumber ranges to mask)
masked_interval_starts: [-1, 2500]
masked_interval_ends: [1800, 9999]
random_mask: False

# Loss configuration
mean: 'Arithmetic'      # 'Arithmetic' or 'Geometric'
loss_fn: 'CorrGamma'    # 'CorrGamma' or 'MSE'

# Early stopping
is_early_stop: False
early_stop_min_delta: 1e-4
early_stop_patience: 5
```

### Model Variants

| Variant | embedding_dim | num_heads | num_layers |
|---------|---------------|-----------|------------|
| Min     | 64            | 8         | 2          |
| Base    | 256           | 16        | 8          |
| Micro   | 64            | 2         | 2          |

## Run Training

### Basic Training

```bash
python train_script.py \
    --model-tag my_model \
    --material your-material
```

### Multi-GPU Training

By default, training uses all available GPUs:

```bash
python train_script.py \
    --model-tag my_model \
    --material your-material \
    --regime multi-gpu
```

### Single-GPU Training

Force single-GPU training:

```bash
python train_script.py \
    --model-tag my_model \
    --material your-material \
    --regime single-gpu
```

### Command-Line Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--model-tag` | `min70_highf` | Model tag (matches `configs/configs_{tag}.yaml`) |
| `--material` | `SiC-high-f` | Material directory under `data/parsed_data_spatial/` |
| `--regime` | `multi-gpu` | Training regime: `single-gpu` or `multi-gpu` |
| `--debug-nans` | `True` | Enable JAX NaN debugging |

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=logs --samples_per_plugin images=1000
```

This shows:

- Training and validation loss curves
- Model predictions on sample data
- Gradient statistics
- GPU utilization

### GPU Usage

Monitor GPU usage in real-time:

```bash
watch -n 1 nvidia-smi
```

## Checkpoints

Checkpoints are automatically saved to `checkpoints/{model_tag}/` and include:

- Model weights
- Optimizer state
- Training configuration (as metadata)

Training automatically resumes from the latest checkpoint if one exists.

## Using Your Trained Model

After training, use your model for inference:

```bash
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:my_model \
    --input data/new_spectra.nc \
    --output results/unmixed.nc
```

## Tips

### Batch Size

The batch size must be divisible by the number of GPUs when using multi-GPU training. Use multiples of 12 for maximum compatibility (divisible by 1, 2, 3, 4, 6, 12).

### Learning Rate Schedule

The default schedule uses multiple cosine decay cycles with warmup:

```bash
    peak_value
        *___
   ____/    \____
  /              \____*
 /                    end_value
*
init_value

|-- warmup --|-- decay --|
```

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce `batch_size`
2. Use a smaller model variant (reduce `embedding_dim`, `num_heads`, `num_layers`)
3. Use single-GPU training with `--regime single-gpu`
