---
title: Running Inference
---

# Running Inference

SpectraFormer provides a command-line tool for running spectral unmixing inference on Raman spectra data.

## Quick Start

```bash
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/my_spectra.nc \
    --output results/unmixed_spectra.nc
```

::: tip
Make sure you have SpectraFormer installed. See the [Installation guide](/installation) for details.
:::

## Command-Line Options

| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `--checkpoint` | Yes | - | Path to the checkpoint directory |
| `--input` | Yes | - | Input NetCDF file (`.nc`) or directory |
| `--output` | Yes | - | Output file or directory path |
| `--device` | No | `auto` | Device: `auto`, `cpu`, or `gpu` |

### Device Selection

- **`auto`** (default): Automatically uses GPU if available, otherwise falls back to CPU
- **`cpu`**: Force CPU-only inference
- **`gpu`**: Force GPU inference (will fail if no GPU is available)

## Usage Examples

### Single File

Process a single NetCDF file:

```bash
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/parsed_data_spatial/sample.nc \
    --output results/unmixed_sample.nc
```

### Batch Processing

Process all `.nc` files in a directory (recursively):

```bash
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/parsed_data_spatial/ \
    --output results/unmixed/
```

The output directory structure mirrors the input structure, with files prefixed with `unmixed_`.

### Force CPU

Run on CPU even if GPU is available:

```bash
spectraformer-unmix \
    --checkpoint checkpoints/spectraformer:min70_highf \
    --input data/sample.nc \
    --output results/unmixed.nc \
    --device cpu
```

## Input Data Format

Input files must be NetCDF (`.nc`) files containing Raman spectra data with:

- A `wave_number` dimension for the spectral axis
- Intensity values as the data variable
- Optional spatial dimensions (e.g., `x`, `y` for hyperspectral maps)

## Output Data Format

The output NetCDF file contains:

| Variable | Description |
| -------- | ----------- |
| `spectra` | Original input spectra (preprocessed) |
| `masked_spectra` | Spectra with SiC regions masked |
| `mask` | Boolean mask indicating masked regions |
| `predicted_spectra` | Model's prediction for the masked regions |
| `predicted_difference` | Difference between original and predicted (graphene signal) |

The `predicted_difference` variable contains the extracted graphene contribution after SiC substrate removal.

## Checkpoints

Checkpoints are stored in `checkpoints/` and contain:

- Model weights
- Training configuration (stored as metadata)
- Optimizer state

The checkpoint name (e.g., `spectraformer:min70_highf`) identifies the model variant and training configuration.

## Troubleshooting

### "Checkpoint does not contain configuration metadata"

This error occurs when using a checkpoint from an older version of SpectraFormer. Re-train the model or use a compatible checkpoint.

### Out of Memory on GPU

For large datasets, process files individually or use `--device cpu` for lower memory usage.

### JAX Platform Errors

If you encounter JAX platform errors, explicitly set the device:

```bash
# Force CPU
JAX_PLATFORMS=cpu spectraformer-unmix ...

# Force GPU
JAX_PLATFORMS=cuda spectraformer-unmix ...
```
