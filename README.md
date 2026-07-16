# SpectraFormer GammaNLL File-Level Pipeline

This fork contains a modified SpectraFormer pipeline for Raman SiC background reconstruction and subtraction.

The original dynamic-masking codebase is preserved unchanged in:

LEGACY/

The repository root contains the new pipeline only.

## Main changes

The new pipeline uses:

- file-level train/validation split, to avoid leakage between spectra from the same Raman map;
- dynamic masking during training;
- a probabilistic Gamma output with two heads:
  - mu: predicted mean SiC spectrum;
  - alpha: predicted Gamma dispersion parameter;
- Gamma negative log-likelihood as training loss;
- validation MSE on mu as best-checkpoint criterion;
- one best checkpoint kept during training;
- inference based by default on mu.

By default, tau = 0, meaning that the subtracted SiC spectrum is the predicted mean mu.

A conservative Gamma quantile can optionally be used at inference by setting tau > 0.

## Legacy code

The folder LEGACY/ is a snapshot of the original experiment/dynamic-masking branch before the modifications in this fork.

It is kept only for reference and reproducibility of the previous pipeline.

## Configuration

The main configuration file for the new pipeline is:

configs/configs_min70_highf_filelevel_gammanll.yaml

It uses:

- 300 epochs;
- batch size 24;
- file-level split;
- 20% validation maps;
- split seed 123;
- GammaNLL training loss;
- best checkpoint selected by validation MSE on mu;
- default inference tau = 0.

## Training

Example:

python train_script_DynMask.py --model-tag min70_highf_filelevel_gammanll --material SiC-high-f --regime single-gpu

The script expects parsed NetCDF data under:

data/parsed_data_spatial/SiC-high-f/

Checkpoints are written under:

checkpoints/min70_highf_filelevel_gammanll/

Logs are written under:

logs/min70_highf_filelevel_gammanll/

## Inference

Example with default mean subtraction:

python -m spectraformer.cli --checkpoint checkpoints/min70_highf_filelevel_gammanll --config configs/configs_min70_highf_filelevel_gammanll.yaml --input path/to/input_or_folder --output inference_outputs --tau 0

Example with conservative Gamma quantile:

python -m spectraformer.cli --checkpoint checkpoints/min70_highf_filelevel_gammanll --config configs/configs_min70_highf_filelevel_gammanll.yaml --input path/to/input_or_folder --output inference_outputs --tau 0.15

## Inference outputs

Each output NetCDF contains:

- spectra;
- masked_spectra;
- mask;
- predicted_spectra;
- predicted_mu;
- predicted_alpha;
- predicted_difference;
- predicted_spectra_quantile;
- predicted_difference_quantile.

When tau = 0, the quantile reconstruction is identical to mu.
