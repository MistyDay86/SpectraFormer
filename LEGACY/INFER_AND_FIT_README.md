# SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline

## Overview

`infer_and_fit_raman_map.py` is a complete end-to-end pipeline designed for analyzing **homogeneous material growth across substrate** by measuring Raman spectra at multiple locations:

1. **Loading** all `.txt` Raman spectral maps from input folder
2. **Aggregating** all spatial maps into a single large dataset
3. **Preprocessing** aggregated data using the SpectraFormer preprocessing pipeline
4. **Inference** via the trained SpectraFormer model on the combined dataset
5. **Computing** median spectrum and standard deviation across all spectra
6. **Validity check** to skip fitting if no material is detected (max intensity < 0.05)
7. **Fitting** the median spectrum with multiple Voigt profiles
8. **Loss calculation** based on peak area ratios to quantify growth quality
9. **Reporting** results including loss function and generating visualizations

## Key Features

- ✅ **Folder-based input**: Scans folder for all `.txt` files and aggregates them
- ✅ **Multi-map aggregation**: Flattens all spatial dimensions into single (spectra, wave_number) dataset
- ✅ **Single inference**: Runs SpectraFormer once on aggregated dataset (faster & more consistent)
- ✅ **Robust statistics**: Median+std computation across all spectra (handles low SNR and outliers)
- ✅ **Validity check**: Skips fitting if max intensity < 0.05 (no material growth), assignsloss = 4.0
- ✅ **Loss quantification**: Calculates loss based on B/G/L/2D peak areas:
  - Loss_term_1 = Area(2D) / Area(G) → should be ≈ 0
  - Loss_term_2 = (Area(B)+Area(G)+Area(L)) / sum(all areas) → should be ≈ 1
  - Loss = (Loss_term_1 + Loss_term_2)² → range [0, 4]
- ✅ **Comprehensive results**: Saves peak parameters, uncertainties, quality metrics, AND loss function
- ✅ **Publication-ready**: Generates professional 2-panel plots with uncertainty bands

## Input Requirements

### 1. Raman Spectral Maps Folder

Folder containing multiple `.txt` files, each with format:

```text
X_0    X_1    X_2    ...    wave_number    intensity
1.0    1.0    1.0    ...    1000.0         150.5
1.0    1.0    2.0    ...    1000.5         155.2
...
```

- **Each file**: Different measurement location on substrate
- **Spatial columns** (X_0, X_1, ...): Coordinates defining the spatial map grid per file
- **Last 2 columns**: `wave_number` (Raman shift in cm⁻¹) and `intensity` values
- **All files** are aggregated into single (total_spectra, wave_number) dataset

**Example folder structure:**
```
substrate_measurements/
  ├── location_1.txt  (5×5 spatial points)
  ├── location_2.txt  (5×5 spatial points)
  ├── location_3.txt  (5×5 spatial points)
  └── location_4.txt  (5×5 spatial points)
Total: 100 spectra aggregated into one for inference
```

### 2. Model Checkpoint

- Pre-trained SpectraFormer checkpoint stored in `checkpoints/` directory
- Expected naming: `spectraformer:model_name`
  (e.g., `spectraformer:min70_highf`)

### 3. Peak Parameters

Peak parameters are **hardcoded** in the script for easy modification:

- `DEFAULT_PEAKS["centers"]`: Raman shift positions (cm⁻¹)
- `DEFAULT_PEAKS["widths"]`: Initial peak width estimates (cm⁻¹)
- `DEFAULT_PEAKS["gamma_ratios"]`: Lorentzian fractions [0,1]
- `DEFAULT_PEAKS["peak_names"]`: Peak identifiers (e.g., B, L, G, 2D)

## Usage

### Basic Example (All .txt files in folder)

```bash
python infer_and_fit_raman_map.py \
    --input "data/substrate_measurements/" \
    --output "results/output"
```

The script will use hardcoded default parameters. To modify:

Edit `DEFAULT_PEAKS` and `DEFAULT_FITTING` in the script:

```python
DEFAULT_PEAKS = {
    "centers": [1492, 1564, 1607, 2735],  # B, L, G, 2D only
    "widths": [40.0, 40.0, 40.0, 40.0],
    "gamma_ratios": [0.01, 0.01, 0.01, 0.01],
    "peak_names": ["B", "L", "G", "2D"],
}

DEFAULT_FITTING = {
    "center_windows": [200.0, 200.0, 200.0, 200.0],
    "min_widths": [30.0, 30.0, 30.0, 30.0],
    "max_widths": [150.0, 150.0, 150.0, 150.0],
    "maxfev": 30000,
}

DEFAULT_MODEL = "min70_highf"
DEFAULT_CKPTS_DIR = "./saved_models/checkpoints"
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
| ---------- | ------------- | --------- |
| `--input` | Path to folder containing .txt Raman maps | `data/substrate_measurements/` |
| `--output` | Output directory for results | `results/output` |

### Configuration

All model, peak, and fitting parameters are defined in the script constants for
easy modification without command-line arguments.

## Output Structure

The script creates the following output files:

```text
output_dir/
├── fitting_results.json          # Complete results with uncertainties AND loss function
├── fitting_parameters.csv        # Peak parameters in CSV format
└── fitting_visualization.png     # Publication-quality plot
```

### fitting_results.json Structure (with loss function)

```json
{
  "metadata": {
    "timestamp": "2026-04-07 15:23:18",
    "n_peaks": 4,
    "peak_names": ["B", "L", "G", "2D"],
    "n_spectra_averaged": 100,
    "model_config": {
      "num_heads": 8,
      "num_layers": 4,
      "embedding_dim": 128
    }
  },
  "peaks": [
    {
      "name": "B",
      "expected_center_cm_inv": 1492,
      "parameters": {
        "center": {"value": 1492.3, "uncertainty": 0.5},
        "width": {"value": 45.2, "uncertainty": 1.2},
        "area": {"value": 85.4, "uncertainty": 2.3},
        "gamma_ratio": {"value": 0.01, "uncertainty": 0.003}
      }
    }
  ],
  "quality_metrics": {
    "r_squared": 0.992,
    "sum_squared_error": 0.0156,
    "mean_absolute_error": 0.0032
  },
  "loss_function": {
    "loss": 0.25,
    "loss_term_1_2D_over_G": 0.15,
    "loss_term_2_structural_ratio": 0.35,
    "area_2D": 45.2,
    "area_G": 301.5,
    "area_B": 85.4,
    "area_L": 92.8,
    "total_area": 524.9,
    "description": {
      "loss_term_1": "Area(2D) / Area(G) - should be 0 for good growth",
      "loss_term_2": "(Area(B)+Area(G)+Area(L)) / sum(all areas) - should be 1 for structural",
      "loss_formula": "loss = (loss_term_1 + loss_term_2)^2",
      "range": "0 (best/perfect growth) to 4 (worst/no growth)"
    }
  }
}
```

**Loss Function Interpretation:**
- **loss ≈ 0**: Perfect growth (2D suppressed, structural peaks dominant)
- **loss ≈ 2**: Mixed growth (some 2D present, structural still present)
- **loss ≈ 4**: No growth detected (max intensity < 0.05, fitting skipped)

### fitting_parameters.csv Structure

```csv
peak_name,parameter,value,uncertainty
B,center,1.492300e+03,5.000000e-01
B,width,4.520000e+01,1.200000e+00
B,area,8.540000e+01,2.300000e+00
B,gamma_ratio,1.000000e-02,3.000000e-03
G,center,1.607200e+03,4.800000e-01
...
```

## Pipeline Steps (Multi-Location Aggregation Workflow)

### 1. Load all `.txt` files from folder

- Scans input folder for all `.txt` files matching pattern `*.txt`
- Parses each file using `parse_dataset()` to get (X_0, X_1, ..., wave_number) DataArray
- Logs loaded files with individual shapes
- Returns list of DataArrays ready for aggregation

**Example**: Folder with 4 files → each 5×5×1800 → 4 DataArrays loaded

### 2. Preprocess individual maps

- Applies `preprocess_dataset()` with `whitaker_hayes_with_outliers` option on each map
- Stacks spatial dimensions (X_0, X_1, ...) into a single `spectra` dimension per file
- Performs Whitaker-Hayes spike detection and removal
- Normalizes to [0,1] range

### 3. Aggregate all preprocessed maps

- Takes all preprocessed DataArrays from step 2
- Concatenates along the `spectra` dimension
- **Result**: Single (total_spectra, wave_number) dataset ready for inference batch processing

**Example**: 4 maps of 5×5 spectra → 100 spectra, 1800 wavenumbers → one large (100, 1800) dataset

### 4. Run single SpectraFormer inference

- Loads SpectraFormer checkpoint once
- Runs inference on the entire aggregated dataset in batches
- Extracts `predicted_difference = original_spectrum - predicted_spectrum` for each spectrum
- **Advantage**: Single model initialization, consistent constraints across all spectra

### 5. Extract and compute median & std

- Collects `predicted_difference` from all inference outputs (e.g., 100 spectra)
- Computes **median** spectrum: `median_diff = np.median(predicted_diffs, axis=0)`
- Computes **standard deviation**: `std_diff = np.std(predicted_diffs, axis=0)`
- **Result**: One robust median spectrum representing all locations

### 6. Validity check - skip fitting if no signal

- Checks if `max(median_spectrum) < 0.05`
- **If true** (no growth): Skip fitting, assign loss = 4.0, save minimal results with reason
- **If false** (material present): Continue to fitting

### 7. [If valid] Fit median spectrum

- Sets up Voigt profile model with hardcoded peaks (B, L, G, 2D for loss calculation)
- Uses `curve_fit` with bounded Levenberg-Marquardt optimization
- Fits all peaks simultaneously
- Extracts covariance matrix for error estimates

### 8. Calculate loss function

- Extracts peak areas from fitted parameters
- **Loss_term_1** = Area(2D) / Area(G) → should be ≈ 0 (2D suppressed)
- **Loss_term_2** = (Area(B) + Area(G) + Area(L)) / sum(all areas) → should be ≈ 1 (structural dominant)
- **Loss** = (Loss_term_1 + Loss_term_2)² → range [0, 4]
- Logs detailed area values for interpretation

### 9. Save results and create visualizations

- Saves JSON with peak parameters, uncertainties, quality metrics, AND loss function
- Saves CSV with all peak parameters
- Creates 2-panel visualization showing median spectrum, fit, ±1std band, and individual components
- If no fitting: minimal JSON output with loss=4.0 reason

## Loss Function Reference

The loss function is designed to quantify material growth quality by measuring the spatial homogeneity of structural (B, G, L) vs non-structural (2D) Raman peaks.

### Mathematical Definition

```
Loss_term_1 = Area(2D peak) / Area(G peak)
Loss_term_2 = (Area(B) + Area(G) + Area(L)) / Sum(All Areas)
Loss = (Loss_term_1 + Loss_term_2)²
Range: [0, 4]
```

### Physical Interpretation

| Loss Value | Interpretation | Implication |
|--|--|--|
| ≈ 0 | Perfect growth | 2D suppressed, all signal in D-bands |
| 0.1 - 0.5 | Excellent growth | Mostly structural, minimal 2D |
| 0.5 - 1.5 | Good growth | Structural dominant but some 2D present |
| 1.5 - 3.0 | Mixed growth | Significant 2D and structural signal |
| 4.0 | No growth | Max intensity < 0.05, fitting skipped |

### Multi-Location Measurement Value

By aggregating measurements from multiple substrate locations:
- Loss < 1.0 from all locations → **Homogeneous growth**
- Loss varies significantly → **Inhomogeneous growth** (process optimization needed)
- Some locations loss = 4.0 → **Partial/selective growth** (spatial patterning)

## Physics Background

### Voigt Profile

The Voigt profile is a convolution of Gaussian and Lorentzian profiles,
representing the combined effects of:

- **Gaussian component**: Instrument broadening, thermal effects
- **Lorentzian component**: Homogeneous line broadening, lifetime effects

The `gamma_ratio` parameter controls the blend:

- gamma_ratio = 0.0 → Pure Gaussian
- gamma_ratio = 0.5 → Mixed Gaussian-Lorentzian
- gamma_ratio = 1.0 → Pure Lorentzian

### Median vs Mean

Individual spectra in a Raman map often have low signal-to-noise ratio (SNR).
By computing **median** across all aggregated spectra:

- Noise is reduced by √N (where N = total number of spectra)
- **Median is more robust** to outliers and cosmic rays than mean
- Standard deviation quantifies point-wise uncertainty
- Fitting becomes more stable and reproducible
- Visualization shows uncertainty band as median ± 1 standard deviation

## Troubleshooting

### Issue: "Input folder not found"

**Solution**: Verify folder path exists and contains `.txt` files.

```bash
ls data/substrate_measurements/*.txt  # Check files exist
```

### Issue: "No .txt files found"

**Solution**: Ensure `.txt` files are in root of input folder (not in subfolders).

### Issue: "Fitting failed: singular matrix in solve"

**Solution**: Peak parameters may be poorly constrained. Edit `DEFAULT_FITTING`:
- Reduce `center_windows` to tighten initial guesses
- Increase `min_widths` or `max_widths`

### Issue: "Checkpoint not found"

**Solution**: Verify checkpoint exists and path is correct:

```bash
ls ./saved_models/checkpoints/spectraformer:min70_highf/
```

Update `DEFAULT_CKPTS_DIR` and `DEFAULT_MODEL` in script if needed.

### Issue: "Loss = 4.0 for all locations"

**Solution**: Maximum intensity < 0.05 expected. Check:
- SpectraFormer model is appropriate for your material
- Preprocessing is not over-normalizing the signal
- Input spectra have sufficient intensity range

## Integration with Existing Pipeline

This script provides end-to-end functionality with reused components from:

- **`data_parser_script.py`**: `.txt` file parsing via `parse_dataset()`
- **`dashboard_gradio.py`**: Model loading and inference logic via `load_model_and_predict()`
- **`PL_fitting_individual_comprehensive_2.py`**: Voigt profile definitions and fitting via `multi_voigt_free_gamma()`, `batch_voigt_profiles()`, `fit_averaged_spectrum()`

**New multi-map specific functions:**

- `load_all_txt_files()`: Scan folder for .txt files
- `aggregate_preprocessed_maps()`: Concatenate multiple processed maps along the 'spectra' dimension
- `calculate_loss()`: Compute loss function from peak areas for growth quality assessment
- `save_results_no_fit()`: Handle case when no material is detected
- `create_fit_visualization_no_fit()`: Visualization without fitting

## Requirements

```text
jax>=0.4.0
jaxlib>=0.4.0
flax>=0.7.0
xarray>=2022.0.0
numpy>=1.23.0
scipy>=1.10.0
matplotlib>=3.6.0
orbax-checkpoint>=0.3.0
optax>=0.14.0
```

Install via:

```bash
pip install -r requirements.txt
```

## Example Output Log

```text
2026-03-18 14:30:00 [INFO] Step 1: Loading Raman map
2026-03-18 14:30:00 [INFO]   ✓ Loaded data shape: (5, 5, 1800)
2026-03-18 14:30:00 [INFO]   ✓ Dimensions: ['X_0', 'X_1', 'wave_number']
2026-03-18 14:30:01 [INFO] Step 2: Preprocessing
2026-03-18 14:30:01 [INFO]   ✓ Preprocessed shape: (25, 1800)
2026-03-18 14:30:02 [INFO] Step 2b: Filtering outliers
2026-03-18 14:30:02 [INFO]   ✓ Filtered shape: (25, 1800)
2026-03-18 14:30:05 [INFO] Step 3: Running SpectraFormer inference
2026-03-18 14:30:05 [INFO] Successful checkpoint load (NEW format)
2026-03-18 14:30:05 [INFO] Running inference...
2026-03-18 14:30:08 [INFO] Inference complete. Generated 25 predictions
2026-03-18 14:30:08 [INFO] Step 4: Extracting median and std of predicted_difference
2026-03-18 14:30:08 [INFO]   ✓ Computed from 25 spectra
2026-03-18 14:30:08 [INFO]   ✓ Median spectrum shape: (1800,)
2026-03-18 14:30:08 [INFO]   ✓ Value range: [-0.0523, 0.1847]
2026-03-18 14:30:08 [INFO]   ✓ Std range: [0.0012, 0.0234]
2026-03-18 14:30:08 [INFO] Step 5: Fitting median spectrum with Voigt profiles
2026-03-18 14:30:08 [INFO]   Fitting 6 peaks with 24 parameters...
2026-03-18 14:30:12 [INFO]   ✓ Fitting successful!
2026-03-18 14:30:12 [INFO] Step 6: Saving results
2026-03-18 14:30:12 [INFO]   ✓ JSON: fitting_results.json
2026-03-18 14:30:12 [INFO]   ✓ CSV: fitting_parameters.csv
2026-03-18 14:30:12 [INFO]   ✓ PNG: fitting_visualization.png
2026-03-18 14:30:12 [INFO] ============================================
2026-03-18 14:30:12 [INFO] ✓ Pipeline completed (12.34 seconds)
2026-03-18 14:30:12 [INFO] ✓ Results saved to: results/output
2026-03-18 14:30:12 [INFO] ============================================
```

## Citation

If you use this pipeline, please cite:

- SpectraFormer: [Your paper reference]
- Original Voigt fitting principles: Olivero & Longbothum (1977), JQSRT

## Author Notes

- **January 2026**: Initial pipeline development
- **March 2026**: Added averaging strategy for low-SNR maps
- **March 2026**: Integrated with SpectraFormer inference pipeline
- **March 2026**: Added outlier filtering
- **March 2026**: Simplified CLI to hardcoded parameters
- **March 2026**: Switched to median, added std uncertainty band
- **April 2026**: Upgraded to multi-file aggregation for homogeneity analysis
- **April 2026**: Folder-based input with spatial dimension flattening
- **April 2026**: Single inference pass on aggregated dataset
- **April 2026**: Validity check skips fitting if max intensity < 0.05 (loss = 4.0)
- **April 2026**: Loss function calculation based on B/G/L/2D peak area ratios
