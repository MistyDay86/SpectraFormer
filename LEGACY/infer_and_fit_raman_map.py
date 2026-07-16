#!/.venv/bin/env python3
"""
Run SpectraFormer inference on one folder of Raman .txt maps, then fit Voigt peaks
on the aggregated median predicted-difference spectrum.

Pipeline summary:
1) Load all .txt maps from an input folder
2) Preprocess each map with `whitaker_hayes_with_outliers`
3) Aggregate all spectra and run one inference pass
4) Compute median/std across predictions
5) Skip fitting when signal is below threshold, otherwise run constrained Voigt fit
6) Compute peak-ratio loss and save JSON/CSV/PNG outputs

Output path is auto-generated as:
`<output_root>/<material>/<sample>/<YYYYMMDD_HHMMSS>/`
where material/sample are inferred from input path relative to `data/raw_data`.

Usage examples:
    python infer_and_fit_raman_map.py --model-tag "min79_highf" --input "data/raw_data/buffer+graphene/G1850A11"
    python infer_and_fit_raman_map.py --model-tag "min79_highf" --input "data/raw_data/SiC-high-f/6H_spectra_20250423/5s_5p" --no-mask-shading
    python infer_and_fit_raman_map.py --model-tag "min79_highf" --input "data/raw_data/buffer+graphene/20260410_buffer2" --output-root "temp/fit_output"
"""

import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import wofz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import rcParams
rcParams['font.size'] = 24

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Hardcoded peak parameters for graphene Raman spectra
peak_names = {
    "D": 1360,
    "B": 1492,
    "L": 1564,
    "G": 1607,
    "2D": 2735,
}

DEFAULT_PEAKS = {
    "centers": list(peak_names.values()),
    "widths": [40.0] * len(peak_names),
    "gamma_ratios": [0.01] * len(peak_names),
    "peak_names": list(peak_names.keys()),
}

DEFAULT_FITTING = {
    "center_windows": [20.0] * len(DEFAULT_PEAKS["peak_names"]),
    "min_widths": [25.0] * len(DEFAULT_PEAKS["peak_names"]),
    "max_widths": [250, 100, 50, 100, 250],
    "maxfev": 30000,
}

DEFAULT_MODEL = "min79_highf"
DEFAULT_CKPTS_DIR = "./saved_models/checkpoints"
DEFAULT_CONFIGS_DIR = "./configs"
DEFAULT_OUTPUT_ROOT = "temp/fit_output"

PENALTY_LOSS = 100.0
FIT_THRESHOLD = 0.1


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_dataset(path: str) -> xr.DataArray:
    """Load one ASCII map file into an xarray with spatial dims + `wave_number`."""
    _data = np.loadtxt(path, unpack=True)
    wave_number, _counts = _data[-2], _data[-1]
    num_coords = len(_data) - 2
    coords = [_data[idx] for idx in range(num_coords)]
    unique_coords = [np.unique(coord, return_inverse=True) for coord in coords]
    unique_coords += [np.unique(wave_number, return_inverse=True)]
    idxs, values = [coord[1] for coord in unique_coords], [coord[0] for coord in unique_coords]
    counts_shape = tuple([len(coord[0]) for coord in unique_coords])
    counts = np.zeros(counts_shape, dtype=_counts.dtype)

    for z in zip(*idxs, _counts):
        counts[tuple(z[:-1])] = z[-1]

    dimension_names = [f'X_{i}' for i in range(len(values) - 1)] + ['wave_number']
    counts = xr.DataArray(counts, coords=values, dims=dimension_names)
    counts.wave_number.attrs['units'] = 'cm^{-1}'
    return counts


def load_all_txt_files(folder_path: Path) -> List[xr.DataArray]:
    """Parse all `*.txt` files in `folder_path` and return them in sorted order."""
    txt_files = sorted(folder_path.glob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found in {folder_path}")
        return []
    
    dataarrays = []
    for txt_file in txt_files:
        logger.info(f"  Loading: {txt_file.name}")
        da = parse_dataset(str(txt_file))
        dataarrays.append(da)
    
    return dataarrays


def aggregate_preprocessed_maps(dataarrays: List[xr.DataArray]) -> xr.DataArray:
    """Concatenate preprocessed maps along `spectra` for batch inference."""
    # Concatenate along spectra dimension
    aggregated = xr.concat(dataarrays, dim='spectra')
    
    logger.info(f"  ✓ Aggregated shape: {aggregated.shape}")
    logger.info(f"  ✓ Total spectra: {aggregated.sizes['spectra']}")
    
    return aggregated


# ============================================================================
# VOIGT PROFILE FITTING (Adapted from PL_fitting_individual_comprehensive_2.py)
# ============================================================================

def true_vectorized_voigt(
    x: np.ndarray, 
    centers: np.ndarray, 
    sigmas: np.ndarray,
    gammas: np.ndarray, 
    areas: np.ndarray
) -> np.ndarray:
    """Compute area-normalized Voigt profiles for many peaks using broadcasting."""
    # Reshape for broadcasting: (n_spectra, 1) vs (1, n_energy)
    x_reshaped = x.reshape(1, -1)
    centers_reshaped = centers.reshape(-1, 1)
    sigmas_reshaped = sigmas.reshape(-1, 1)
    gammas_reshaped = gammas.reshape(-1, 1)
    
    # Compute z for all spectra and energies at once
    z = ((x_reshaped - centers_reshaped) + 1j * gammas_reshaped) / (sigmas_reshaped * np.sqrt(2.0))
    
    # Compute Faddeeva function for all values
    w_values = wofz(z)
    
    # Compute profiles
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    profiles = np.real(w_values) / (sigmas_reshaped * sqrt_2pi)
    
    # Scale by areas
    return profiles * areas.reshape(-1, 1)


def batch_voigt_profiles(
    x: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    areas: np.ndarray,
    gamma_fracs: np.ndarray,
    min_width: float = 1e-10
) -> np.ndarray:
    """Evaluate Voigt profiles from center/FWHM/area/lorentz-fraction parameters."""
    n_spectra = len(centers)
    if n_spectra == 0:
        return np.zeros((0, len(x)))
    
    sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
    
    # Clamp parameters
    widths = np.maximum(widths, min_width)
    gamma_fracs = np.clip(gamma_fracs, 0.0, 1.0)
    
    sigma = np.empty(n_spectra, dtype=np.float64)
    gamma = np.empty(n_spectra, dtype=np.float64)
    
    # Pure Gaussian (gamma_frac ≈ 0)
    is_gaussian = gamma_fracs < 1e-8
    if np.any(is_gaussian):
        sigma[is_gaussian] = widths[is_gaussian] / (2.0 * sqrt_2ln2)
        gamma[is_gaussian] = 1e-15
    
    # Pure Lorentzian (gamma_frac ≈ 1)
    is_lorentz = gamma_fracs > (1.0 - 1e-8)
    if np.any(is_lorentz):
        gamma[is_lorentz] = widths[is_lorentz] / 2.0
        sigma[is_lorentz] = 1e-15
    
    # Mixed profiles (Olivero-Longbothum decomposition)
    is_mixed = ~(is_gaussian | is_lorentz)
    if np.any(is_mixed):
        V = widths[is_mixed]
        f = gamma_fracs[is_mixed]
        L = f * V
        term = V - 0.5346 * L
        G_sq = np.maximum(0.0, term**2 - 0.2166 * L**2)
        G = np.sqrt(G_sq)
        sigma[is_mixed] = G / (2.0 * sqrt_2ln2)
        gamma[is_mixed] = L / 2.0
    
    sigma = np.maximum(sigma, 1e-15)
    gamma = np.maximum(gamma, 1e-15)
    
    result = true_vectorized_voigt(x, centers, sigma, gamma, areas)
    return result


def voigt_profile(
    x: np.ndarray, 
    center: float, 
    voigt_fwhm: float, 
    area: float,
    lorentz_frac: float, 
    min_width: float = 1e-10
) -> np.ndarray:
    """Convenience wrapper returning one Voigt profile over axis `x`."""
    return batch_voigt_profiles(
        x.reshape(1, -1),
        np.array([center]),
        np.array([voigt_fwhm]),
        np.array([area]),
        np.array([lorentz_frac]),
        min_width
    )[0]


def multi_voigt_free_gamma(x: np.ndarray, *flat_params: float) -> np.ndarray:
    """Return the sum of Voigt peaks from flattened 4-parameter groups."""
    x = np.asarray(x)
    y_model = np.zeros_like(x, dtype=float)
    n_params = len(flat_params)
    if n_params % 4 != 0:
        raise ValueError("flat_params length must be a multiple of 4")
    for i in range(0, n_params, 4):
        c, w, A, r = flat_params[i:i + 4]
        y_model += voigt_profile(x, c, w, A, r)
    return y_model


def calculate_fwhm(energy: np.ndarray, spectrum: np.ndarray, baseline_mode: Optional[str] = None) -> float:
    """Estimate FWHM by linear interpolation at half-height crossings."""
    x = np.asarray(energy, dtype=float)
    y = np.asarray(spectrum, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size < 3:
        return np.nan

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.nan

    # Sort if needed
    if not np.all(np.diff(x) > 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    peak_idx = int(np.nanargmax(y))
    peak_val = float(y[peak_idx])

    # Baseline handling
    if baseline_mode is None:
        half = peak_val * 0.5
    elif baseline_mode == "min":
        half = 0.5 * (peak_val + np.min(y))
    elif baseline_mode == "edge":
        n_edge = max(1, int(0.01 * x.size))
        edge_vals = np.concatenate([y[:n_edge], y[-n_edge:]])
        half = 0.5 * (peak_val + float(np.median(edge_vals)))
    else:
        raise ValueError("baseline_mode must be None, 'min', or 'edge'")

    # Find left crossing
    left_x = None
    for i in range(peak_idx - 1, -1, -1):
        y0 = y[i]
        y1 = y[i + 1]
        if (y0 - half) * (y1 - half) < 0:
            # Linear interpolation for crossing point
            left_x = x[i] + (x[i + 1] - x[i]) * (half - y0) / (y1 - y0)
            break

    # Find right crossing
    right_x = None
    for i in range(peak_idx, x.size - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if (y0 - half) * (y1 - half) < 0:
            right_x = x[i] + (x[i + 1] - x[i]) * (half - y0) / (y1 - y0)
            break

    if left_x is None or right_x is None:
        return np.nan

    return float(abs(right_x - left_x))


def build_initial_guesses_and_bounds(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    widths: List[float],
    amp_maxs: List[Optional[float]],
    gamma_ratios: List[float],
    center_windows: List[float],
    min_widths: List[float],
    max_widths: List[float]
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Build `curve_fit` initial parameters and bounds for all Voigt peaks."""
    p0_list, lb_list, ub_list = [], [], []
    
    for i, c0 in enumerate(centers):
        width = widths[i]
        amp_max = amp_maxs[i]
        gamma_ratio = gamma_ratios[i]
        center_window = center_windows[i]
        min_width = min_widths[i]
        max_width = max_widths[i]
        ymax = np.nanmax(y) if amp_max is None else amp_max
        
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        
        # Center
        p0_list.append(float(c0))
        lb_list.append(float(c0 - center_window))
        ub_list.append(float(c0 + center_window))
        
        # Width (physical bounds)
        width = max(min_width, min(width, max_width))
        p0_list.append(float(width))
        lb_list.append(float(min_width))
        ub_list.append(float(max_width))
        
        # Area with physics-based initialization
        height_guess = float(max(1e-12, ymax))
        
        if gamma_ratio < 0.1:
            area_factor = np.sqrt(2.0 * np.log(2.0)) / 2.0
        elif gamma_ratio > 0.9:
            area_factor = 2.0 / np.pi
        else:
            area_factor = 0.5 + gamma_ratio * (np.sqrt(2.0 * np.log(2.0)) / 2.0 - 0.5 + 2.0 / np.pi)
        
        A0 = max(1e-12, height_guess * width * area_factor)
        p0_list.append(A0)
        lb_list.append(1e-6)
        ub_list.append(1e6 * A0)
        
        # Gamma ratio (physical bounds)
        gamma_ratio = np.clip(gamma_ratio, 0.0, 1.0)
        p0_list.append(float(gamma_ratio))
        lb_list.append(0.01)
        ub_list.append(0.99)
    
    p0 = np.array(p0_list, dtype=float)
    lb = np.array(lb_list, dtype=float)
    ub = np.array(ub_list, dtype=float)
    return p0, (lb, ub)


# ============================================================================
# INFERENCE
# ============================================================================

def load_model_and_predict(
    checkpoint_tag: str,
    dataset: xr.DataArray,
    ckpts_path: Path,
    configs: Any
) -> Tuple[List[Dict[str, Any]], Any, List[Tuple[int, int]]]:
    """Restore a SpectraFormer checkpoint and run predictions on `dataset`."""
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from spectraformer.inference import predict
    from spectraformer.input_pipeline import batch_sampler
    from spectraformer.model import CustomTrainState, SpectraFormer

    # Build full checkpoint path
    full_tag = (
        f"spectraformer:{checkpoint_tag}"
        if not checkpoint_tag.startswith("spectraformer:")
        else checkpoint_tag
    )
    checkpoint_path = (ckpts_path / full_tag).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize checkpoint manager
    ckpt_options = ocp.CheckpointManagerOptions(
        read_only=True, save_interval_steps=0, create=False
    )
    ckpt_manager = ocp.CheckpointManager(checkpoint_path, options=ckpt_options)

    # Build learning rate schedule (required for checkpoint restoration)
    cosine_kwargs = []
    learning_rate = getattr(configs, "learning_rate", 1e-3)
    init_value = 0.1 * learning_rate
    peak_value = learning_rate
    warmup_steps = getattr(configs, "warmup_steps", 1000)
    decay_steps = getattr(configs, "decay_steps", 2000)
    decline_coeff = getattr(configs, "decline_coeff", 1)
    num_cycles = getattr(configs, "num_cycles", 100)

    for _ in range(num_cycles):
        end_value = decline_coeff * init_value
        cycle_dict = {
            "init_value": init_value,
            "peak_value": peak_value,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "end_value": end_value,
        }
        cosine_kwargs.append(cycle_dict)
        init_value = end_value
        peak_value *= decline_coeff

    learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)

    learning_rate_decay = getattr(configs, "learning_rate_decay", "Constant")
    if learning_rate_decay == "Multiple cosine decay cycles":
        tx = optax.adam(learning_rate=learning_rate_fn)
    else:
        tx = optax.adam(learning_rate=configs.learning_rate)

    # Get mask windows
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )

    # Create dummy batch
    dummy_example = next(batch_sampler(dataset, mask_windows, batch_size=1))

    # Initialize model
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
    )

    root_key = jax.random.key(seed=configs.root_rng_seed)
    _, params_key, _ = jax.random.split(key=root_key, num=3)

    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=False,
    )

    state = CustomTrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training",)),
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )

    # Restore checkpoint
    latest_step = ckpt_manager.latest_step()
    try:
        state = ckpt_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state),
        )
        logger.info(f"Successfully loaded checkpoint (NEW format)")
    except Exception as e_new:
        try:
            restored_dict = ckpt_manager.restore(
                latest_step,
                args=ocp.args.StandardRestore({"state": state})
            )
            state = restored_dict["state"]
            logger.info(f"Successfully loaded checkpoint (OLD format)")
        except Exception as e_old:
            raise ValueError(
                f"Failed to restore checkpoint in both formats.\n"
                f"New format error: {str(e_new)}\n"
                f"Old format error: {str(e_old)}"
            )

    # Run predictions
    logger.info("Running inference...")
    test_data = list(batch_sampler(dataset, mask_windows, shuffle=False, batch_size=1))
    predictions = [
        predict(
            state.apply_fn,
            {"params": state.params},
            datapoint,
            datapoint["mask"],
        )
        for datapoint in test_data
    ]

    return predictions, configs, mask_windows


def build_output_dir_from_input(input_folder: Path, output_root: Path) -> Path:
    """Create timestamped output directory inferred from input path structure."""
    parts = input_folder.resolve().parts
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if "raw_data" in parts:
        raw_idx = parts.index("raw_data")
        if len(parts) >= raw_idx + 3:
            material = parts[raw_idx + 1]
            sample_parts = parts[raw_idx + 2:]
            return output_root / material / Path(*sample_parts) / timestamp

    logger.warning(
        "Could not infer material/sample from input path relative to 'raw_data'. "
        "Using fallback path unknown_material/<input_folder_name>."
    )
    return output_root / "unknown_material" / input_folder.name / timestamp


# ============================================================================
# FITTING & LOSS CALCULATION
# ============================================================================

def fit_averaged_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    widths: List[float],
    amp_maxs: List[Optional[float]],
    gamma_ratios: List[float],
    center_windows: List[float],
    min_widths: List[float],
    max_widths: List[float],
    maxfev: int = 20000,
    tolerances: Optional[Dict[str, float]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Fit the median spectrum with constrained multi-Voigt model parameters."""
    if tolerances is None:
        tolerances = {"ftol": 1e-5, "xtol": 1e-5, "gtol": 1e-5}
    
    # Validate spectrum
    if not np.any(np.isfinite(y)) or np.all(y == 0) or np.all(np.isnan(y)):
        return None, None, "Spectrum is invalid (NaN, all zeros, or no finite values)"
    
    # Build initial guesses and bounds
    p0, bounds = build_initial_guesses_and_bounds(
        x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths
    )
    
    try:
        logger.info(f"Fitting {len(centers)} peaks with {len(p0)} parameters...")
        popt, pcov = curve_fit(
            multi_voigt_free_gamma, x, y,
            p0=p0, bounds=bounds, maxfev=maxfev,
            ftol=tolerances.get('ftol', 1e-5),
            xtol=tolerances.get('xtol', 1e-5),
            gtol=tolerances.get('gtol', 1e-5)
        )
        logger.info("Fitting successful!")
        # Logging all the fitting parameters per each peak in human readable format
        for i in range(0, len(popt), 4):
            c, w, A, r = popt[i:i + 4]
            peak_name = peak_names[i // 4] if i // 4 < len(peak_names) else f"Peak {i//4 + 1}"
            logger.info(f"      ✓ Peak {peak_name}: Center={c:.2f}, Width={w:.2f}, Area={A:.2f}, Gamma_frac={r:.3f}")
        return popt, pcov, "Success"
    except Exception as exc:
        error_msg = f"Fitting failed: {exc}"
        logger.warning(error_msg)
        return None, None, error_msg

def loss_term(value, target: float = 0.0):
    """Return squared error to target for one scalar loss component."""
    return ((value - target)**2)


def calculate_loss(popt: np.ndarray, peak_names: List[str], centers: List[float]) -> Dict[str, float]:
    """Compute growth loss from fitted peak areas using three ratio-based terms."""
    # Extract peak indices for B, G, L, 2D
    try:
        idx_B = peak_names.index('B')
        idx_G = peak_names.index('G')
        idx_L = peak_names.index('L')
        idx_2D = peak_names.index('2D')
    except ValueError as e:
        logger.warning(f"Could not find required peaks for loss calculation: {e}")
        return {
            'loss': 2.0,
            'loss_term_1': np.nan,
            'loss_term_2': np.nan,
            'loss_term_3': np.nan,
            'error': f'Missing peak: {e}'
        }
    
    # Extract areas (parameter index 2, 6, 10, ... for each peak)
    area_B = popt[idx_B * 4 + 2]
    area_G = popt[idx_G * 4 + 2]
    area_L = popt[idx_L * 4 + 2]
    area_2D = popt[idx_2D * 4 + 2]
    area_all = np.sum(popt[2::4])
    area_BLG = area_B + area_L + area_G
    
    # Calculate loss terms
    # Loss term 1: Area(2D) / Area(G) -> should be 0 (no competition with G)
    if area_G > 0:
        loss_term_1_value = area_2D / area_G
    else:
        loss_term_1_value = np.inf if area_2D > 0 else 0.0
    
    # Loss term 2: Area(B) / Area(G) -> should be 1 (equal contribution)
    if area_G > 0:
        loss_term_2_value = area_B / area_G
    else:
        loss_term_2_value = np.inf if area_B > 0 else 0.0

    # Loss term 3: Area(B+L+G) / Total area -> should be 1
    if area_all > 0:
        loss_term_3_value = area_BLG / area_all
    else:
        loss_term_3_value = 0.0
    
    loss_term_1 = loss_term(loss_term_1_value, target=0.0)
    loss_term_2 = loss_term(loss_term_2_value, target=1.0)
    loss_term_3 = loss_term(loss_term_3_value, target=1.0)
    loss = (loss_term_1 + loss_term_2 + loss_term_3) ** 2
    
    logger.info(f"  ✓ Loss components:")
    logger.info(f"    - Area(2D): {area_2D}")
    logger.info(f"    - Area(G): {area_G}")
    logger.info(f"    - Area(B): {area_B}")
    logger.info(f"    - Area(L): {area_L}")
    logger.info(f"    - Area(B+L+G): {area_BLG}")
    logger.info(f"    - Total area (all peaks): {area_all}")
    logger.info(f"    - Area(2D)/Area(G): {area_2D/area_G if area_G > 0 else 'inf'}")
    logger.info(f"    - Area(B)/Area(G): {area_B/area_G if area_G > 0 else 'inf'}")
    logger.info(f"    - Area(B+L+G)/Total: {loss_term_3_value}")
    logger.info(f"    - Loss_term_1 (2D/G): {loss_term_1}")
    logger.info(f"    - Loss_term_2 (B/G): {loss_term_2}")
    logger.info(f"    - Loss_term_3 ((B+L+G)/Total): {loss_term_3}")
    logger.info(f"    - Loss total: {loss}")
    
    return {
        'loss': float(loss),
        'loss_term_1': float(loss_term_1),
        'loss_term_2': float(loss_term_2),
        'loss_term_3': float(loss_term_3),
        'ratio_BLG_over_total': float(loss_term_3_value),
        'area_2D': float(area_2D),
        'area_G': float(area_G),
        'area_B': float(area_B),
        'area_L': float(area_L),
        'sum_areas': float(area_all)
    }



def create_fit_visualization(
    x: np.ndarray,
    y: np.ndarray,
    popt: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    output_path: Path,
    y_std: Optional[np.ndarray] = None,
    mask_intervals: Optional[List[Tuple[int, int]]] = None
):
    """Plot median spectrum, optional std shading, visible (unmasked) spans, fit, and components."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    
    # Calculate fit and components
    y_fit = multi_voigt_free_gamma(x, *popt)
    
    n_peaks = len(centers)
    colors = [f'C{i}' for i in range(n_peaks)]
    
    # Plot 2: Individual components
    ax.plot(x, y, 'ko-', linewidth=2, label='Median', markersize=3, alpha=0.7)
    
    
    # Add ±1std shaded region if std provided
    if y_std is not None:
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.15, color='blue', label='±1 std')

    if mask_intervals:
        x_min = float(np.min(x))
        x_max = float(np.max(x))

        hidden_intervals = []
        for s, e in mask_intervals:
            start = float(min(s, e))
            end = float(max(s, e))
            span_start = max(start, x_min)
            span_end = min(end, x_max)
            if span_end > span_start:
                hidden_intervals.append((span_start, span_end))

        hidden_intervals.sort(key=lambda t: t[0])
        merged_hidden = []
        for start, end in hidden_intervals:
            if not merged_hidden or start > merged_hidden[-1][1]:
                merged_hidden.append([start, end])
            else:
                merged_hidden[-1][1] = max(merged_hidden[-1][1], end)

        shown_intervals = []
        cursor = x_min
        for start, end in merged_hidden:
            if start > cursor:
                shown_intervals.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < x_max:
            shown_intervals.append((cursor, x_max))
        if not merged_hidden:
            shown_intervals = [(x_min, x_max)]

        for start, end in shown_intervals:
            ax.axvspan(
                start, end,
                color="gray", alpha=0.1, linewidth=0
            )
    
    ax.plot(x, y_fit, 'r-', linewidth=3.5, label='Fit')
    
    for i in range(n_peaks):
        params = popt[i*4:(i+1)*4]
        peak_y = voigt_profile(x, params[0], params[1], params[2], params[3])
        baseline = -0.10
        ax.plot(x, peak_y + baseline, '--', linewidth=2.5, color=colors[i], label=f'{peak_names[i]}')
        ax.fill_between(x, peak_y + baseline, baseline, alpha=0.1, color=colors[i])
    
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize='large')
    ax.set_ylabel('Intensity (a.u.)', fontsize='large')
    ax.legend(fontsize='large', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_ylim(-0.10, 0.25)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to: {output_path}")
    plt.close()


def create_fit_visualization_no_fit(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    output_path: Path,
    y_std: Optional[np.ndarray] = None,
    mask_intervals: Optional[List[Tuple[int, int]]] = None
):
    """Plot median spectrum without fitted peaks, including threshold and visible (unmasked) spans."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    
    # Plot spectrum
    ax.plot(x, y, 'ko-', linewidth=2, label='Median', markersize=3, alpha=0.7)
    
    # Add ±1std shaded region if std provided
    if y_std is not None:
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.25, color='blue', label='±1 std')

    if mask_intervals:
        x_min = float(np.min(x))
        x_max = float(np.max(x))

        hidden_intervals = []
        for s, e in mask_intervals:
            start = float(min(s, e))
            end = float(max(s, e))
            span_start = max(start, x_min)
            span_end = min(end, x_max)
            if span_end > span_start:
                hidden_intervals.append((span_start, span_end))

        hidden_intervals.sort(key=lambda t: t[0])
        merged_hidden = []
        for start, end in hidden_intervals:
            if not merged_hidden or start > merged_hidden[-1][1]:
                merged_hidden.append([start, end])
            else:
                merged_hidden[-1][1] = max(merged_hidden[-1][1], end)

        shown_intervals = []
        cursor = x_min
        for start, end in merged_hidden:
            if start > cursor:
                shown_intervals.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < x_max:
            shown_intervals.append((cursor, x_max))
        if not merged_hidden:
            shown_intervals = [(x_min, x_max)]

        for start, end in shown_intervals:
            ax.axvspan(
                start, end,
                color="gray", alpha=0.1, linewidth=0
            )
    
    ax.axhline(FIT_THRESHOLD, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Growth threshold')
    
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize='large')
    ax.set_ylabel('Intensity (a.u.)', fontsize='large')
    ax.legend(fontsize='large', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_ylim(-0.10, 0.25)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization (no fit) to: {output_path}")
    plt.close()


def save_results_no_fit(
    output_dir: Path,
    centers: List[float],
    peak_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    predictions: List[Dict[str, Any]],
    configs: Any,
    loss_value: float = 4.0,
    loss_reason: str = "No material growth detected"
):
    """Write JSON/CSV outputs for the no-fit path when signal is below threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_peaks = len(centers)
    
    # Build results JSON
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_peaks": n_peaks,
            "peak_names": peak_names,
            "n_spectra_averaged": len(predictions),
            "fitting_performed": False,
            "fitting_skipped_reason": loss_reason,
            "model_config": {
                "num_heads": getattr(configs, "num_heads", None),
                "num_layers": getattr(configs, "num_layers", None),
                "embedding_dim": getattr(configs, "embedding_dim", None),
            }
        },
        "peaks": [],
        "quality_metrics": {
            "r_squared": np.nan,
            "sum_squared_error": np.nan,
            "mean_absolute_error": np.nan,
            "note": "No fitting performed - skipped due to low signal"
        },
        "loss_function": {
            "loss": float(loss_value),
            "loss_reason": loss_reason,
            "loss_term_1": np.nan,
            "loss_term_2": np.nan,
            "loss_term_3": np.nan,
            "description": {
                "note": "No material growth detected",
                "max_intensity": float(np.max(y)),
                "threshold": FIT_THRESHOLD
            }
        }
    }
    
    # Save JSON
    json_path = output_dir / "fitting_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results (no fit) to: {json_path}")
    
    # Save CSV placeholder
    csv_path = output_dir / "fitting_parameters.csv"
    with open(csv_path, 'w') as f:
        f.write("note,value\n")
        f.write(f"fitting_performed,false\n")
        f.write(f"reason,{loss_reason}\n")
        f.write(f"max_intensity,{np.max(y):.6e}\n")
        f.write(f"threshold,{FIT_THRESHOLD}\n")
        f.write(f"loss,{loss_value:.4f}\n")
    logger.info(f"Saved CSV results (no fit) to: {csv_path}")

def save_results(
    output_dir: Path,
    popt: np.ndarray,
    pcov: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    predictions: List[Dict[str, Any]],
    configs: Any,
    loss_dict: Optional[Dict[str, float]] = None
):
    """Write fitting parameters, metrics, and optional loss to JSON/CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_peaks = len(centers)
    
    # Build results JSON
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_peaks": n_peaks,
            "peak_names": peak_names,
            "n_spectra_averaged": len(predictions),
            "model_config": {
                "num_heads": getattr(configs, "num_heads", None),
                "num_layers": getattr(configs, "num_layers", None),
                "embedding_dim": getattr(configs, "embedding_dim", None),
            }
        },
        "peaks": []
    }
    
    # Add peak results
    uncertainties = np.sqrt(np.diagonal(pcov))
    param_names = ['center', 'width', 'area', 'gamma_ratio']
    
    for i, peak_name in enumerate(peak_names):
        peak_data = {
            "name": peak_name,
            "expected_center_cm_inv": centers[i],
            "parameters": {}
        }
        for j, param_name in enumerate(param_names):
            idx = i * 4 + j
            peak_data["parameters"][param_name] = {
                "value": float(popt[idx]),
                "uncertainty": float(uncertainties[idx])
            }
        results["peaks"].append(peak_data)
    
    # Calculate quality metrics
    y_fit = multi_voigt_free_gamma(x, *popt)
    residuals = y - y_fit
    sse = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (sse / sst) if sst > 0 else np.nan
    
    results["quality_metrics"] = {
        "r_squared": float(r_squared),
        "sum_squared_error": float(sse),
        "mean_absolute_error": float(np.mean(np.abs(residuals)))
    }
    
    # Add loss function results if provided
    if loss_dict is not None:
        results["loss_function"] = {
            "loss": loss_dict.get('loss', np.nan),
            "loss_term_1": loss_dict.get('loss_term_1', np.nan),
            "loss_term_2": loss_dict.get('loss_term_2', np.nan),
            "loss_term_3": loss_dict.get('loss_term_3', np.nan),
            "ratio_BLG_over_total": loss_dict.get('ratio_BLG_over_total', np.nan),
            "area_2D": loss_dict.get('area_2D', np.nan),
            "area_G": loss_dict.get('area_G', np.nan),
            "area_B": loss_dict.get('area_B', np.nan),
            "area_L": loss_dict.get('area_L', np.nan),
            "total_area": loss_dict.get('sum_areas', np.nan),
            "description": {
                "loss_term_1": "Area(2D) / Area(G) - should be 0 for good growth",
                "loss_term_2": "Area(B) / Area(G) - should be 1 for structural",
                "loss_term_3": "Area(B+L+G) / Total area - should be 1"
            }
        }
    
    # Save JSON
    json_path = output_dir / "fitting_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to: {json_path}")
    
    # Save CSV with parameter details
    csv_path = output_dir / "fitting_parameters.csv"
    with open(csv_path, 'w') as f:
        f.write("peak_name,parameter,value,uncertainty\n")
        for i, peak_name in enumerate(peak_names):
            for j, param_name in enumerate(param_names):
                idx = i * 4 + j
                f.write(f"{peak_name},{param_name},{popt[idx]:.6e},{uncertainties[idx]:.6e}\n")
    logger.info(f"Saved CSV results to: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Parse CLI arguments and run end-to-end inference, fitting, and export."""
    parser = argparse.ArgumentParser(
        description="SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python infer_and_fit_raman_map.py --model-tag "min79_highf" --input "data/substrate_measurements/"

Peak parameters are hardcoded. To modify, edit DEFAULT_PEAKS in the script.
        """
    )
    parser.add_argument(
        '--model-tag', type=str, default=DEFAULT_MODEL,
        help='Model tag used for config/checkpoint lookup (default: min79_highf)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to folder containing .txt files with Raman spectral maps'
    )
    parser.add_argument(
        '--output-root', type=str, default=DEFAULT_OUTPUT_ROOT,
        help='Root output directory (default: temp/fit_output)'
    )
    parser.add_argument(
        '--mask-shading',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show masked intervals as light shading on plots (default: enabled)'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Validate input
    input_folder = Path(args.input)
    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return
    
    output_root = Path(args.output_root)
    output_dir = build_output_dir_from_input(input_folder, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpts_path = Path(DEFAULT_CKPTS_DIR)
    checkpoint = args.model_tag
    config_path = Path(DEFAULT_CONFIGS_DIR) / f"configs_{checkpoint}.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    try:
        import ml_confs
    except ModuleNotFoundError:
        logger.error(
            "Package 'ml_confs' is required to load model config files. "
            "Install it in your environment and rerun."
        )
        return

    configs = ml_confs.from_file(config_path)
    checkpoint_tag = getattr(configs, "tag", checkpoint)
    
    # Get peak parameters from defaults
    centers = DEFAULT_PEAKS["centers"]
    widths = DEFAULT_PEAKS["widths"]
    gamma_ratios = DEFAULT_PEAKS["gamma_ratios"]
    peak_names = DEFAULT_PEAKS["peak_names"]
    
    center_windows = DEFAULT_FITTING["center_windows"]
    min_widths = DEFAULT_FITTING["min_widths"]
    max_widths = DEFAULT_FITTING["max_widths"]
    maxfev = DEFAULT_FITTING["maxfev"]
    
    amp_maxs = [None] * len(centers)
    
    logger.info("="*70)
    logger.info("SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline")
    logger.info("="*70)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Model checkpoint: {checkpoint_tag}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mask shading: {'enabled' if args.mask_shading else 'disabled'}")
    logger.info(f"Peak parameters: {peak_names}")
    logger.info("="*70)
    
    try:
        # ====== STEP 1: Load all .txt files from folder ======
        logger.info(f"Step 1: Loading all .txt files from {input_folder}")
        dataarrays = load_all_txt_files(input_folder)
        if not dataarrays:
            logger.error("No valid .txt files loaded")
            return
        logger.info(f"  ✓ Loaded {len(dataarrays)} Raman maps")
        
        # ====== STEP 2: Preprocess maps ======
        logger.info("Step 2: Preprocessing individual maps with whitaker_hayes_with_outliers")
        from spectraformer.input_pipeline import preprocess_dataset
        
        preprocessed_arrays = []
        for i, da in enumerate(dataarrays):
            logger.info(f"  Preprocessing map {i+1}/{len(dataarrays)}")
            da_prep = preprocess_dataset(da, option="whitaker_hayes_with_outliers")
            preprocessed_arrays.append(da_prep)
        
        # ====== STEP 3: Aggregate all preprocessed maps ======
        logger.info("Step 3: Aggregating all preprocessed maps into single dataset")
        dataset_preprocessed = aggregate_preprocessed_maps(preprocessed_arrays)
        logger.info(f"  ✓ Aggregated preprocessed shape: {dataset_preprocessed.shape}")
        
        # ====== STEP 4: Inference ======
        logger.info("Step 4: Running single SpectraFormer inference on aggregated dataset")
        predictions, configs, mask_windows = load_model_and_predict(
            checkpoint_tag, dataset_preprocessed, ckpts_path, configs
        )
        logger.info(f"  ✓ Inference complete. Generated {len(predictions)} predictions")
        
        # ====== STEP 5: Extract and compute median & std of predicted_difference ======
        logger.info("Step 5: Extracting median and std of predicted_difference")
        predicted_diffs = np.array([p["predicted_difference"] for p in predictions])
        median_diff = np.median(predicted_diffs, axis=0)
        std_diff = np.std(predicted_diffs, axis=0)
        
        logger.info(f"  ✓ Computed from {len(predictions)} spectra")
        logger.info(f"  ✓ Median spectrum shape: {median_diff.shape}")
        logger.info(f"  ✓ Value range: [{np.min(median_diff):.6f}, {np.max(median_diff):.6f}]")
        logger.info(f"  ✓ Std range: [{np.min(std_diff):.6f}, {np.max(std_diff):.6f}]")
        
        # Get wave_number axis
        wave_number = np.asarray(predictions[0]["wave_number"])
        if np.max(np.abs(wave_number)) < 10:
            wave_number = wave_number * 800 + 2000
        
        # ====== STEP 6: Validity check - skip fitting if no signal ======
        logger.info("Step 6: Checking validity of spectrum for fitting")
        max_intensity = np.max(median_diff)
        logger.info(f"  ✓ Maximum intensity: {max_intensity:.6f}")
        
        if max_intensity < FIT_THRESHOLD:
            logger.warning(F"  ✗ Maximum intensity < {FIT_THRESHOLD} - No material growth detected!")
            logger.warning(F"  ✗ Skipping fitting and assigning loss = {PENALTY_LOSS} (no growth penalty)")
            
            # Save results with loss=4 and no fitting
            save_results_no_fit(
                output_dir, centers, peak_names,
                wave_number, median_diff, predictions, configs,
                loss_value=PENALTY_LOSS, loss_reason=f"No material growth (max_intensity < {FIT_THRESHOLD})"
            )
            
            # Create visualization even without fitting
            viz_path = output_dir / "fitting_visualization.png"
            mask_intervals_plot = mask_windows if args.mask_shading else None
            create_fit_visualization_no_fit(
                wave_number, median_diff, centers, peak_names, viz_path, std_diff, mask_intervals_plot
            )
            logger.info(f"  ✓ Saved visualization (no fit) to: {viz_path}")
            
            elapsed = time.time() - start_time
            logger.info("="*70)
            logger.info(f"✓ Pipeline completed (no fitting) in {elapsed:.2f} seconds")
            logger.info(f"✓ Results saved to: {output_dir}")
            logger.info(f"✓ Loss = {PENALTY_LOSS} (no growth)")
            logger.info("="*70)
            return
        
        # ====== STEP 7: Fit median spectrum ======
        logger.info("Step 7: Fitting median spectrum with Voigt profiles")
        popt, pcov, fit_status = fit_averaged_spectrum(
            wave_number, median_diff,
            centers, peak_names, widths, amp_maxs, gamma_ratios,
            center_windows, min_widths, max_widths,
            maxfev=maxfev
        )
        
        if popt is None:
            logger.error(f"  ✗ Fitting failed: {fit_status}")
            return
        
        logger.info(f"  ✓ {fit_status}")
        
        # ====== STEP 8: Calculate loss function ======
        logger.info("Step 8: Calculating loss function from peak areas")
        loss_dict = calculate_loss(popt, peak_names, centers)
        
        # ====== STEP 9: Save results and visualizations ======
        logger.info("Step 9: Saving results")
        save_results(
            output_dir, popt, pcov, centers, peak_names,
            wave_number, median_diff, predictions, configs, loss_dict
        )
        
        # Create visualization
        viz_path = output_dir / "fitting_visualization.png"
        mask_intervals_plot = mask_windows if args.mask_shading else None
        create_fit_visualization(
            wave_number, median_diff, popt, centers, peak_names, viz_path, std_diff, mask_intervals_plot
        )
        logger.info(f"  ✓ Saved visualization to: {viz_path}")
        
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info(f"✓ Pipeline completed successfully in {elapsed:.2f} seconds")
        logger.info(f"✓ Results saved to: {output_dir}")
        logger.info(f"✓ Loss = {loss_dict['loss']:.4f}")
        logger.info("="*70)
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
