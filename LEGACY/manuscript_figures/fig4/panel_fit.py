import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
from pathlib import Path
from matplotlib import rcParams

from typing import List, Tuple, Sequence, Optional, Dict, Any, Union
from scipy.optimize import curve_fit
from scipy.special import wofz

rcParams['font.size'] = 24

# -----------------------------
# Core plotting logic
# -----------------------------

def plot_spatial_dataset_mean(ds):
    fig = plt.figure(figsize=(12.5, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    wave_number = ds["wave_number"].values

    # Determine reduction dimensions (everything except wave_number)
    reduce_dims = [d for d in ds["spectra"].dims if d != "wave_number"]

    # -------------------------
    # Mask (union over space)
    # -------------------------
    mask_any = ds["mask"].any(dim=reduce_dims).values.astype(bool)

    mask_intervals = []
    start = None
    for i, v in enumerate(mask_any):
        if v and start is None:
            start = i
        elif not v and start is not None:
            mask_intervals.append((start, i - 1))
            start = None
    if start is not None:
        mask_intervals.append((start, len(mask_any) - 1))

    for s, e in mask_intervals:
        for ax in (ax1, ax2):
            ax.axvspan(
                wave_number[s], wave_number[e],
                color="gray", alpha=0.1, linewidth=0
            )

    # -------------------------
    # Top panel
    # -------------------------
    for key, label, color in [
        ("spectra", "Spectra", "C0"),
        ("predicted_spectra", "Predicted spectra", "C1"),
    ]:
        da = ds[key]

        mean = da.mean(dim=reduce_dims).values
        std = da.std(dim=reduce_dims).values

        ax1.plot(wave_number, mean, color=color, lw=1.2, label=label)
        ax1.fill_between(
            wave_number,
            mean - std,
            mean + std,
            color=color,
            alpha=0.4,
            linewidth=0
        )

    ax1.legend(frameon=True, fontsize='small')
    ax1.set_ylabel("Intensity, a.u.")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    # ax1.grid(True)

    # -------------------------
    # Bottom panel
    # -------------------------
    diff = ds["predicted_difference"]

    diff_mean = diff.mean(dim=reduce_dims).values
    diff_std = diff.std(dim=reduce_dims).values

    ax2.plot(wave_number, diff_mean, color="C2", lw=1.2, label="Predicted difference")
    ax2.fill_between(
        wave_number,
        diff_mean - diff_std,
        diff_mean + diff_std,
        color="C2",
        alpha=0.4,
        linewidth=0
    )

    ax2.axhline(0, color='k', alpha=0.4)
    ax2.legend(frameon=True, fontsize='small')
    ax2.set_xlabel("Raman shift, cm$^{-1}$")
    ax2.set_ylabel("Intensity, a.u.")
    # ymin_cur, ymax_cur = ax2.get_ylim()
    # ymin_req, ymax_req = -0.08, 0.18

    # if ymin_cur > ymin_req or ymax_cur < ymax_req:
    #     ax2.set_ylim(
    #         min(ymin_cur, ymin_req),
    #         max(ymax_cur, ymax_req)
    #     )
    ax2.set_ylim(-0.08, 0.32)
    # ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()

    return fig

def plot_mean_spectrum_with_voigt_fit(ds):
    # -------------------------
    # Parameters
    # -------------------------
    x_min, x_max = 1250.0, 1850.0

    rough_centers = np.array([1320, 1370, 1485, 1556, 1606], dtype=float)
    rough_width = 40.0
    min_width = 30.0
    max_width = 100.0
    rough_amp_max = None
    rough_gamma_ratio = 0.0
    center_window = 40.0
    peak_names = ['D1^*', 'D2^*', 'B', 'L', 'G', '2D']

    # -------------------------
    # Prepare data
    # -------------------------
    wave_number = ds["wave_number"].values

    reduce_dims = [d for d in ds["spectra"].dims if d != "wave_number"]

    spectra_mean = ds["spectra"].mean(dim=reduce_dims).values
    spectra_std  = ds["spectra"].std(dim=reduce_dims).values

    diff = ds["predicted_difference"]
    diff_mean = diff.mean(dim=reduce_dims).values
    diff_std  = diff.std(dim=reduce_dims).values

    # Restrict x-range
    mask = (wave_number >= x_min) & (wave_number <= x_max)

    x_fit = wave_number[mask]
    y_fit = diff_mean[mask]

    # -------------------------
    # Build fitting config
    # -------------------------
    centers = rough_centers
    widths = np.full_like(centers, rough_width)
    amp_maxs = [rough_amp_max] * len(centers)
    gamma_ratios = np.full_like(centers, rough_gamma_ratio)
    center_windows = np.full_like(centers, center_window)
    min_widths = np.full_like(centers, min_width)
    max_widths = np.full_like(centers, max_width)

    config = {
        "fitting": {
            "maxfev": 30000,
            "tolerances": dict(ftol=1e-6, xtol=1e-6, gtol=1e-6),
        }
    }
    def build_initial_guesses_and_bounds(
        x: np.ndarray, y: np.ndarray, centers: Sequence[float],
        widths: Sequence[float], amp_maxs: Sequence[Optional[float]],
        gamma_ratios: Sequence[float], center_windows: Sequence[float],
        min_widths: Sequence[float], max_widths: Sequence[float],
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build initial parameter guesses and bounds for curve fitting."""
        p0_list, lb_list, ub_list = [], [], []
        fitting_config = config.get('fitting', {})
        tol_config = fitting_config.get('tolerances', {})
        
        for i, c0 in enumerate(centers):
            width = widths[i]
            amp_max = amp_maxs[i]
            gamma_ratio = gamma_ratios[i]
            center_window = center_windows[i]
            min_width = min_widths[i]
            max_width = max_widths[i]
            ymax = np.nanmax(y) if amp_max is None else amp_max
            
            # Ensure numerical stability
            if not np.isfinite(ymax) or ymax <= 0:
                ymax = 1.0
            
            # Center
            p0_list.append(float(c0))
            lb_list.append(float(c0 - center_window))
            ub_list.append(float(c0 + center_window))
            
            # Width (ensure physical bounds)
            width = max(min_width, min(width, max_width))
            p0_list.append(float(width))
            lb_list.append(float(min_width))
            ub_list.append(float(max_width))
            
            # Area with physics-based initialization
            height_guess = float(max(1e-12, ymax))
            
            # Correct area factor for Voigt profiles
            if gamma_ratio < 0.1:  # Near-Gaussian
                area_factor = 1.064467  # sqrt(pi/(4*ln(2)))
            elif gamma_ratio > 0.9:  # Near-Lorentzian
                area_factor = np.pi / 2  # pi/2 for Lorentzian
            else:  # Mixed profile - non-linear interpolation
                G_factor = 1.064467
                L_factor = np.pi / 2
                # Non-linear weighting based on Voigt properties
                area_factor = G_factor * (1 - gamma_ratio)**1.5 + L_factor * gamma_ratio**1.5
            
            A0 = max(1e-12, height_guess * width * area_factor)
            p0_list.append(A0)
            lb_list.append(0.0)
            ub_list.append(1e6 * A0)
            
            # Gamma ratio (ensure physical bounds)
            gamma_ratio = np.clip(gamma_ratio, 0.0, 1.0)
            p0_list.append(float(gamma_ratio))
            lb_list.append(0.0)
            ub_list.append(1.0)
            
        p0 = np.array(p0_list, dtype=float)
        lb = np.array(lb_list, dtype=float)
        ub = np.array(ub_list, dtype=float)
        return p0, (lb, ub)
    
    def true_vectorized_voigt(x: np.ndarray, centers: np.ndarray, sigmas: np.ndarray, 
                            gammas: np.ndarray, areas: np.ndarray) -> np.ndarray:
        """
        Truly vectorized Voigt profile computation using broadcasting.
        
        Parameters:
        -----------
        x : np.ndarray
            Energy axis (shape: (n_energy,))
        centers : np.ndarray
            Peak centers (shape: (n_spectra,))
        sigmas : np.ndarray
            Gaussian standard deviations (shape: (n_spectra,))
        gammas : np.ndarray
            Lorentzian half-widths (shape: (n_spectra,))
        areas : np.ndarray
            Peak areas (shape: (n_spectra,))
        
        Returns:
        --------
        np.ndarray
            Voigt profiles (shape: (n_spectra, n_energy))
        """
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
        """
        Vectorized computation of multiple Voigt profiles with shared energy axis.
        Uses true vectorization via broadcasting for optimal performance.
        
        Parameters:
        -----------
        x : np.ndarray
            Energy axis (shape: (n_energy,))
        centers : np.ndarray 
            Peak centers (shape: (n_spectra,))
        widths : np.ndarray
            Voigt FWHM (shape: (n_spectra,))
        areas : np.ndarray
            Peak areas (shape: (n_spectra,))
        gamma_fracs : np.ndarray
            Lorentzian fractions (shape: (n_spectra,))
        min_width : float
            Minimum width for numerical stability
        
        Returns:
        --------
        np.ndarray
            Array of Voigt profiles (shape: (n_spectra, n_energy))
        """
        n_spectra = len(centers)
        if n_spectra == 0:
            return np.zeros((0, len(x)))
        
        # Pre-compute constants
        sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
        sqrt_2pi = np.sqrt(2.0 * np.pi)
        
        # Clamp parameters to physical ranges
        widths = np.maximum(widths, min_width)
        gamma_fracs = np.clip(gamma_fracs, 0.0, 1.0)
        
        # Pre-allocate parameters
        sigma = np.empty(n_spectra, dtype=np.float64)
        gamma = np.empty(n_spectra, dtype=np.float64)
        
        # Handle all cases using vectorized operations
        # Pure Gaussian profiles (gamma_frac ≈ 0)
        is_gaussian = gamma_fracs < 1e-8
        if np.any(is_gaussian):
            sigma[is_gaussian] = widths[is_gaussian] / (2.0 * sqrt_2ln2)
            gamma[is_gaussian] = 1e-15
        
        # Pure Lorentzian profiles (gamma_frac ≈ 1)
        is_lorentz = gamma_fracs > (1.0 - 1e-8)
        if np.any(is_lorentz):
            gamma[is_lorentz] = widths[is_lorentz] / 2.0
            sigma[is_lorentz] = 1e-15
        
        # Mixed profiles using proper Olivero-Longbothum decomposition
        is_mixed = ~(is_gaussian | is_lorentz)
        if np.any(is_mixed):
            V = widths[is_mixed]
            f = gamma_fracs[is_mixed]
            
            # Correct decomposition: L = f*V, solve for G
            L = f * V
            term = V - 0.5346 * L
            G_sq = np.maximum(0.0, term**2 - 0.2166 * L**2)
            G = np.sqrt(G_sq)
            
            # Convert to distribution parameters
            sigma[is_mixed] = G / (2.0 * sqrt_2ln2)
            gamma[is_mixed] = L / 2.0
        
        # Final numerical safeguards
        sigma = np.maximum(sigma, 1e-15)
        gamma = np.maximum(gamma, 1e-15)
        
        # True vectorized computation
        result = true_vectorized_voigt(x, centers, sigma, gamma, areas)
        
        return result
    
    def voigt_profile(x: np.ndarray, center: float, voigt_fwhm: float, area: float, 
                    lorentz_frac: float, min_width: float = 1e-10) -> np.ndarray:
        """
        Area-normalized Voigt profile with exact physical parameterization.
        This is kept for compatibility with curve_fit, but batch_voigt_profiles should be used for reconstruction.
        """
        return batch_voigt_profiles(
            x.reshape(1, -1),
            np.array([center]),
            np.array([voigt_fwhm]),
            np.array([area]),
            np.array([lorentz_frac]),
            min_width
        )[0]
    
    def multi_voigt_free_gamma(x: np.ndarray, *flat_params: float) -> np.ndarray:
        """
        Sum of N Voigt peaks. flat_params should be multiple of 4: (center, width, area, gamma_ratio) per peak.
        """
        x = np.asarray(x)
        y_model = np.zeros_like(x, dtype=float)
        n_params = len(flat_params)
        if n_params % 4 != 0:
            raise ValueError("flat_params length must be a multiple of 4")
        for i in range(0, n_params, 4):
            c, w, A, r = flat_params[i:i + 4]
            y_model += voigt_profile(x, c, w, A, r)
        return y_model
    
    p0, bounds = build_initial_guesses_and_bounds(
        x_fit, y_fit,
        centers, widths, amp_maxs,
        gamma_ratios, center_windows,
        min_widths, max_widths,
        config
    )

    # -------------------------
    # Perform fit
    # -------------------------
    popt, _ = curve_fit(
        multi_voigt_free_gamma,
        x_fit,
        y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=config["fitting"]["maxfev"],
        ftol=config["fitting"]["tolerances"]["ftol"],
        xtol=config["fitting"]["tolerances"]["xtol"],
        gtol=config["fitting"]["tolerances"]["gtol"],
    )

    y_voigt = multi_voigt_free_gamma(x_fit, *popt)

    # -------------------------
    # Plot
    # -------------------------
    n_peaks = len(popt) // 4
    components = []

    for i in range(n_peaks):
        c, w, A, r = popt[i*4:i*4+4]
        comp = voigt_profile(x_fit, c, w, A, r)
        components.append(comp)
    
    fig = plt.figure(figsize=(12.5, 6.25))

    ax1 = fig.add_subplot()

    # --- top panel ---
    line_diff, = ax1.plot(wave_number, diff_mean, color="C2", lw=1.4, label="Predicted difference")
    ax1.fill_between(
        wave_number,
        diff_mean - diff_std,
        diff_mean + diff_std,
        color="C2",
        alpha=0.35,
        linewidth=0
    )

    line_envelope, = ax1.plot(x_fit, y_voigt, color="red", lw=2.2, label="Voigt fit")
    # Individual components
    colors = plt.cm.tab10.colors
    for i, comp in enumerate(components):
        if '^' in peak_names[i]:
            base, superscript = peak_names[i].split('^')
            latex_name = f"${base}^{{{superscript}}}$"
        else:
            latex_name = f"${peak_names[i]}$"
        ax1.plot(x_fit, comp, lw=1.6, ls='--', color=colors[i % 10], label=latex_name)
        ax1.annotate(
            latex_name,
            xy=(x_fit[np.argmax(comp)], np.max(comp)),
            xytext=(50, 50),  # offset so label doesn't overlap
            textcoords='offset points',
            fontsize='small',
            color='black',
            ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.75)
        )

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel("Intensity, a.u.")
    ax1.legend(frameon=True, fontsize="small", handles=[line_diff, line_envelope])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    plt.tight_layout()

    return fig


# -----------------------------
# Callbacks
# -----------------------------

def load_dataset(b):
    global ds_to_plot, model_tag, material_plots_folder

    with out:
        clear_output(wait=True)

        model_tag = model_dropdown.value
        rel_path = dataset_dropdown.value
        nc_file = unmixed_dir / model_tag / rel_path

        ds_to_plot = xr.load_dataset(nc_file)

        material_plots_folder = maindir / "temp" / "unmixing_pictures"
        material_plots_folder.mkdir(parents=True, exist_ok=True)

        fig = plot_spatial_dataset_mean(ds_to_plot)
        plt.show()
        fig2 = plot_mean_spectrum_with_voigt_fit(ds_to_plot)
        plt.show()

        print(f"Loaded and plotted spatial dataset: {model_tag}/{rel_path}")


def save_plot(b):
    if 'ds_to_plot' not in globals():
        return

    fig = plot_spatial_dataset_mean(ds_to_plot)

    fname = material_plots_folder / f"{model_tag}_{dataset_dropdown.value.stem}_spatial_mean"
    fig.savefig(fname.with_suffix(".svg"), transparent=True, bbox_inches='tight')
    fig.savefig(fname.with_suffix(".pdf"), transparent=True, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {fname.name}")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.resolve()
    ds_filepath = SCRIPT_DIR / 'data/unmixed_by_min70_highf_RUN3REC2_Buffer_20241011_1_11x11_pt1.nc'
    ds = xr.load_dataset(ds_filepath)
    fig1 = plot_spatial_dataset_mean(ds)
    plt.show()
    fig2 = plot_mean_spectrum_with_voigt_fit(ds)
    plt.show()
