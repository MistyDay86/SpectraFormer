import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from spectraformer.input_pipeline import Batch


plt.rcParams['font.size'] = 24


def _restore_wave_number(wave_number):
    wave_number = np.asarray(wave_number)
    if np.max(np.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000
    return wave_number


def predict(apply_fn, variables, batch: Batch, *apply_fn_args):
    pred = apply_fn(
        variables,
        batch["masked_spectra"],
        batch["wave_number"],
        *apply_fn_args,
        training=False,
    )
    res = {k: np.squeeze(v) for k, v in batch.items()}
    res["predicted_spectra"] = np.squeeze(pred)
    res["predicted_difference"] = res["spectra"] - res["predicted_spectra"]
    return res

def plot_results_train(predictions, step, epoch, current_model_tag):
    fig = plt.figure(figsize=(12.5, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    wave_number = _restore_wave_number(predictions["wave_number"])

    # -------------------------
    # Mask spans (union over all spectra)
    # -------------------------
    mask = np.asarray(predictions["mask"])
    if mask.ndim > 1:
        mask_any = np.any(mask, axis=tuple(range(mask.ndim - 1)))
    else:
        mask_any = mask.astype(bool)

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
                float(wave_number[s]), float(wave_number[e]),
                color="gray", alpha=0.1, linewidth=0
            )

    # -------------------------
    # Top panel: Data and Model output
    # -------------------------
    spectra = np.asarray(predictions["spectra"])
    pred_spectra = np.asarray(predictions["predicted_spectra"])

    if spectra.ndim > 1:
        spectra = spectra.T
    if pred_spectra.ndim > 1:
        pred_spectra = pred_spectra.T

    ax1.plot(wave_number, spectra, color="C0", lw=1.2, label="Data")
    ax1.plot(wave_number, pred_spectra, color="C1", lw=1.2, label="Model output")

    ax1.legend(frameon=True, fontsize='small')
    ax1.set_ylabel("Intensity, a.u.")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    # -------------------------
    # Bottom panel: Spectral subtraction (difference)
    # -------------------------
    difference = np.asarray(predictions["predicted_difference"])
    if difference.ndim > 1:
        difference = difference.T

    ax2.plot(wave_number, difference, color="C2", lw=1.2, label="Spectral subtraction")
    ax2.axhline(0, color='k', alpha=0.4)
    ax2.legend(frameon=True, fontsize='small')
    ax2.set_xlabel("Raman shift, cm$^{-1}$")
    ax2.set_ylabel("Intensity, a.u.")
    ax2.set_ylim(-0.08, 0.32)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    fig.suptitle(f'{current_model_tag}\nStep {step} -- Epoch {epoch}', y=0.975)
    fig.align_ylabels([ax1, ax2])

    return fig, ax1

def plot_loss(dummy_wave_number, loss, step, epoch, current_model_tag, mask=None):
    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    dummy_wave_number = _restore_wave_number(dummy_wave_number)

    loss = np.asarray(loss)
    full_loss = loss
    if mask is not None:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.ndim > 1:
            mask_bool = np.any(mask_bool, axis=tuple(range(1, mask_bool.ndim)))
        hidden_mask = ~mask_bool
        visible_mask = mask_bool
        masked_loss = np.where(hidden_mask, loss, np.nan)
        arithmetic_mean = np.nanmean(masked_loss)
    else:
        hidden_mask = np.ones_like(loss, dtype=bool)
        visible_mask = np.ones_like(loss, dtype=bool)
        masked_loss = loss
        arithmetic_mean = np.mean(masked_loss)
    
    ax.fill_between(
        dummy_wave_number,
        1e-14,
        full_loss,
        where=visible_mask,
        color='C0',
        alpha=0.08,
        linewidth=0,
    )
    ax.plot(dummy_wave_number, full_loss, label='Loss', color='C0', lw=0.9, alpha=0.75, ls='--')
    ax.plot(dummy_wave_number, masked_loss, label='Masked-region loss', color='C0', lw=2.2)
    ax.axhline(float(arithmetic_mean), label="Masked mean", color="r", alpha=1, linestyle=":")
    
    ax.set_xlabel("Raman shift, cm$^{-1}$")
    ax.set_ylabel("Loss, a.u.")
    ax.set_title(f'Loss for {current_model_tag}\nStep {step} -- Epoch {epoch}')
    ax.legend(frameon=True, fontsize='small')
    ax.grid(visible=True, which='both', axis='both', alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    
    ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e+1)
    return fig, ax
