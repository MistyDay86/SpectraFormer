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
    pred_mu, pred_alpha = apply_fn(
        variables,
        batch["masked_spectra"],
        batch["wave_number"],
        *apply_fn_args,
        training=False,
    )
    res = {k: np.squeeze(v) for k, v in batch.items()}
    res["predicted_spectra"] = np.squeeze(pred_mu)
    res["predicted_mu"] = np.squeeze(pred_mu)
    res["predicted_alpha"] = np.squeeze(pred_alpha)
    res["predicted_difference"] = res["spectra"] - res["predicted_spectra"]
    return res

def plot_results_train(apply_fn, variables, batch: Batch, raman_shift):
    pred_mu, pred_alpha = apply_fn(
        variables,
        batch["masked_spectra"],
        batch["wave_number"],
        batch["mask"],
        training=False,
    )
    res = {k: np.squeeze(v) for k, v in batch.items()}
    res["predicted_spectra"] = np.squeeze(pred_mu)
    res["predicted_mu"] = np.squeeze(pred_mu)
    res["predicted_alpha"] = np.squeeze(pred_alpha)
    res["predicted_difference"] = res["spectra"] - res["predicted_spectra"]
    res["raman_shift"] = raman_shift
    return res


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
