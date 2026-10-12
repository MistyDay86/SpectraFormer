import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import io
import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "pretrained_checkpoints/min70_highf_filelevel_gammanll_best/84366/default"
CONFIG_PATH = ROOT / "configs/configs_min70_highf_filelevel_gammanll.yaml"
MODE_STANDARD = "Standard context â€” 1800â€“2500 cmâ»Â¹"
MODE_EXTENDED = "Extended context â€” 1200â€“1300, 1660â€“2500 and 2800â€“2900 cmâ»Â¹"
OBSERVED_WINDOWS = {
    MODE_STANDARD: [(1800, 2500)],
    MODE_EXTENDED: [(1200, 1300), (1660, 2500), (2800, 2900)],
}
MAX_SPECTRA = 1000
MAX_POINTS = 2048


def read_input(content, name):
    if len(content) > 25 * 1024 * 1024:
        raise ValueError("Maximum upload size is 25 MB.")
    if name.lower().endswith(".nc"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.nc"
            path.write_bytes(content)
            data = xr.load_dataarray(path)
        if "wave_number" not in data.dims:
            raise ValueError("NetCDF input needs a wave_number dimension.")
        spatial = [d for d in data.dims if d != "wave_number"]
        if spatial:
            data = data.stack(sample=spatial).transpose("sample", "wave_number")
            labels = [str(v) for v in data.sample.values]
        else:
            data = data.expand_dims(sample=[0]).transpose("sample", "wave_number")
            labels = ["Spectrum 1"]
        wave = np.asarray(data.wave_number.values, dtype=np.float64)
        counts = np.asarray(data.values, dtype=np.float64)
    else:
        text = content.decode("utf-8-sig")
        rows = []
        started = False
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.replace(";", " ").replace(",", " ").split()
            try:
                row = [float(value) for value in fields]
                if len(row) < 2:
                    raise ValueError()
            except ValueError:
                if started:
                    raise ValueError("Non-numeric row inside the spectrum. Use decimal points.")
                continue
            rows.append(row)
            started = True
        if not rows or len({len(r) for r in rows}) != 1:
            raise ValueError("Expected numeric columns: [coordinates ...] Raman shift, intensity.")
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Input contains NaN or infinite values.")
        coords = values[:, :-2]
        if coords.shape[1]:
            keys, inverse = np.unique(coords, axis=0, return_inverse=True)
            labels = [", ".join(f"X{i}={v:g}" for i, v in enumerate(key)) for key in keys]
        else:
            inverse = np.zeros(len(values), dtype=int)
            labels = ["Spectrum 1"]
        wave = np.unique(values[:, -2])
        if len(labels) > MAX_SPECTRA or len(wave) > MAX_POINTS:
            raise ValueError(f"Limit: {MAX_SPECTRA} spectra and {MAX_POINTS} points per spectrum.")
        counts = np.full((len(labels), len(wave)), np.nan)
        for sample, shift, intensity in zip(inverse, values[:, -2], values[:, -1]):
            index = np.searchsorted(wave, shift)
            if not np.isnan(counts[sample, index]):
                raise ValueError("Duplicate Raman shift for the same spatial coordinate.")
            counts[sample, index] = intensity
    if counts.shape[0] > MAX_SPECTRA or len(wave) > MAX_POINTS:
        raise ValueError(f"Limit: {MAX_SPECTRA} spectra and {MAX_POINTS} points per spectrum.")
    if len(wave) < 32 or not np.isfinite(wave).all() or not np.all(np.diff(wave) > 0):
        raise ValueError("The Raman axis must contain at least 32 finite, distinct, increasing values.")
    if not np.isfinite(counts).all():
        raise ValueError("Incomplete spectra or non-finite intensities. All spectra need the same Raman grid.")
    if wave.min() > 1300 or wave.max() < 2800 or wave.min() < 800 or wave.max() > 3300:
        raise ValueError("This release accepts high-frequency SiC spectra covering 1300â€“2800 cmâ»Â¹, within 800â€“3300 cmâ»Â¹.")
    return wave, counts, labels



@st.cache_resource
def load_engine():
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from spectraformer.model import SpectraFormer
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT.relative_to(ROOT)}")
    handler = ocp.PyTreeCheckpointer()
    try:
        restored = handler.restore(str(CHECKPOINT))
    finally:
        handler.close()
    params = restored["params"]
    model = SpectraFormer(embedding_dim=64, num_heads=8, num_layers=2, dropout_rate=0.2)
    predict = jax.jit(lambda y, wave, mask: model.apply({"params": params}, y, wave, mask, training=False))
    return predict, threading.Lock()


def observed_mask(wave, mode):
    if mode not in OBSERVED_WINDOWS:
        raise ValueError("Unknown inference mode.")
    visible = np.zeros(len(wave), dtype=bool)
    for start, end in OBSERVED_WINDOWS[mode]:
        visible |= (wave >= start) & (wave <= end)
    if visible.sum() < 32:
        raise ValueError("At least 32 data points are required within the observed windows.")
    return visible


def shade_context(fig, wave, visible):
    edges = np.flatnonzero(np.diff(np.r_[False, visible, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        fig.add_vrect(x0=wave[start], x1=wave[end-1],
                      fillcolor="#64748b", opacity=0.16, line_width=0)


def analyze(wave, counts, labels, mode=MODE_STANDARD):
    import jax
    import jax.numpy as jnp
    import yaml
    from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
    config = yaml.safe_load(CONFIG_PATH.read_text())
    fixed_visible = observed_mask(wave, mode)
    windows = OBSERVED_WINDOWS[mode]
    seed = int(config.get("root_rng_seed", 0))
    data = xr.DataArray(counts, dims=("position", "wave_number"),
                        coords={"position": np.arange(len(counts)), "wave_number": wave})
    data = preprocess_dataset(data, is_filter=False, option="whitaker_hayes")
    kept = np.asarray([int(v[0]) for v in data.spectra.values])
    if not len(kept):
        raise ValueError("All spectra were excluded by the model's outlier filter.")
    order = np.arange(len(kept))
    np.random.RandomState(seed).shuffle(order)
    data = data.isel(spectra=order)
    kept = kept[order]
    predict, lock = load_engine()
    means, shapes, inputs, masks, masked_inputs = [], [], [], [], []
    mask_windows_used = []
    with lock:
        for batch in batch_sampler(
                data, [], batch_size=int(config["batch_size"]),
                shuffle=False, drop_last=False, default_mask_value=config.get("default_mask_value", -1)):
            batch["mask"] = jnp.asarray(fixed_visible)
            batch["masked_spectra"] = jnp.where(batch["mask"][None, :, None], batch["spectra"],
                                                config.get("default_mask_value", -1))
            visible = np.asarray(batch["mask"], dtype=bool)
            if not visible.any():
                raise ValueError("The selected mask leaves no observed points.")
            # Evaluate one spectrum at a time with fixed observed windows.
            # This avoids allocating 24 full attention matrices simultaneously.
            for i in range(batch["spectra"].shape[0]):
                mu, alpha = predict(batch["masked_spectra"][i:i+1], batch["wave_number"], batch["mask"])
                mu, alpha = jax.device_get((mu, alpha))
                means.append(np.asarray(mu)[0, :, 0])
                shapes.append(np.asarray(alpha)[0, :, 0])
                masks.append(visible)
                inputs.append(np.asarray(batch["spectra"])[i, :, 0])
                masked_inputs.append(np.asarray(batch["masked_spectra"])[i, :, 0])
                mask_windows_used.append([[float(a), float(b)] for a,b in windows])
    mu, alpha, normalized = np.stack(means), np.stack(shapes), np.stack(inputs)
    if not np.isfinite(mu).all() or not np.isfinite(alpha).all():
        raise ValueError("The model produced non-finite predictions for this input.")
    return {"wave": wave, "spectra": normalized, "mu": mu, "alpha": alpha,
            "residual": normalized - mu, "visible": np.stack(masks),
            "masked_spectra": np.stack(masked_inputs), "mode": mode,
            "mask_windows": mask_windows_used,
            "labels": [labels[i] for i in kept],
            "excluded": [label for i,label in enumerate(labels) if i not in set(kept)]}


def exports(result):
    wave = result["wave"]
    rows = ["sample,coordinate,raman_shift_cm-1,preprocessed,SiC_mean,residual,gamma_alpha,observed"]
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(rows[0].split(","))
    for i, label in enumerate(result["labels"]):
        for j, shift in enumerate(wave):
            writer.writerow([i, label, shift, result["spectra"][i,j], result["mu"][i,j],
                             result["residual"][i,j], result["alpha"][i,j], int(result["visible"][i,j])])
    dataset = xr.Dataset({name: (("sample", "wave_number"), result[key]) for name, key in
                          [("spectra", "spectra"), ("masked_spectra", "masked_spectra"), ("predicted_mu", "mu"),
                           ("predicted_alpha", "alpha"), ("predicted_difference", "residual")]},
                         coords={"sample": np.arange(len(result["labels"])), "wave_number": wave})
    dataset["coordinate"] = ("sample", result["labels"])
    dataset["observed_mask"] = (("sample", "wave_number"), result["visible"].astype(np.int8))
    dataset.wave_number.attrs["units"] = "cm^-1"
    dataset.attrs.update(checkpoint="min70_highf_filelevel_gammanll_best/84366", tau=0.0,
                         intensity_units="normalized", inference_mode=result["mode"],
                         mask_windows=json.dumps(result["mask_windows"]),
                         excluded_coordinates=json.dumps(result["excluded"]),
                         preprocessing="Whitaker-Hayes kernel=3 threshold=8; per-spectrum min-max; map filter threshold=0.15")
    return output.getvalue().encode(), dataset.to_netcdf(engine="scipy")


def main():
    st.set_page_config(page_title="SpectraFormer", page_icon="ðŸ”¬", layout="wide")
    st.title("SpectraFormer")
    st.write("SpectraFormer estimates the Raman contribution of a silicon carbide (SiC) substrate from the observed parts of a measured spectrum. Subtracting this estimate reveals the remaining signal, including possible graphene or buffer-layer features, noise, and reconstruction error.")
    st.markdown("Upload an unprocessed spectrum or Raman map for automatic preprocessing and inference. Input spectra must cover **1300â€“2800 cmâ»Â¹**, with all Raman shifts within **800â€“3300 cmâ»Â¹**. [Source code and model](https://github.com/MistyDay86/SpectraFormer/tree/gammanll-filelevel-pipeline).")
    with st.expander("Data preparation and preprocessing"):
        st.markdown("Text files: use Raman shift (cmâ»Â¹) and intensity as the last two numeric columns, optionally preceded by spatial coordinates. Use decimal points and a common spectral grid across a map. NetCDF inputs need a `wave_number` dimension. No manual normalization is required.")
        st.markdown("Preprocessing removes spikes using Whitakerâ€“Hayes (kernel 3, threshold 8), normalizes each spectrum to [0,1], and excludes map spectra whose maximum deviation from the pointwise median is â‰¥ 0.15. Raman shifts are transformed as (Î½âˆ’2000)/800. Output intensity is normalized.")
    st.caption("Residual = preprocessed spectrum âˆ’ estimated SiC contribution. All output intensities are normalized.")
    upload = st.file_uploader("Upload the original Raman file", type=["txt", "csv", "tsv", "nc"])
    st.caption("Text: Raman shift and intensity in the last two columns; optional spatial coordinates before them. Decimal points required. Maximum 25 MB, 1000 spectra, 2048 spectral points.")
    mode = st.radio("Inference mode", [MODE_STANDARD, MODE_EXTENDED])
    if mode == MODE_STANDARD:
        st.caption("Uses a fixed observed window at 1800â€“2500 cmâ»Â¹ for every spectrum.")
    else:
        st.caption("Uses fixed observed windows at 1200â€“1300, 1660â€“2500 and 2800â€“2900 cmâ»Â¹ with the same pretrained model.")
    if upload is None:
        return
    import hashlib
    content = upload.getvalue()
    signature = hashlib.sha256(content + mode.encode()).hexdigest()
    if st.session_state.get("signature") != signature:
        for key in ["result", "downloads"]:
            st.session_state.pop(key, None)
        st.session_state.signature = signature
    try:
        wave, counts, labels = read_input(content, upload.name)
    except Exception as exc:
        st.error(str(exc))
        return
    st.write(f"{len(labels)} spectra Â· {len(wave)} points Â· {wave[0]:.1f}â€“{wave[-1]:.1f} cmâ»Â¹")
    for start, end in OBSERVED_WINDOWS[mode]:
        if not np.any((wave >= start) & (wave <= end)):
            st.warning(f"No data are available in the {start}â€“{end} cmâ»Â¹ observed window. Missing data are not extrapolated.")
        elif wave.min() > start or wave.max() < end:
            st.warning(f"The {start}â€“{end} cmâ»Â¹ observed window is partially covered by the input. Only available data are used; missing data are not extrapolated.")
    with st.expander("Preview original spectrum"):
        raw = go.Figure(go.Scatter(x=wave, y=counts[0], name="Original intensity"))
        raw.update_layout(xaxis_title="Raman shift (cmâ»Â¹)", yaxis_title="Original intensity")
        st.plotly_chart(raw, use_container_width=True)
    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Preprocessing and reconstructing the SiC backgroundâ€¦"):
                result = analyze(wave, counts, labels, mode)
                downloads = exports(result)
            st.session_state.result = result
            st.session_state.downloads = downloads
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
    result = st.session_state.get("result")
    if result is None:
        return
    st.success(f"{len(result['labels'])} spectra analyzed; {len(result['excluded'])} excluded by the outlier filter.")
    options = list(range(len(result["labels"])))
    selection = st.selectbox("Spectrum", options, format_func=lambda i: result["labels"][i])
    summary = st.checkbox("Show median across retained spectra", disabled=len(options) == 1)
    fig = go.Figure()
    for key, name, color in [("spectra", "Preprocessed spectrum", "#64748b"),
                             ("mu", "Reconstructed SiC", "#e68a2e"),
                             ("residual", "Residual", "#078a77")]:
        y = np.median(result[key], axis=0) if summary else result[key][selection]
        fig.add_trace(go.Scatter(x=result["wave"], y=y, name=name, line={"color": color}))
    visible = result["visible"][selection]
    shade_context(fig, result["wave"], visible)
    fig.update_layout(xaxis_title="Raman shift (cmâ»Â¹)", yaxis_title="Normalized intensity", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded regions indicate the spectral data provided to the model and are identical for all spectra, including the median view. Negative residual values are retained.")
    st.subheader("Residual spectrum")
    residual = np.median(result["residual"], axis=0) if summary else result["residual"][selection]
    residual_fig = go.Figure(go.Scatter(x=result["wave"], y=residual, name="Residual", line={"color": "#078a77"}))
    residual_fig.add_hline(y=0, line_dash="dot", line_color="#64748b")
    shade_context(residual_fig, result["wave"], visible)
    residual_fig.update_layout(xaxis_title="Raman shift (cmâ»Â¹)", yaxis_title="Normalized residual", height=400)
    st.plotly_chart(residual_fig, use_container_width=True)
    csv_bytes, nc_bytes = st.session_state.downloads
    left, right = st.columns(2)
    left.download_button("Download CSV", csv_bytes, "spectraformer_results.csv", "text/csv")
    right.download_button("Download NetCDF", nc_bytes, "spectraformer_results.nc", "application/x-netcdf")
    if result["excluded"]:
        with st.expander("Excluded spatial coordinates"):
            st.write(result["excluded"])


if __name__ == "__main__":
    main()
