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
        raise ValueError("This release accepts high-frequency SiC spectra covering 1300–2800 cm⁻¹, within 800–3300 cm⁻¹.")
    visible = (wave >= 1800) & (wave <= 2500)
    if visible.sum() < 32:
        raise ValueError("At least 32 observed points are required between 1800 and 2500 cm⁻¹.")
    return wave, counts, labels


def preprocess(counts):
    from spectraformer.input_pipeline import whitaker_hayes_spectrum
    cleaned = np.stack([whitaker_hayes_spectrum(row, 3, 8) for row in counts])
    minimum = cleaned.min(axis=1, keepdims=True)
    scale = cleaned.max(axis=1, keepdims=True) - minimum
    if not np.isfinite(cleaned).all() or np.any(scale <= 0):
        raise ValueError("A spectrum is constant or became invalid during spike removal.")
    normalized = (cleaned - minimum) / scale
    keep = np.max(np.abs(normalized - np.median(normalized, axis=0)), axis=1) < 0.15
    if not keep.any():
        raise ValueError("All spectra were excluded by the model's map-level outlier filter.")
    return normalized[keep].astype(np.float32), keep


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


def analyze(wave, counts, labels):
    import jax
    import jax.numpy as jnp
    normalized, keep = preprocess(counts)
    visible = (wave >= 1800) & (wave <= 2500)
    masked = np.where(visible[None, :], normalized, -1.0).astype(np.float32)
    predict, lock = load_engine()
    means, shapes = [], []
    with lock:
        for spectrum in masked:
            mu, alpha = predict(jnp.asarray(spectrum[None, :, None]),
                                jnp.asarray(((wave - 2000) / 800).astype(np.float32)[:, None]),
                                jnp.asarray(visible))
            mu, alpha = jax.device_get((mu, alpha))
            means.append(np.asarray(mu)[0, :, 0])
            shapes.append(np.asarray(alpha)[0, :, 0])
    mu, alpha = np.stack(means), np.stack(shapes)
    if not np.isfinite(mu).all() or not np.isfinite(alpha).all():
        raise ValueError("The model produced non-finite predictions for this input.")
    return {"wave": wave, "spectra": normalized, "mu": mu, "alpha": alpha,
            "residual": normalized - mu, "visible": visible,
            "labels": [label for label, selected in zip(labels, keep) if selected],
            "excluded": [label for label, selected in zip(labels, keep) if not selected]}


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
                             result["residual"][i,j], result["alpha"][i,j], int(result["visible"][j])])
    dataset = xr.Dataset({name: (("sample", "wave_number"), result[key]) for name, key in
                          [("spectra", "spectra"), ("predicted_mu", "mu"),
                           ("predicted_alpha", "alpha"), ("predicted_difference", "residual")]},
                         coords={"sample": np.arange(len(result["labels"])), "wave_number": wave})
    dataset["coordinate"] = ("sample", result["labels"])
    dataset["observed_mask"] = ("wave_number", result["visible"].astype(np.int8))
    dataset.wave_number.attrs["units"] = "cm^-1"
    dataset.attrs.update(checkpoint="min70_highf_filelevel_gammanll_best/84366", tau=0.0,
                         intensity_units="normalized", mask="hidden below 1800 and above 2500 cm^-1",
                         excluded_coordinates=json.dumps(result["excluded"]),
                         preprocessing="Whitaker-Hayes kernel=3 threshold=8; per-spectrum min-max; map filter threshold=0.15")
    return output.getvalue().encode(), dataset.to_netcdf(engine="scipy")


def main():
    st.set_page_config(page_title="SpectraFormer", page_icon="🔬", layout="wide")
    st.title("SpectraFormer")
    st.write("Reconstruct the SiC Raman background and inspect the remaining signal.")
    st.caption("GammaNLL · file-level checkpoint 84366 · mean subtraction (τ = 0)")
    st.info("For high-frequency graphene/buffer-layer spectra on SiC. Results use normalized intensity; residuals include noise and reconstruction error. Support for other acquisition conditions has not been established.")
    upload = st.file_uploader("Upload the original Raman file", type=["txt", "csv", "tsv", "nc"])
    st.caption("Text: Raman shift and intensity in the last two columns; optional spatial coordinates before them. Decimal points required. Maximum 25 MB, 1000 spectra, 2048 spectral points.")
    if upload is None:
        return
    import hashlib
    content = upload.getvalue()
    signature = hashlib.sha256(content).hexdigest()
    if st.session_state.get("signature") != signature:
        for key in ["result", "downloads"]:
            st.session_state.pop(key, None)
        st.session_state.signature = signature
    try:
        wave, counts, labels = read_input(content, upload.name)
    except Exception as exc:
        st.error(str(exc))
        return
    st.write(f"{len(labels)} spectra · {len(wave)} points · {wave[0]:.1f}–{wave[-1]:.1f} cm⁻¹")
    with st.expander("Preview original spectrum"):
        raw = go.Figure(go.Scatter(x=wave, y=counts[0], name="Original intensity"))
        raw.update_layout(xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Original intensity")
        st.plotly_chart(raw, use_container_width=True)
    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Preprocessing and reconstructing the SiC background…"):
                result = analyze(wave, counts, labels)
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
    fig.add_vrect(x0=1800, x1=2500, fillcolor="#64748b", opacity=0.08, line_width=0,
                  annotation_text="Observed context")
    fig.update_layout(xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Normalized intensity", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded region: observed context. The SiC signal in the surrounding regions is reconstructed from this context. Negative residuals are preserved.")
    csv_bytes, nc_bytes = st.session_state.downloads
    left, right = st.columns(2)
    left.download_button("Download CSV", csv_bytes, "spectraformer_results.csv", "text/csv")
    right.download_button("Download NetCDF", nc_bytes, "spectraformer_results.nc", "application/x-netcdf")
    if result["excluded"]:
        with st.expander("Excluded spatial coordinates"):
            st.write(result["excluded"])


if __name__ == "__main__":
    main()
