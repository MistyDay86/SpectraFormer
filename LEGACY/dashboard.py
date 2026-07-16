"""
SpectraFormer Dashboard - Interactive visualization for spectral unmixing results.

Run with: streamlit run dashboard.py
"""

import logging
from pathlib import Path

import streamlit as st

# Suppress absl "read only" warnings from orbax checkpoint
logging.getLogger("absl").setLevel(logging.ERROR)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="SpectraFormer Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling (theme-agnostic)
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Code block styling for model summary */
    div[data-testid="stCode"] pre code {
        font-size: 9px;
        line-height: 1.2;
    }
    div[data-testid="stCode"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_available_checkpoints(ckpts_path: Path) -> list[str]:
    """Scan checkpoint directory for available models."""
    available = []
    if ckpts_path.exists():
        for elem in ckpts_path.iterdir():
            if elem.is_dir():
                tagname = elem.name
                # Remove 'spectraformer:' prefix for display
                display_name = tagname.replace("spectraformer:", "")
                available.append(display_name)
    return sorted(available)


def get_available_datasets(data_path: Path) -> list[str]:
    """Scan data directory for available NetCDF files."""
    available = []
    if data_path.exists():
        # Find all .nc files recursively
        for nc_file in data_path.rglob("*.nc"):
            relative = nc_file.relative_to(data_path)
            available.append(str(relative))
    return sorted(available)


@st.cache_resource
def load_model_and_predict(checkpoint_tag: str, dataset_path: Path, ckpts_path: Path):
    """Load model from checkpoint and run predictions on dataset."""
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    import xarray as xr

    from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
    from spectraformer.model import CustomTrainState, SpectraFormer
    from spectraformer.inference import predict

    # Build full checkpoint path
    full_tag = f"spectraformer:{checkpoint_tag}" if not checkpoint_tag.startswith("spectraformer:") else checkpoint_tag
    checkpoint_path = ckpts_path / full_tag

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize checkpoint manager with read-only options
    ckpt_options = ocp.CheckpointManagerOptions(read_only=True, save_interval_steps=0, create=False)
    ckpt_manager = ocp.CheckpointManager(checkpoint_path, options=ckpt_options)

    # Get config from checkpoint metadata
    configs_dict = ckpt_manager.metadata()
    if configs_dict is None:
        raise ValueError(
            "Checkpoint does not contain configuration metadata. "
            "This checkpoint may have been created with an older version."
        )

    # Convert dict to namespace-like object for attribute access
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    configs = Config(configs_dict)

    # Load and preprocess dataset
    dataarray = xr.load_dataarray(dataset_path)
    if len(dataarray.dims) == 1:
        dataarray = dataarray.expand_dims("sample")
    dataset = preprocess_dataset(dataarray, option="whitaker_hayes_with_outliers")

    # Build learning rate schedule (needed for checkpoint restoration)
    cosine_kwargs = []
    init_value = 0.1 * configs.learning_rate
    peak_value = configs.learning_rate
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

    # Get mask windows from config
    mask_windows = list(zip(configs.masked_interval_starts, configs.masked_interval_ends))

    # Create dummy batch for model initialization
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

    # Restore checkpoint - direct state restoration (not wrapped in dict)
    state = ckpt_manager.restore(
        ckpt_manager.latest_step(),
        args=ocp.args.StandardRestore(state),
    )

    # Run predictions
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

    # Count model parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))

    return state, predictions, num_params, configs, mask_windows


def create_spectrum_plot(prediction: dict, model_name: str):
    """Create an interactive Plotly figure for spectrum visualization."""
    import numpy as np
    import plotly.graph_objects as go

    wave_number = np.asarray(prediction["wave_number"])
    # Un-normalize wave_number if needed
    if np.max(np.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000

    fig = go.Figure()

    # Add traces for each data series
    colors = {"spectra": "#1f77b4", "predicted_spectra": "#ff7f0e", "predicted_difference": "#2ca02c"}
    names = {"spectra": "Spectra", "predicted_spectra": "Predicted spectra", "predicted_difference": "Predicted difference"}

    for key in ["spectra", "predicted_spectra", "predicted_difference"]:
        data = np.asarray(prediction[key])
        fig.add_trace(
            go.Scatter(
                x=wave_number,
                y=data,
                mode="lines",
                name=names[key],
                line=dict(color=colors[key], width=1.5),
                hovertemplate=f"{names[key]}<br>Raman shift: %{{x:.1f}} cm⁻¹<br>Intensity: %{{y:.4f}}<extra></extra>",
            )
        )

    # Add vertical lines at mask boundaries
    mask = np.asarray(prediction["mask"])
    mask_boundaries = np.argwhere(np.diff(mask, prepend=np.array([True]))).flatten()
    for bdr in mask_boundaries:
        fig.add_vline(
            x=wave_number[bdr],
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )

    fig.update_layout(
        title=dict(text=model_name, font=dict(size=16)),
        xaxis_title="Raman shift (cm⁻¹)",
        yaxis_title="Intensity (a.u.)",
        yaxis_range=[-0.3, 1.5],
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
    )

    return fig


def main():
    """Main dashboard application."""
    # Setup paths
    maindir = Path(__file__).parent.resolve()
    ckpts_path = maindir / "checkpoints"
    data_path = maindir / "data"

    # --- Sidebar ---
    with st.sidebar:
        st.title("SpectraFormer Dashboard")
        st.markdown(
            """
            **SpectraFormer** is a transformer-based model
            for spectral unmixing of Raman spectra.

            Select a model and dataset to visualize
            predictions.
            """
        )
        st.markdown("---")

        # Model selection
        st.subheader("Model Selection")
        available_models = get_available_checkpoints(ckpts_path)

        if not available_models:
            st.warning("No checkpoints found in `checkpoints/`")
            current_model_tag = None
        else:
            current_model_tag = st.selectbox(
                "Select checkpoint:",
                available_models,
                index=0,
                help="Choose a trained model checkpoint",
            )

        st.markdown("---")

        # Dataset selection
        st.subheader("Dataset Selection")

        # Option 1: Select from available datasets
        available_datasets = get_available_datasets(data_path)

        dataset_source = st.radio(
            "Dataset source:",
            ["Local file", "Upload file"],
            horizontal=True,
        )

        dataset_path = None
        if dataset_source == "Local file":
            if not available_datasets:
                st.warning("No .nc files found in `data/` directory")
            else:
                selected_dataset = st.selectbox(
                    "Select dataset:",
                    available_datasets,
                    help="Choose a NetCDF file from the data directory",
                )
                if selected_dataset:
                    dataset_path = data_path / selected_dataset
        else:
            uploaded_file = st.file_uploader(
                "Upload NetCDF file:",
                type=["nc"],
                help="Upload a .nc file containing spectral data",
            )
            if uploaded_file:
                # Save uploaded file temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    dataset_path = Path(tmp.name)

        st.markdown("---")

    # --- Main content ---

    if current_model_tag is None:
        st.error("No model checkpoints available. Please add checkpoints to `checkpoints/`.")
        return

    if dataset_path is None:
        st.info("Please select or upload a dataset to begin.")
        return

    # Load model and run predictions
    try:
        state, predictions, num_params, configs, mask_windows = load_model_and_predict(
            checkpoint_tag=current_model_tag,
            dataset_path=dataset_path,
            ckpts_path=ckpts_path,
        )
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e)
        return

    # Model info
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    with col1:
        model_name = configs.tag if hasattr(configs, "tag") else f"spectraformer:{current_model_tag}"
        st.metric("Model", model_name)
    with col2:
        # Format parameter count nicely (e.g., 1.2M, 340K)
        if num_params >= 1_000_000:
            params_str = f"{num_params / 1_000_000:.1f}M"
        elif num_params >= 1_000:
            params_str = f"{num_params / 1_000:.0f}K"
        else:
            params_str = str(num_params)
        st.metric("Parameters", params_str)

    st.markdown("---")

    # Predictions explorer
    st.subheader("Predictions Explorer")

    col_slider, col_info = st.columns([3, 1])
    with col_slider:
        data_idx = st.slider(
            "Select spectrum index:",
            min_value=0,
            max_value=len(predictions) - 1,
            value=0,
            help="Navigate through different spectra in the dataset",
        )
    with col_info:
        st.info(f"Showing spectrum {data_idx + 1} of {len(predictions)}")

    # Plot
    model_name = configs.tag if hasattr(configs, "tag") else f"spectraformer:{current_model_tag}"
    fig = create_spectrum_plot(predictions[data_idx], model_name)
    st.plotly_chart(fig, width='stretch')

    # Configuration details
    with st.expander("Configuration Details", expanded=False):
        config_cols = st.columns(3)
        with config_cols[0]:
            st.markdown("**Architecture**")
            st.write(f"- Layers: {configs.num_layers}")
            st.write(f"- Heads: {configs.num_heads}")
            st.write(f"- Embedding dim: {configs.embedding_dim}")
        with config_cols[1]:
            st.markdown("**Training**")
            st.write(f"- Learning rate: {configs.learning_rate}")
            lr_decay = getattr(configs, "learning_rate_decay", "Constant")
            st.write(f"- LR schedule: {lr_decay}")
        with config_cols[2]:
            st.markdown("**Mask Windows**")
            for i, (start, end) in enumerate(mask_windows):
                st.write(f"- Window {i + 1}: [{start}, {end}]")


if __name__ == "__main__":
    main()
