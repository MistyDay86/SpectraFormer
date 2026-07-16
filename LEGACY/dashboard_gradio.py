"""
SpectraFormer Dashboard (Gradio) - Interactive visualization for spectral unmixing results.

Run with: python dashboard_gradio.py
Or: gradio dashboard_gradio.py

UPDATED: Handles both old (dict-wrapped) and new (direct) checkpoint formats.
"""

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

# Suppress absl "read only" warnings from orbax checkpoint
logging.getLogger("absl").setLevel(logging.ERROR)


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


def load_model_and_predict(checkpoint_tag: str, dataset_path: Path, ckpts_path: Path):
    """Load model from checkpoint and run predictions on dataset."""
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    import xarray as xr

    from spectraformer.inference import predict
    from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
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

    # Initialize checkpoint manager with read-only options
    ckpt_options = ocp.CheckpointManagerOptions(
        read_only=True, save_interval_steps=0, create=False
    )
    ckpt_manager = ocp.CheckpointManager(checkpoint_path, options=ckpt_options)

    # Get config from checkpoint metadata
    configs_dict = ckpt_manager.metadata()
    if configs_dict is None:
        raise ValueError(
            "Checkpoint does not contain configuration metadata. "
            "This checkpoint may have been created with an older version."
        )
    if "custom" in configs_dict:
        configs_dict = configs_dict["custom"]

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

    # Get mask windows from config
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )

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

    # Restore checkpoint - try both new and old formats
    latest_step = ckpt_manager.latest_step()
    
    # First, try new format (direct state restoration)
    try:
        state = ckpt_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state),
        )
        print(f"Successfully loaded checkpoint using NEW format (direct state)")
    except Exception as e_new:
        # If new format fails, try old format (dict-wrapped)
        try:
            restored_dict = ckpt_manager.restore(
                latest_step,
                args=ocp.args.StandardRestore({"state": state})
            )
            state = restored_dict["state"]
            print(f"Successfully loaded checkpoint using OLD format (dict-wrapped)")
        except Exception as e_old:
            # If both fail, provide detailed error message
            raise ValueError(
                f"Failed to restore checkpoint in both formats.\n"
                f"New format error: {str(e_new)}\n"
                f"Old format error: {str(e_old)}"
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
    wave_number = np.asarray(prediction["wave_number"])
    # Un-normalize wave_number if needed
    if np.max(np.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000

    fig = go.Figure()

    # Add traces for each data series
    colors = {
        "spectra": "#1f77b4",
        "predicted_spectra": "#ff7f0e",
        "predicted_difference": "#2ca02c",
    }
    names = {
        "spectra": "Spectra",
        "predicted_spectra": "Predicted spectra",
        "predicted_difference": "Predicted difference",
    }

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
        yaxis_title="Intensity (normalized)",
        hovermode="x unified",
        showlegend=True,
        width=1000,
        height=500,
    )

    return fig


def create_dashboard(
    ckpts_path: Path = Path("./saved_models/checkpoints"),
    data_path: Path = Path("./data"),
):
    """Create the Gradio dashboard interface."""

    # Get available models and datasets
    available_models = get_available_checkpoints(ckpts_path)
    available_datasets = get_available_datasets(data_path)

    # Cache for loaded models (avoid reloading on every slider change)
    model_cache = {}

    def get_or_load_model(checkpoint_tag: str, dataset_path: Path, ckpts_path: Path):
        """Load model from cache or disk."""
        cache_key = (checkpoint_tag, str(dataset_path))
        if cache_key not in model_cache:
            model_cache[cache_key] = load_model_and_predict(
                checkpoint_tag, dataset_path, ckpts_path
            )
        return model_cache[cache_key]

    def run_inference(
        model_choice: str,
        dataset_source: str,
        local_dataset: str | None,
        uploaded_file,
        spectrum_idx: int,
    ):
        """Run inference and update the plot."""
        if not model_choice:
            return (
                None,
                "No model selected",
                "",
                "Please select a model checkpoint.",
            )

        # Determine dataset path
        if dataset_source == "Local file":
            if not local_dataset:
                return (
                    None,
                    "No dataset",
                    "",
                    "Please select a dataset.",
                )
            dataset_path = data_path / local_dataset
        else:
            if uploaded_file is None:
                return (
                    None,
                    "No dataset",
                    "",
                    "Please upload a NetCDF file.",
                )
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                tmp.write(open(uploaded_file.name, "rb").read())
                dataset_path = Path(tmp.name)

        try:
            state, predictions, num_params, configs, mask_windows = get_or_load_model(
                checkpoint_tag=model_choice,
                dataset_path=dataset_path,
                ckpts_path=ckpts_path,
            )
        except FileNotFoundError as e:
            return None, "Error", "", f"File not found: {e}"
        except ValueError as e:
            return None, "Error", "", f"Configuration error: {e}"
        except Exception as e:
            return None, "Error", "", f"Error loading model: {e}"

        # Clamp spectrum index
        spectrum_idx = max(0, min(spectrum_idx, len(predictions) - 1))

        # Format parameter count
        if num_params >= 1_000_000:
            params_str = f"{num_params / 1_000_000:.1f}M"
        elif num_params >= 1_000:
            params_str = f"{num_params / 1_000:.0f}K"
        else:
            params_str = str(num_params)

        # Create plot
        model_name = (
            configs.tag
            if hasattr(configs, "tag")
            else f"spectraformer:{model_choice}"
        )
        fig = create_spectrum_plot(predictions[spectrum_idx], model_name)

        # Build config details
        lr_decay = getattr(configs, "learning_rate_decay", "Constant")
        mask_windows_str = ", ".join(
            [f"[{start}, {end}]" for start, end in mask_windows]
        )
        config_text = f"""### Architecture
- Layers: {configs.num_layers}
- Heads: {configs.num_heads}
- Embedding dim: {configs.embedding_dim}

### Training
- Learning rate: {configs.learning_rate}
- LR schedule: {lr_decay}

### Mask Windows
{mask_windows_str}
"""

        # Model info
        model_info = f"**Model:** {model_name} | **Parameters:** {params_str} | **Spectrum:** {spectrum_idx + 1} of {len(predictions)}"

        return fig, model_info, config_text, ""

    def update_slider_max(
        model_choice: str,
        dataset_source: str,
        local_dataset: str | None,
        uploaded_file,
    ):
        """Update slider maximum based on loaded predictions."""
        if not model_choice:
            return gr.update(maximum=0, value=0)

        # Determine dataset path
        if dataset_source == "Local file":
            if not local_dataset:
                return gr.update(maximum=0, value=0)
            dataset_path = data_path / local_dataset
        else:
            if uploaded_file is None:
                return gr.update(maximum=0, value=0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                tmp.write(open(uploaded_file.name, "rb").read())
                dataset_path = Path(tmp.name)

        try:
            _, predictions, _, _, _ = get_or_load_model(
                checkpoint_tag=model_choice,
                dataset_path=dataset_path,
                ckpts_path=ckpts_path,
            )
            return gr.update(maximum=len(predictions) - 1, value=0)
        except Exception:
            return gr.update(maximum=0, value=0)

    # Build UI
    with gr.Blocks(
        title="SpectraFormer Dashboard",
    ) as demo:
        gr.Markdown(
            """
            # SpectraFormer Dashboard

            **SpectraFormer** is a transformer-based model for spectral unmixing of Raman spectra.

            Select a model and dataset to visualize predictions.
            """
        )

        with gr.Row():
            # Sidebar column
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Select checkpoint",
                    info="Choose a trained model checkpoint",
                )

                gr.Markdown("### Dataset Selection")
                dataset_source_radio = gr.Radio(
                    choices=["Local file", "Upload file"],
                    value="Local file",
                    label="Dataset source",
                )

                local_dataset_dropdown = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Select dataset",
                    info="Choose a NetCDF file from the data directory",
                    visible=True,
                )

                file_upload = gr.File(
                    label="Upload NetCDF file",
                    file_types=[".nc"],
                    visible=False,
                )

                run_btn = gr.Button("Run Inference", variant="primary")

            # Main content column
            with gr.Column(scale=3):
                error_box = gr.Markdown("", visible=True)
                model_info_md = gr.Markdown("Select a model and dataset to begin.")

                spectrum_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    label="Spectrum index",
                    info="Navigate through different spectra in the dataset",
                )

                plot_output = gr.Plot(label="Spectrum Visualization")

                with gr.Accordion("Configuration Details", open=False):
                    config_md = gr.Markdown("")

        # Toggle visibility based on dataset source
        def toggle_dataset_inputs(source):
            if source == "Local file":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        dataset_source_radio.change(
            fn=toggle_dataset_inputs,
            inputs=[dataset_source_radio],
            outputs=[local_dataset_dropdown, file_upload],
        )

        # Run inference on button click
        run_btn.click(
            fn=run_inference,
            inputs=[
                model_dropdown,
                dataset_source_radio,
                local_dataset_dropdown,
                file_upload,
                spectrum_slider,
            ],
            outputs=[plot_output, model_info_md, config_md, error_box],
        ).then(
            fn=update_slider_max,
            inputs=[
                model_dropdown,
                dataset_source_radio,
                local_dataset_dropdown,
                file_upload,
            ],
            outputs=[spectrum_slider],
        )

        # Update plot when slider changes
        spectrum_slider.change(
            fn=run_inference,
            inputs=[
                model_dropdown,
                dataset_source_radio,
                local_dataset_dropdown,
                file_upload,
                spectrum_slider,
            ],
            outputs=[plot_output, model_info_md, config_md, error_box],
        )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(
        theme=gr.themes.Soft(),
    )
