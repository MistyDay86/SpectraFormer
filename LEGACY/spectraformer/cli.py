"""
SpectraFormer CLI for spectral unmixing inference.

Usage (single file):
    spectraformer-unmix --checkpoint path/to/checkpoint --input data.nc

Usage (directory):
    spectraformer-unmix --checkpoint path/to/checkpoint --input data_dir/

Options:
    --output, -o             Output path (default: unmixed_<input> for files, unmixed/ for directories)
    --device auto|cpu|gpu    Device to run on (default: auto - uses GPU if available)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Annotated

from loguru import logger
import tyro

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


@dataclass
class UnmixArgs:
    """Arguments for spectral unmixing inference."""

    checkpoint: Path
    """Path to the checkpoint directory (e.g., checkpoints/spectraformer:min70_highf)"""

    input: Path
    """Path to input NetCDF file (.nc) or directory containing .nc files"""

    output: Annotated[Path | None, tyro.conf.arg(aliases=["-o"])] = None
    """Path for output file (.nc) or directory. Defaults to 'unmixed_<input>' for files or 'unmixed/' for directories."""

    device: Literal["auto", "cpu", "gpu"] = "auto"
    """Device to run inference on (auto uses GPU if available, falls back to CPU)"""


def _detect_device(requested: str) -> str:
    """Detect available device and return the platform to use."""
    import os

    if requested == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        return "cpu"

    if requested == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        return "gpu"

    # Auto-detect: try GPU first, fall back to CPU
    try:
        # Temporarily allow all platforms to check availability
        import jax

        devices = jax.devices()
        has_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices)
        if has_gpu:
            return "gpu"
        return "cpu"
    except Exception:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return "cpu"


def run_unmixing(args: UnmixArgs) -> None:
    """Run spectral unmixing on input data."""
    # Detect device before heavy imports
    device = _detect_device(args.device)
    logger.info(f"Using device: {device}")

    # Now import JAX and related modules
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    import xarray as xr

    from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
    from spectraformer.model import CustomTrainState, SpectraFormer
    from spectraformer.inference import predict

    logger.debug(f"JAX devices: {jax.devices()}")

    # Convert paths to absolute
    args.checkpoint = args.checkpoint.resolve()
    args.input = args.input.resolve()

    # Compute default output path if not provided
    if args.output is None:
        if args.input.is_dir():
            args.output = args.input.parent / "unmixed"
        else:
            args.output = args.input.parent / f"unmixed_{args.input.name}"
    args.output = args.output.resolve()

    # Validate inputs
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint directory not found: {args.checkpoint}")
        sys.exit(1)

    if not args.input.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    # Determine if input is file or directory
    if args.input.is_dir():
        input_files = list(args.input.rglob("*.nc"))
        if not input_files:
            logger.error(f"No .nc files found in {args.input}")
            sys.exit(1)
        is_batch = True
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Found {len(input_files)} .nc files to process")
    else:
        if args.input.suffix.lower() != ".nc":
            logger.error(f"Input file must be a NetCDF file (.nc), got: {args.input.suffix}")
            sys.exit(1)
        input_files = [args.input]
        is_batch = False
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint with metadata (config)
    logger.info(f"Loading checkpoint: {args.checkpoint.name}")

    ckpt_options = ocp.CheckpointManagerOptions(read_only=True, save_interval_steps=0, create=False)
    ckpt_manager = ocp.CheckpointManager(
        args.checkpoint,
        options=ckpt_options,
    )

    # Get config from checkpoint metadata
    configs = ckpt_manager.metadata()
    if configs is None:
        logger.error("Checkpoint does not contain configuration metadata")
        logger.error("This checkpoint may have been created with an older version")
        sys.exit(1)

    # Convert dict to namespace-like object for attribute access
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    configs = Config(configs)

    logger.info(f"Model: {configs.tag} (layers={configs.num_layers}, heads={configs.num_heads}, dim={configs.embedding_dim})")

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

    # Load first file to initialize model
    first_file = input_files[0]
    logger.debug(f"Loading {first_file.name} for model initialization")
    dataarray = xr.load_dataarray(first_file)
    if len(dataarray.dims) == 1:
        dataarray = dataarray.expand_dims("sample")
    dataset = preprocess_dataset(dataarray, option="whitaker_hayes_with_outliers")

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

    # Restore checkpoint
    state = ckpt_manager.restore(
        ckpt_manager.latest_step(),
        args=ocp.args.StandardRestore(state),
    )
    logger.info(f"Restored checkpoint at step {state.step}")

    # Process each input file
    for file_idx, input_file in enumerate(input_files):
        logger.info(f"Processing [{file_idx + 1}/{len(input_files)}]: {input_file.name}")

        # Load and preprocess
        dataarray = xr.load_dataarray(input_file)
        if len(dataarray.dims) == 1:
            dataarray = dataarray.expand_dims("sample")
        dataset = preprocess_dataset(dataarray, option="whitaker_hayes_with_outliers")

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

        # Convert predictions to arrays
        N = len(predictions)
        M = len(predictions[0]["wave_number"])

        arr_spectra = np.zeros((N, M), dtype=np.float32)
        arr_masked_spectra = np.zeros((N, M), dtype=np.float32)
        arr_mask = np.zeros((N, M), dtype=bool)
        arr_predicted_spectra = np.zeros((N, M), dtype=np.float32)
        arr_predicted_difference = np.zeros((N, M), dtype=np.float32)

        for i, d in enumerate(predictions):
            arr_spectra[i, :] = np.asarray(jax.device_get(d["spectra"]))
            arr_masked_spectra[i, :] = np.asarray(jax.device_get(d["masked_spectra"]))
            arr_mask[i, :] = np.asarray(jax.device_get(d["mask"]))
            arr_predicted_spectra[i, :] = np.asarray(jax.device_get(d["predicted_spectra"]))
            arr_predicted_difference[i, :] = np.asarray(jax.device_get(d["predicted_difference"]))

        # Un-normalize wave_number
        arr_wave_number = np.asarray(jax.device_get(predictions[0]["wave_number"])) * 800 + 2000

        # Build output dataset preserving spatial structure
        spatial_dims = [dim for dim in dataarray.dims if dim != "wave_number"]
        coords_dict = {dim: dataarray.coords[dim].values for dim in spatial_dims}
        coords_dict["wave_number"] = arr_wave_number

        spatial_shape = [len(coords_dict[dim]) for dim in spatial_dims]

        if arr_spectra.shape[0] == np.prod(spatial_shape):
            # Reshape to preserve spatial dimensions
            arr_spectra = arr_spectra.reshape(*spatial_shape, M)
            arr_masked_spectra = arr_masked_spectra.reshape(*spatial_shape, M)
            arr_mask = arr_mask.reshape(*spatial_shape, M)
            arr_predicted_spectra = arr_predicted_spectra.reshape(*spatial_shape, M)
            arr_predicted_difference = arr_predicted_difference.reshape(*spatial_shape, M)
            dims = spatial_dims + ["wave_number"]
        else:
            # Fallback to sample dimension
            coords_dict["sample"] = np.arange(N)
            dims = ["sample", "wave_number"]

        ds = xr.Dataset(
            {
                "spectra": (dims, arr_spectra),
                "masked_spectra": (dims, arr_masked_spectra),
                "mask": (dims, arr_mask),
                "predicted_spectra": (dims, arr_predicted_spectra),
                "predicted_difference": (dims, arr_predicted_difference),
            },
            coords=coords_dict,
        )

        # Determine output path
        if is_batch:
            # Preserve directory structure for batch processing
            relative_path = input_file.relative_to(args.input)
            output_file = args.output / relative_path.with_name(f"unmixed_{relative_path.name}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file = args.output

        ds.to_netcdf(output_file, engine="netcdf4")
        logger.debug(f"Saved: {output_file}")

    logger.info(f"Done. Processed {len(input_files)} file(s)")


def main() -> None:
    """Entry point for the CLI."""
    import tyro

    args = tyro.cli(UnmixArgs)
    run_unmixing(args)


if __name__ == "__main__":
    main()
