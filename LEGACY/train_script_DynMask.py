"""
SpectraFormer training script.

Usage:
    1:
    export CUDA_DIR=/opt/share/libs/intel/nvidia/cuda-12.8.0
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/lib64:/usr/lib
    export LD_PRELOAD=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib/libcusparse.so.12:$CUDA_DIR/lib64/libnvJitLink.so.12
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
    export XLA_FLAGS="--xla_gpu_compilation_cache_dir=/work/dpoteryayev/.xla_cache --xla_gpu_compilation_cache_capacity_bytes=2147483648"
    export XLA_FLAGS="$XLA_FLAGS --xla_gpu_autotune_level=1"

    python train_script_DynMask.py --model-tag min72_highf --material SiC-high-f --regime multi-gpu --no-stream-datasets --debug-logging --debug_log_every_batches 1


    2:
    export CUDA_DIR=/opt/share/libs/intel/nvidia/cuda-12.8.0
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/lib64:/usr/lib
    export LD_PRELOAD=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib/libcusparse.so.12:$CUDA_DIR/lib64/libnvJitLink.so.12
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_DIR --xla_gpu_autotune_level=1"
    export JAX_COMPILATION_CACHE_DIR=/work/dpoteryayev/.xla_cache
    python train_script_DynMask.py   --model-tag min79_highf   --material SiC-high-f   --regime multi-gpu   --no-stream-datasets   --debug-logging   --debug-log-every-batches 1




    python train_script_DynMask.py --model-tag min72_highf --material SiC-high-f
    python train_script_DynMask.py --model-tag min72_highf --material SiC-high-f --regime multi-gpu --no-stream-datasets
"""

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

@dataclass
class TrainArgs:
    """Training configuration arguments."""

    model_tag: str = "min70_highf"
    """Model tag - must match configs/configs_{model_tag}.yaml"""

    material: str = "SiC-high-f"
    """Material/dataset directory name under data/parsed_data_spatial/"""

    regime: Literal["single-gpu", "multi-gpu"] = "multi-gpu"
    """Training regime: single-gpu or multi-gpu (uses all available devices)"""

    debug_nans: bool = True
    """Enable JAX NaN debugging (slower but catches numerical issues)"""

    debug_logging: bool = False
    """Enable verbose debug logging"""

    debug_compile_logging: bool = False
    """Enable JAX compile logging (short debug runs only)"""

    debug_log_every_batches: int = 1
    """Log every N batches when debug logging is enabled"""

    stream_datasets: bool = False
    """Load datasets one at a time instead of preloading all"""


def main(args: TrainArgs) -> None:
    """Run training with the given arguments."""
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    if args.debug_compile_logging:
        os.environ.setdefault("JAX_LOG_COMPILES", "1")
    logger.remove()
    log_level = "DEBUG" if args.debug_logging else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.add(f"temp/{args.model_tag}_{timestamp}.log")
    if args.debug_compile_logging:
        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING, force=True)
        logging.getLogger("jax").setLevel(logging.WARNING)
        logging.getLogger("jaxlib").setLevel(logging.WARNING)
    import gpustat
    import jax
    import jax.numpy as jnp
    import ml_confs
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    from flax.training.train_state import TrainState
    from flax.training.early_stopping import EarlyStopping
    from tensorboardX import SummaryWriter

    from spectraformer.model import CustomTrainState, SpectraFormer
    from spectraformer.input_pipeline import batch_sampler, dataset_loader
    from spectraformer.train_DynMask import (
        train_epoch, validation_epoch,
        train_epoch_pmap, validation_epoch_pmap,
        log_gpu_usage,
        warmup_compile_single,
        warmup_compile_pmap,
        warmup_lower_compile_pmap,
        _apply_mask_to_batch,
        _build_dynamic_mask_windows_np,
    )
    from spectraformer.inference import plot_results_train, predict, plot_loss

    jax.config.update("jax_debug_nans", args.debug_nans)

    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"JAX devices: {devices} ({num_devices} total)")
    if args.debug_logging:
        logger.debug(
            "Env: JAX_COMPILATION_CACHE_DIR={} XLA_FLAGS={}",
            os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
            os.environ.get("XLA_FLAGS", ""),
        )
        if args.debug_compile_logging:
            logger.debug(
                "Env: JAX_LOG_COMPILES={}",
                os.environ.get("JAX_LOG_COMPILES", ""),
            )
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
        if cache_dir:
            logger.debug(
                "Cache dir exists={} writable={} path={}",
                os.path.isdir(cache_dir),
                os.access(cache_dir, os.W_OK),
                cache_dir,
            )

    @jax.pmap
    def update_epoch(state):
        return state.replace(epoch=state.epoch + 1)

    # Directories
    maindir = Path(__file__).parent.resolve()
    logdir = maindir / "logs"
    ckptdir = maindir / "checkpoints"
    logdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)

    datadir = maindir / "data"
    configsdir = maindir / "configs"
    configsdir.mkdir(parents=True, exist_ok=True)

    # Map CLI regime to internal naming
    training_regime = "All devices" if args.regime == "multi-gpu" else "One device"

    # Load config
    config_file_path = configsdir / f"configs_{args.model_tag}.yaml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    parsed_datadir = datadir / "parsed_data_spatial"
    material_dir = parsed_datadir / args.material
    nc_files = list(material_dir.rglob("*.nc"))
    if not nc_files:
        raise ValueError(f"No .nc files found in {material_dir}")

    configs = ml_confs.from_file(config_file_path)
    stream_datasets = args.stream_datasets
    configs.tabulate()
    is_early_stop = True if not hasattr(configs, 'is_early_stop') else configs.is_early_stop # turning on early stopping process
    min_delta = 1e-4 if not hasattr(configs, 'early_stop_min_delta') else configs.early_stop_min_delta
    patience = 5 if not hasattr(configs, 'early_stop_patience') else configs.early_stop_patience
    is_masked_loss = False if not hasattr(configs, 'is_masked_loss') else configs.is_masked_loss
    
    if training_regime=="All devices" and configs.batch_size % num_devices !=0:
        raise Exception(f"Sharding requires batch size divisibility by the number of devices. Change it accordingly (preferably to 24).")
    
    # This is an implementation of learning rate schedule - multiple cosine decay cycles from init_value to init_value*alpha, then repeating from init_value.  
    cosine_kwargs = []
    
    init_value = 0.1*configs.learning_rate if not hasattr(configs, 'warmup_coeff') else configs.warmup_coeff*configs.learning_rate
    peak_value = configs.learning_rate
    warmup_steps = 1000 if not hasattr(configs, 'warmup_steps') else configs.warmup_steps
    decay_steps = 2000 if not hasattr(configs, 'decay_steps') else configs.decay_steps
    decline_coeff = 1 if not hasattr(configs, 'decline_coeff') else configs.decline_coeff
    
    for i in range(20 if not hasattr(configs, 'num_cycles') else configs.num_cycles):
        end_value = decline_coeff * init_value
        # 20 cycles - arbitrary large number to ensure enough cycles
        cycle_dict = {
            "init_value": init_value, 
            "peak_value": peak_value, 
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,            
            "end_value": end_value
        }
        cosine_kwargs.append(cycle_dict)
        init_value = end_value
        peak_value *= decline_coeff
    
    #                           LR schedule graph
    #
    # - - - - - - - - - - - - - - - ___* ___________ - - - - - - - - - - - - - - - - - - - - - - - - > configs.learning_rate (without a schedule it is constant)
    #|                     _______*/   |            \*___                |                         ^
    #|            _______*/            |                 \*___           |                         |
    #|  _______*/                      |                      \*         |                         v
    #|*/                               |                        \________* - - - - - - - - - - - - - > 
    #|                                 |                                 |
    #|           warmup_steps          |                                 |
    #|<------------------------------->|                                 |--------------------------->
    #|       Linear warm-up from       |                                 |
    #|          init_value to          |                                 | Repeat the cycle 100 times
    #|            peak_value           |    decay_steps - warmup_steps   |
    #|                                 |<------------------------------->|
    #|                                 |        Cosine decay from        |
    #|                                 |          peak_value to          |
    #|                                 |            end_value            |
    #|                           decay_steps                             |
    #|<----------------------------------------------------------------->|
    
    learning_rate_decay = getattr(configs, 'learning_rate_decay', 'Constant')
    match learning_rate_decay:
        case "Multiple cosine decay cycles":
            learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)
            tx = optax.adam(learning_rate=learning_rate_fn)
        case "Constant":
            tx = optax.adam(learning_rate=configs.learning_rate)
        case _:
            raise Exception(f"You didn't specify a learning rate schedule!")
    
    # New automatic dataset loading (streamed per dataset)
    is_filter = getattr(configs, 'is_filter', False)
    dataset_specs = [(nc_file, False) for nc_file in nc_files]
    if is_filter:
        dataset_specs += [(nc_file, True) for nc_file in nc_files]
        logger.info("Filtering enabled: doubling data")

    logger.info(f"Found {len(dataset_specs)}/{len(nc_files)} dataset entries from {material_dir}")
    if args.debug_logging:
        logger.debug(
            "Dataset loading mode: stream_datasets={} entries={}",
            stream_datasets,
            len(dataset_specs),
        )

    def load_dataset_for_file(nc_file, use_filter: bool):
        relative_path = nc_file.relative_to(parsed_datadir)
        return dataset_loader(
            datadir=parsed_datadir,
            file_location_with_name=str(relative_path),
            shuffle_rng_seed=configs.root_rng_seed,
            is_filter=use_filter,
            option='whitaker_hayes'
        )
    
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    datasets = []
    dummy_example = None
    if stream_datasets:
        for nc_file, use_filter in dataset_specs:
            if args.debug_logging:
                load_start = time.perf_counter()
            train_ds, val_ds = load_dataset_for_file(nc_file, use_filter)
            if args.debug_logging:
                logger.debug(
                    "Loaded dataset {} (filter={}) in {:.3f}s",
                    nc_file.name,
                    use_filter,
                    time.perf_counter() - load_start,
                )
            if train_ds.sizes['spectra'] >= configs.batch_size:
                dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
                del train_ds, val_ds
                gc.collect()
                break
            del train_ds, val_ds
            gc.collect()
    else:
        if args.debug_logging:
            preload_start = time.perf_counter()
        for nc_file, use_filter in dataset_specs:
            if args.debug_logging:
                load_start = time.perf_counter()
            train_ds, val_ds = load_dataset_for_file(nc_file, use_filter)
            if args.debug_logging:
                logger.debug(
                    "Loaded dataset {} (filter={}) in {:.3f}s",
                    nc_file.name,
                    use_filter,
                    time.perf_counter() - load_start,
                )
            if train_ds.sizes['spectra'] >= configs.batch_size and val_ds.sizes['spectra'] >= configs.batch_size:
                datasets.append((train_ds, val_ds, nc_file.name, use_filter))
                if dummy_example is None:
                    dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
            else:
                del train_ds, val_ds
        logger.info(f"Preloaded {len(datasets)} datasets into memory")
        if args.debug_logging:
            logger.debug(
                "Preload completed in {:.3f}s",
                time.perf_counter() - preload_start,
            )

    if dummy_example is None:
        raise ValueError("No dataset has enough spectra for the current batch size.")
    dummy_wave_number = jnp.squeeze(dummy_example["wave_number"])
    
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
        dropout_rate=configs.dropout_rate
    )
    
    # RNG Keys
    root_key = jax.random.PRNGKey(seed=configs.root_rng_seed)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    window_RNG_key = jax.random.split(main_key, num=1)[0]
    
    # Model Initialization
    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=True,
    )
    
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )
    
    # Checkpointing: load from checkpoint and resume training if available
    ckpt_options = ocp.CheckpointManagerOptions(
        #----------------------------------------------------------------------------------------------------#
        max_to_keep=patience+1, # this is for having best model in case of training process only worsens the loss
        enable_async_checkpointing=False,
        #----------------------------------------------------------------------------------------------------#
        )
    
    if not (ckptdir / configs.tag).exists():
        (ckptdir / configs.tag).mkdir()
        (ckptdir / configs.tag / ".tmp").touch()
    
    ckpt_manager = ocp.CheckpointManager(
        ckptdir / configs.tag,
        options=ckpt_options,
        metadata=configs.to_dict(),
    )
    
    # After initialization remove the dummy file
    if (ckptdir / configs.tag / ".tmp").exists():
        (ckptdir / configs.tag / ".tmp").unlink()
    
    if len(ckpt_manager.all_steps()) > 0:
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(),
            args=ocp.args.StandardRestore(state)
        )
        logger.info(f"Resuming from epoch {state.epoch}, step {state.step}")
    else:
        logger.info(f"No checkpoint found for '{configs.tag}', training from scratch")

    restored_epoch = state.epoch
    
    metric_writer = SummaryWriter(logdir / configs.tag)
    rng_streams = {"dropout": dropout_key}
    mean_streams = {"mean": "Not specified" if not hasattr(configs, 'mean') else configs.mean}
    
    # Early stopping initialization
    early_stop = EarlyStopping(min_delta=min_delta, patience=patience)
    
    train_metrics = []
    val_metrics = []
    
    # This is for drawing on TensorBoard both train and validation losses on a single graph
    layout = {
        "my_layout": {
            "loss_step": ["Multiline", ["train/train_loss_step", "val/val_loss_step"]],
            "loss_epoch": ["Multiline", ["train/train_loss_epoch", "val/val_loss_epoch"]],
            },
        }
    metric_writer.add_custom_scalars(layout)
    
    # metric_writer.add_graph(model=model)
    
    # Replicate state across all devices for pmap
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=('i',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    state = jax.tree.map(lambda x: jnp.stack([x] * num_devices), state)
    state = jax.device_put(state, sharding)

    def _build_warmup_batch(example, batch_size: int):
        spectra = jnp.repeat(example["spectra"], repeats=batch_size, axis=0)
        masked_spectra = jnp.repeat(example["masked_spectra"], repeats=batch_size, axis=0)
        return {
            "spectra": spectra,
            "masked_spectra": masked_spectra,
            "wave_number": example["wave_number"],
            "mask": example["mask"],
        }

    def _apply_dynamic_mask_for_train(batch, wave_number_raw, rng_key, batch_idx: int):
        if not getattr(configs, "dynamic_mask", False):
            return batch
        mask_seed = int(jax.random.fold_in(rng_key, batch_idx)[0])
        mask_rng = np.random.default_rng(mask_seed)
        mask_windows = _build_dynamic_mask_windows_np(
            wave_number_raw,
            mask_rng,
            chunk_size=getattr(configs, "mask_chunk_size", 200),
            min_chunks=getattr(configs, "mask_chunk_min", 2),
            max_chunks=getattr(configs, "mask_chunk_max", 6),
        )
        return _apply_mask_to_batch(
            batch,
            wave_number_raw,
            mask_windows,
            default_mask_value=getattr(configs, "default_mask_value", -1),
        )

    def _apply_dynamic_mask_for_val(batch, wave_number_raw, epoch_idx: int):
        if not getattr(configs, "dynamic_mask", False):
            return batch
        base_key = jax.random.PRNGKey(getattr(configs, "root_rng_seed", 0))
        mask_seed = int(jax.random.fold_in(base_key, epoch_idx)[0])
        mask_rng = np.random.default_rng(mask_seed)
        mask_windows = _build_dynamic_mask_windows_np(
            wave_number_raw,
            mask_rng,
            chunk_size=getattr(configs, "mask_chunk_size", 200),
            min_chunks=getattr(configs, "mask_chunk_min", 2),
            max_chunks=getattr(configs, "mask_chunk_max", 6),
        )
        return _apply_mask_to_batch(
            batch,
            wave_number_raw,
            mask_windows,
            default_mask_value=getattr(configs, "default_mask_value", -1),
        )

    warmup_epoch = int(restored_epoch) + 1
    mask_windows_for_loader = [] if getattr(configs, "dynamic_mask", False) else list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    if stream_datasets:
        warmup_nc_file, warmup_filter = dataset_specs[0]
        warmup_train_ds, warmup_val_ds = load_dataset_for_file(
            warmup_nc_file,
            warmup_filter,
        )
        warmup_dataset_name = warmup_nc_file.name
    else:
        warmup_train_ds, warmup_val_ds, warmup_dataset_name, _ = datasets[0]

    warmup_train_batch = next(
        batch_sampler(
            warmup_train_ds,
            mask_windows_for_loader,
            batch_size=configs.batch_size,
            rng_seed=warmup_epoch,
            shuffle=True,
        )
    )
    warmup_val_batch = next(
        batch_sampler(
            warmup_val_ds,
            mask_windows_for_loader,
            batch_size=configs.batch_size,
            rng_seed=warmup_epoch,
            shuffle=True,
        )
    )

    warmup_train_batch = _apply_dynamic_mask_for_train(
        warmup_train_batch,
        jnp.asarray(warmup_train_ds["wave_number"].values),
        window_RNG_key,
        batch_idx=0,
    )
    warmup_val_batch = _apply_dynamic_mask_for_val(
        warmup_val_batch,
        jnp.asarray(warmup_val_ds["wave_number"].values),
        warmup_epoch,
    )

    if args.debug_logging:
        logger.debug(
            "Warmup batch shapes train_spectra={} val_spectra={} wave={} mask={}",
            getattr(warmup_train_batch["spectra"], "shape", "?"),
            getattr(warmup_val_batch["spectra"], "shape", "?"),
            getattr(warmup_train_batch["wave_number"], "shape", "?"),
            getattr(warmup_train_batch["mask"], "shape", "?"),
        )

    if stream_datasets:
        del warmup_train_ds, warmup_val_ds
        gc.collect()
    warmup_steps = 2
    if args.debug_logging:
        logger.debug(
            "Warmup compile start: regime={} batch_size={} steps={}",
            training_regime,
            configs.batch_size,
            warmup_steps,
        )
        logger.debug("Warmup compile dataset={}", warmup_dataset_name)
    if training_regime == "All devices":
        if args.debug_logging:
            logger.debug("Warmup lower+compile start")
        warmup_lower_compile_pmap(
            state,
            warmup_train_batch,
            rng_streams,
            mean_streams,
            num_devices,
            configs.loss_fn,
            is_masked_loss,
        )
        if args.debug_logging:
            logger.debug("Warmup lower+compile completed")
        warmup_compile_pmap(
            state,
            warmup_train_batch,
            rng_streams,
            mean_streams,
            num_devices,
            configs.loss_fn,
            steps=warmup_steps,
            run_train=True,
            run_val=False,
            is_masked_loss=is_masked_loss,
        )
        warmup_compile_pmap(
            state,
            warmup_val_batch,
            rng_streams,
            mean_streams,
            num_devices,
            configs.loss_fn,
            steps=1,
            run_train=False,
            run_val=True,
            is_masked_loss=is_masked_loss,
        )
    else:
        warmup_compile_single(
            state,
            warmup_train_batch,
            rng_streams,
            mean_streams,
            steps=warmup_steps,
            run_train=True,
            run_val=False,
            is_masked_loss=is_masked_loss,
        )
        warmup_compile_single(
            state,
            warmup_val_batch,
            rng_streams,
            mean_streams,
            steps=1,
            run_train=False,
            run_val=True,
            is_masked_loss=is_masked_loss,
        )
    if args.debug_logging:
        logger.debug("Warmup compile completed")

    for epoch in range(restored_epoch + 1, restored_epoch + configs.num_epochs + 1):
        epoch_start = time.perf_counter()
        window_RNG_key = jax.random.split(window_RNG_key, num=1)[0]

        epoch_train_metrics = []
        epoch_val_metrics = []

        dataset_order = jax.random.permutation(
            window_RNG_key,
            len(dataset_specs) if stream_datasets else len(datasets),
        )
        
        for ds_idx in dataset_order:
            if stream_datasets:
                nc_file, use_filter = dataset_specs[int(ds_idx)]
                if args.debug_logging:
                    load_start = time.perf_counter()
                train_ds, val_ds = load_dataset_for_file(nc_file, use_filter)
                logger.debug(
                    f"Dataset {nc_file.name} (filter={use_filter}): train={train_ds.shape[1]}, val={val_ds.shape[1]}"
                )
                if args.debug_logging:
                    logger.debug(
                        "Dataset load time {}: {:.3f}s",
                        nc_file.name,
                        time.perf_counter() - load_start,
                    )
                    logger.debug(
                        "Dataset {} wave_len={} train_steps={} val_steps={}",
                        nc_file.name,
                        train_ds["wave_number"].shape[0],
                        -(-train_ds.sizes["spectra"] // configs.batch_size),
                        -(-val_ds.sizes["spectra"] // configs.batch_size),
                    )
                if train_ds.sizes['spectra'] < configs.batch_size or val_ds.sizes['spectra'] < configs.batch_size:
                    logger.warning(
                        f"Skipping {nc_file.name}: insufficient spectra for batch size {configs.batch_size}"
                    )
                    del train_ds, val_ds
                    gc.collect()
                    continue
            else:
                train_ds, val_ds, dataset_name, use_filter = datasets[int(ds_idx)]
                logger.debug(
                    f"Dataset {dataset_name} (filter={use_filter}): train={train_ds.shape[1]}, val={val_ds.shape[1]}"
                )
                if args.debug_logging:
                    logger.debug(
                        "Dataset {} wave_len={} train_steps={} val_steps={}",
                        dataset_name,
                        train_ds["wave_number"].shape[0],
                        -(-train_ds.sizes["spectra"] // configs.batch_size),
                        -(-val_ds.sizes["spectra"] // configs.batch_size),
                    )
            
            match training_regime:
                case "One device":
                    
                    # Training
                    if args.debug_logging:
                        train_start = time.perf_counter()
                    state, train_metrics_ds = train_epoch(
                        state, epoch, train_ds, configs, rng_streams,
                        metric_writer, ckpt_manager, window_RNG_key, mean_streams,
                        is_masked_loss=is_masked_loss
                    )
                    if args.debug_logging:
                        logger.debug(
                            "Train epoch time (dataset {}) {:.3f}s",
                            nc_file.name if stream_datasets else dataset_name,
                            time.perf_counter() - train_start,
                        )
                    # Validation
                    if args.debug_logging:
                        val_start = time.perf_counter()
                    state, val_metrics_ds = validation_epoch(
                        state, epoch, val_ds, configs, rng_streams, 
                        metric_writer, ckpt_manager, mean_streams,
                        is_masked_loss=is_masked_loss
                    )
                    if args.debug_logging:
                        logger.debug(
                            "Validation time (dataset {}) {:.3f}s",
                            nc_file.name if stream_datasets else dataset_name,
                            time.perf_counter() - val_start,
                        )
                case "All devices":
                    
                    if args.debug_logging:
                        train_start = time.perf_counter()
                    state, train_metrics_ds = train_epoch_pmap(
                        state=state, epoch=epoch, train_ds=train_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager, 
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams,
                        is_masked_loss=is_masked_loss
                        )
                    if args.debug_logging:
                        logger.debug(
                            "Train epoch pmap time (dataset {}) {:.3f}s",
                            nc_file.name if stream_datasets else dataset_name,
                            time.perf_counter() - train_start,
                        )
                    if args.debug_logging:
                        val_start = time.perf_counter()
                    state, val_metrics_ds = validation_epoch_pmap(
                        state=state, epoch=epoch, val_ds=val_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager,
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams,
                        is_masked_loss=is_masked_loss
                        )
                    if args.debug_logging:
                        logger.debug(
                            "Validation epoch pmap time (dataset {}) {:.3f}s",
                            nc_file.name if stream_datasets else dataset_name,
                            time.perf_counter() - val_start,
                        )
                case _:
                    raise Exception(f"Specify training_regime correctly!")
            
            epoch_train_metrics.append(train_metrics_ds)
            epoch_val_metrics.append(val_metrics_ds)

            if stream_datasets:
                del train_ds, val_ds
                gc.collect()

        if not epoch_train_metrics or not epoch_val_metrics:
            logger.warning("No datasets were processed this epoch. Check batch size and dataset sizes.")
            break

        train_metrics.append(
            jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_train_metrics)
        )
        val_metrics.append(
            jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_val_metrics)
        )
        
        # Write epoch+1 to the state
        state = update_epoch(state)
        
        # Logging
        if epoch % configs.log_every_epochs == 0:
            
            params0 = jax.tree.map(lambda x: x[0], state.params)
            
            # Making a prediction on a dummy for logging in tensorboard
            dummy_prediction = predict(
                state.apply_fn,
                {"params": params0},
                dummy_example,
                dummy_example["mask"],
            )
            
            # Calculating a loss for plotting
            dummy_spectra = jnp.squeeze(dummy_example["spectra"])
            dummy_pred_spectra = jnp.squeeze(dummy_prediction["predicted_spectra"])
            
            match configs.loss_fn if hasattr(configs, 'loss_fn') else "CorrGamma":
                case "MSE":
                    loss = (dummy_pred_spectra - dummy_spectra) ** 2
                case "CorrGamma":
                    dummy_ratio = dummy_spectra / dummy_pred_spectra
                    loss = (( dummy_ratio - 1) - jnp.log( dummy_ratio ))
                case _:
                    raise Exception(f"Specify loss_fn correctly in config!")
            
            fig_res, ax_res = plot_results_train(dummy_prediction, state.step[0], state.epoch[0], args.model_tag)
            metric_writer.add_figure('model_predictions', fig_res, global_step=state.epoch[0])

            fig_loss, ax_loss = plot_loss(
                dummy_wave_number,
                loss,
                state.step[0],
                state.epoch[0],
                args.model_tag,
                mask=dummy_example["mask"],
            )
            metric_writer.add_figure('model_prediction_losses', fig_loss, global_step=state.epoch[0])
            
            metric_writer.add_scalar("train/train_loss_epoch",          train_metrics[-1]["train_loss_step"],   state.epoch[0])
            metric_writer.add_scalar("val/val_loss_epoch",              val_metrics[-1]["val_loss_step"],       state.epoch[0])
            
            metric_writer.add_scalar("train/train_loss_step",           train_metrics[-1]["train_loss_step"],   state.step[0])
            metric_writer.add_scalar("val/val_loss_step",               val_metrics[-1]["val_loss_step"],       state.step[0])
            
            metric_writer.add_scalar("grad/train/grad_min_step",        train_metrics[-1]["grad_min"],          state.step[0])
            metric_writer.add_scalar("grad/train/grad_mean_step",       train_metrics[-1]["grad_mean"],         state.step[0])
            metric_writer.add_scalar("grad/train/grad_median_step",     train_metrics[-1]["grad_median"],       state.step[0])
            metric_writer.add_scalar("grad/train/grad_max_step",        train_metrics[-1]["grad_max"],          state.step[0])
            
            for gpu_stats in gpustat.new_query():
                log_gpu_usage(gpu_stats.entry, state.step[0], metric_writer)
            
            # Extract first replica and convert to host arrays (removes sharding metadata)
            single_state = jax.device_get(jax.tree.map(lambda x: x[0], state))
            if args.debug_logging:
                ckpt_start = time.perf_counter()
            ckpt_manager.save(
                step=int(single_state.step),
                args=ocp.args.StandardSave(single_state),
            )
            if args.debug_logging:
                logger.debug(
                    "checkpoint saved in {:.3f}s",
                    time.perf_counter() - ckpt_start,
                )

        early_stop = early_stop.update(val_metrics[-1]["val_loss_step"])
        best_metric_value = min(metric["val_loss_step"] for metric in val_metrics)
        metrics_diff = val_metrics[-1]["val_loss_step"] - best_metric_value
        total_epoch_time = time.perf_counter() - epoch_start
        if is_early_stop:
            logger.info(
                f"Epoch {epoch} | "
                f"Time: {total_epoch_time:.2f}s | "
                f"train_loss={train_metrics[-1]['train_loss_step']:.4e} | "
                f"val_loss={val_metrics[-1]['val_loss_step']:.4e} | "
                f"patience={early_stop.patience_count}/{patience}"
            )
        else:
            logger.info(
                f"Epoch {epoch} | "
                f"Time: {total_epoch_time:.2f}s | "
                f"train_loss={train_metrics[-1]['train_loss_step']:.4e} | "
                f"val_loss={val_metrics[-1]['val_loss_step']:.4e}"
            )
        if is_early_stop and early_stop.should_stop:
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

    ckpt_manager.wait_until_finished()
    metric_writer.close()
    logger.info("Training completed")


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)
