import gpustat
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import lax
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
import optax
from functools import partial
from loguru import logger

from spectraformer.input_pipeline import Batch, batch_sampler

def nan_inf_check(array):
    # NaN or Inf check using lax
    nan_check_array = jnp.any(jnp.isnan(array))
    inf_check_array = jnp.any(jnp.isinf(array))
    # Use lax.cond to act on the condition
    lax.cond(nan_check_array, lambda _: jax.debug.print(f"NaN detected in array"), lambda _: None, operand=None)
    lax.cond(inf_check_array, lambda _: jax.debug.print(f"Inf detected in array"), lambda _: None, operand=None)

def my_geometric_mean(loss, eps=1e-8):
    """
    Geometric mean calculating using a formula:
    GM(x)=exp( 1/N * sum( log(x_i) ) )
    """
    
    # Making sure to have no negative values in the loss with all information keeping
    non_negative = abs(loss)
    # Making sure having strictly positive values
    clipped = jnp.clip(non_negative, eps)
    # Calculating the log
    log_values = jnp.log(clipped)
    # Log averaging
    mean_log = jnp.mean(log_values)
    # Going back from log to normal value by exponentiation
    return jnp.exp(mean_log)


def _hidden_region_mask(mask):
    """Convert a visible-token mask into a broadcastable hidden-region mask."""
    hidden_mask = jnp.logical_not(mask)
    if hidden_mask.ndim == 1:
        hidden_mask = hidden_mask[None, :, None]
    elif hidden_mask.ndim == 2:
        hidden_mask = hidden_mask[:, :, None]
    else:
        hidden_mask = jnp.expand_dims(hidden_mask, axis=-1)
    return hidden_mask


def _masked_gamma_loss(target, pred, mask, reduction: str, eps: float = 1e-8, is_masked_loss=True):
    """Gamma deviance computed only on hidden positions."""
    target = jnp.clip(target, eps, None)
    pred = jnp.clip(pred, eps, None)
    ratio = target / pred
    loss_array = (ratio - 1.0) - jnp.log(ratio)

    hidden_mask = _hidden_region_mask(mask).astype(loss_array.dtype) if is_masked_loss else jnp.ones_like(loss_array)
    masked_loss_array = loss_array * hidden_mask
    masked_count = jnp.maximum(jnp.sum(hidden_mask), 1.0)

    match reduction:
        case 'Arithmetic':
            return jnp.sum(masked_loss_array) / masked_count
        case 'Geometric':
            return jnp.exp(
                jnp.sum(jnp.log(jnp.clip(loss_array, eps)) * hidden_mask) / masked_count
            )
        case _:
            raise Exception(f"You didn't specify a mean to be used!")


def _masked_mse_loss(target, pred, mask):
    """Mean squared error computed only on hidden positions."""
    hidden_mask = _hidden_region_mask(mask).astype(pred.dtype)
    squared_error = (pred - target) ** 2
    masked_squared_error = squared_error * hidden_mask
    masked_count = jnp.maximum(jnp.sum(hidden_mask), 1.0)
    return jnp.sum(masked_squared_error) / masked_count


def _build_dynamic_mask_windows(
    wave_number,
    rng_key,
    chunk_size: float,
    min_chunks: int,
    max_chunks: int,
):
    wave_min = float(wave_number[0])
    wave_max = float(wave_number[-1])
    span = wave_max - wave_min
    if span <= chunk_size:
        return []

    max_chunks_possible = int(span // chunk_size)
    if max_chunks_possible < 1:
        return []

    min_chunks = max(1, min(min_chunks, max_chunks_possible))
    max_chunks = max(1, min(max_chunks, max_chunks_possible))

    num_chunks = int(
        jax.random.randint(rng_key, (), minval=min_chunks, maxval=max_chunks + 1)
    )
    max_start = wave_max - chunk_size

    windows = []
    attempts = 0
    max_attempts = num_chunks * 50
    key = rng_key

    while len(windows) < num_chunks and attempts < max_attempts:
        key, subkey = jax.random.split(key)
        start = float(jax.random.uniform(subkey, (), minval=wave_min, maxval=max_start))
        end = start + chunk_size
        overlap = any((start < w_end) and (end > w_start) for w_start, w_end in windows)
        if not overlap:
            windows.append((start, end))
        attempts += 1

    windows.sort(key=lambda w: w[0])
    return windows


def _build_dynamic_mask_windows_np(
    wave_number,
    rng: np.random.Generator,
    chunk_size: float,
    min_chunks: int,
    max_chunks: int,
):
    wave_min = float(wave_number[0])
    wave_max = float(wave_number[-1])
    span = wave_max - wave_min
    if span <= chunk_size:
        return []

    max_chunks_possible = int(span // chunk_size)
    if max_chunks_possible < 1:
        return []

    min_chunks = max(1, min(min_chunks, max_chunks_possible))
    max_chunks = max(1, min(max_chunks, max_chunks_possible))

    num_chunks = int(rng.integers(min_chunks, max_chunks + 1))
    max_start = wave_max - chunk_size

    windows = []
    attempts = 0
    max_attempts = num_chunks * 50

    while len(windows) < num_chunks and attempts < max_attempts:
        start = float(rng.uniform(wave_min, max_start))
        end = start + chunk_size
        overlap = any((start < w_end) and (end > w_start) for w_start, w_end in windows)
        if not overlap:
            windows.append((start, end))
        attempts += 1

    windows.sort(key=lambda w: w[0])
    return windows


def _apply_mask_to_batch(
    batch: Batch,
    wave_number_raw,
    mask_windows,
    default_mask_value: float = -1,
) -> Batch:
    mask = jnp.ones_like(wave_number_raw, dtype=bool)
    for start, end in mask_windows:
        mask = mask & ~((wave_number_raw > start) & (wave_number_raw < end))

    mask_reshaped = mask[None, :, None]
    masked_spectra = jnp.where(mask_reshaped, batch["spectra"], default_mask_value)

    return Batch(
        spectra=batch["spectra"],
        masked_spectra=masked_spectra,
        wave_number=batch["wave_number"],
        mask=mask,
    )


def _summarize_mask_windows(mask_windows, max_list: int = 6) -> str:
    if not mask_windows:
        return "chunks=0"
    if len(mask_windows) <= max_list:
        return f"chunks={len(mask_windows)} windows={mask_windows}"
    return (
        f"chunks={len(mask_windows)} first={mask_windows[0]} last={mask_windows[-1]}"
    )


def _should_log_batch(batch_idx: int, log_every_batches: int) -> bool:
    return log_every_batches > 0 and batch_idx % log_every_batches == 0


def _log_batch_summary(batch, batch_idx: int, prefix: str) -> None:
    spectra = batch["spectra"]
    masked = batch.get("masked_spectra", None)
    wave = batch.get("wave_number", None)
    mask = batch.get("mask", None)
    logger.debug(
        "{}: batch={} spectra_shape={} dtype={} masked_shape={} wave_shape={} mask_shape={}",
        prefix,
        batch_idx,
        getattr(spectra, "shape", "?"),
        getattr(spectra, "dtype", "?"),
        getattr(masked, "shape", "?"),
        getattr(wave, "shape", "?"),
        getattr(mask, "shape", "?"),
    )
    if mask is not None:
        mask_mean = float(jnp.mean(mask))
        logger.debug("{}: batch={} mask_true_fraction={:.4f}", prefix, batch_idx, mask_mean)


def warmup_compile_single(
    state,
    batch: Batch,
    rng_streams,
    mean_streams,
    steps: int = 1,
    run_train: bool = True,
    run_val: bool = True,
    is_masked_loss: bool = True,
):
    for _ in range(max(1, steps)):
        if run_train:
            _, train_metrics = train_step(
                state,
                batch,
                rng_streams["dropout"],
                mean_streams["mean"],
                is_masked_loss,
            )
            jax.block_until_ready(train_metrics)
        if run_val:
            _, val_metrics = validation_step(
                state,
                batch,
                rng_streams["dropout"],
                mean_streams["mean"],
                is_masked_loss,
            )
            jax.block_until_ready(val_metrics)


def warmup_compile_pmap(
    state,
    batch: Batch,
    rng_streams,
    mean_streams,
    num_devices: int,
    loss_fn: str,
    steps: int = 1,
    run_train: bool = True,
    run_val: bool = True,
    is_masked_loss: bool = True,
):
    match mean_streams["mean"]:
        case "Arithmetic":
            train_step_pmap = train_step_pmap_arithmetic
            validation_step_pmap = validation_step_pmap_arithmetic
        case "Geometric":
            train_step_pmap = train_step_pmap_geometric
            validation_step_pmap = validation_step_pmap_geometric
        case _:
            raise ValueError("Mean is incorrect.")

    devices = jax.devices()
    batch_sharded = shard_batch(batch)
    dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
    dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]
    dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)

    for _ in range(max(1, steps)):
        if run_train:
            _, train_metrics = train_step_pmap(
                state,
                batch_sharded,
                dropout_sharded,
                num_devices,
                loss_fn,
                is_masked_loss,
            )
            jax.block_until_ready(train_metrics)
        if run_val:
            _, val_metrics = validation_step_pmap(
                state,
                batch_sharded,
                dropout_sharded,
                num_devices,
                loss_fn,
                is_masked_loss,
            )
            jax.block_until_ready(val_metrics)


def warmup_lower_compile_pmap(
    state,
    batch: Batch,
    rng_streams,
    mean_streams,
    num_devices: int,
    loss_fn: str,
    is_masked_loss: bool = True,
):
    match mean_streams["mean"]:
        case "Arithmetic":
            train_step_pmap = train_step_pmap_arithmetic
            validation_step_pmap = validation_step_pmap_arithmetic
        case "Geometric":
            train_step_pmap = train_step_pmap_geometric
            validation_step_pmap = validation_step_pmap_geometric
        case _:
            raise ValueError("Mean is incorrect.")

    devices = jax.devices()
    batch_sharded = shard_batch(batch)
    dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
    dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]
    dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)

    train_step_pmap.lower(
        state,
        batch_sharded,
        dropout_sharded,
        num_devices,
        loss_fn,
        is_masked_loss,
    ).compile()

    validation_step_pmap.lower(
        state,
        batch_sharded,
        dropout_sharded,
        num_devices,
        loss_fn,
        is_masked_loss,
    ).compile()


def log_gpu_usage(gpustat_entry, step, writer):
    name = f"[{gpustat_entry['name']}/{gpustat_entry['index']}"

    writer.add_scalar(f"{name}/usage", gpustat_entry["utilization.gpu"], step)
    writer.add_scalar(
        f"{name}/memory",
        100 * gpustat_entry["memory.used"] / gpustat_entry["memory.total"],
        step,
    )

def shard_batch(batch: Batch) -> Batch:
    """Automatically handles device count and batch size alignment"""
    devices = jax.devices()
    num_devices = len(devices)
    batch_size = batch['spectra'].shape[0]
    
    # Ensure batch is divisible by devices
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by {num_devices} devices. "
            f"Try batch_size={num_devices * (batch_size // num_devices)}"
        )

    sharded_batch = {}
    
    for k, v in batch.items():
        # Convert to JAX array to ensure ndim attribute exists
        v = jnp.asarray(v)
        if k in ['wave_number', 'mask']:
            # Constant arrays
            v_expanded = jnp.expand_dims(v, axis=0) if v.ndim < 2 else v
            sharded_batch[k] = jax.device_put_replicated(v_expanded, devices)
        else:
            # Variable arrays
            shards = jnp.split(v, num_devices)
            if len(shards) != num_devices:
                raise RuntimeError(
                    f"Split created {len(shards)} shards but have {num_devices} devices. "
                    f"Batch dim: {v.shape[0]}, devices: {num_devices}"
                )
            sharded_batch[k] = jax.device_put_sharded(shards, devices)
    
    return Batch(**sharded_batch)


@partial(jax.jit, static_argnames=("configs_mean", "is_masked_loss"))
def train_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean,
    is_masked_loss=True
):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction=configs_mean,
            is_masked_loss=is_masked_loss,
        )
        
        return loss
    
    grad_fn = jax.value_and_grad(corrected_gamma_loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Flatten the PyTree of gradients
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    # Concatenate all gradients into a single array for statistics
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    
    nan_inf_check(all_grads)
    
    # Compute gradient parameters for logging
    grad_min = jnp.min(all_grads)
    grad_mean = jnp.mean(all_grads)
    grad_median = jnp.median(all_grads)
    grad_max = jnp.max(all_grads)
    
    state = state.apply_gradients(grads=grads)
    train_metrics = {
        "train_loss": loss,
        "grad_min": grad_min,
        "grad_mean": grad_mean,
        "grad_median": grad_median,
        "grad_max": grad_max
        }
    return state, train_metrics

@partial(jax.jit, static_argnames=("configs_mean", "is_masked_loss"))
def validation_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean,
    is_masked_loss=True
):
    dropout_val_key = jax.random.fold_in(key=dropout_key, data=state.step)
    
    pred_spectra = state.apply_fn(
                {"params": state.params},
                batch["masked_spectra"],
                batch["wave_number"],
                batch["mask"],
                training=False,
                rngs={"dropout": dropout_val_key},
            )
    
    nan_inf_check(pred_spectra)
    
    def val_corrected_gamma_fn(params):
        loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction=configs_mean,
            is_masked_loss=is_masked_loss,
        )
        return loss

    corrected_gamma_loss = val_corrected_gamma_fn(state.params)
    mse = _masked_mse_loss(batch["spectra"], pred_spectra, batch["mask"])             # Mean square error on hidden positions only
    
    val_metrics = {
        "val_corrected_gamma_loss": corrected_gamma_loss,
        "MSE": mse
        }
    return state, val_metrics

def train_epoch(
    state, epoch: int, train_ds, configs, rng_streams, metric_writer, ckpt_manager, window_RNG_key, mean_streams,
    is_masked_loss: bool = True,
):
    masked_interval_starts_config = configs.masked_interval_starts
    masked_interval_ends_config = configs.masked_interval_ends

    dynamic_mask = getattr(configs, "dynamic_mask", False)
    debug_logging = getattr(configs, "debug_logging", False)
    debug_log_every_batches = getattr(configs, "debug_log_every_batches", 1)
    mask_chunk_size = getattr(configs, "mask_chunk_size", 200)
    mask_chunk_min = getattr(configs, "mask_chunk_min", 2)
    mask_chunk_max = getattr(configs, "mask_chunk_max", 6)
    default_mask_value = getattr(configs, "default_mask_value", -1)
    wave_number_raw = jnp.asarray(train_ds["wave_number"].values)

    if debug_logging:
        logger.debug(
            "train_epoch: epoch={} spectra={} wave_len={} batch_size={}",
            epoch,
            train_ds.sizes.get("spectra", "?"),
            wave_number_raw.shape[0],
            configs.batch_size,
        )
    
    ######################################################################################
    if (not dynamic_mask) and configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item()
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        masked_interval_starts_config[1] = window_end
        masked_interval_ends_config[0] = window_start
    ######################################################################################
    
    mask_windows = [] if dynamic_mask else list(
        zip(masked_interval_starts_config, masked_interval_ends_config)
    )

    if debug_logging:
        logger.debug(
            "train_epoch: epoch={} dynamic_mask={} {}",
            epoch,
            dynamic_mask,
            _summarize_mask_windows(mask_windows),
        )
    
    
    if debug_logging:
        loader_start = time.perf_counter()
    data_loader = batch_sampler(
        train_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch_idx, batch in enumerate(data_loader):
        if debug_logging:
            batch_start = time.perf_counter()
        if debug_logging and batch_idx == 0:
            logger.debug(
                "train_epoch: time_to_first_batch={:.3f}s",
                batch_start - loader_start,
            )
            _log_batch_summary(batch, batch_idx, "train_epoch")
        if dynamic_mask:
            if debug_logging:
                mask_build_start = time.perf_counter()
            mask_seed = int(jax.random.fold_in(window_RNG_key, batch_idx)[0])
            mask_rng = np.random.default_rng(mask_seed)
            mask_windows = _build_dynamic_mask_windows_np(
                wave_number_raw,
                mask_rng,
                chunk_size=mask_chunk_size,
                min_chunks=mask_chunk_min,
                max_chunks=mask_chunk_max,
            )
            if debug_logging:
                mask_build_time = time.perf_counter() - mask_build_start
                mask_apply_start = time.perf_counter()
            batch = _apply_mask_to_batch(
                batch,
                wave_number_raw,
                mask_windows,
                default_mask_value=default_mask_value,
            )
            if debug_logging:
                mask_apply_time = time.perf_counter() - mask_apply_start
            if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
                logger.debug(
                    "train_epoch: batch={} {}",
                    batch_idx,
                    _summarize_mask_windows(mask_windows),
                )
                logger.debug(
                    "train_epoch: batch={} mask_build={:.4f}s mask_apply={:.4f}s",
                    batch_idx,
                    mask_build_time,
                    mask_apply_time,
                )
        if debug_logging: 
            step_start = time.perf_counter()
        state, batch_metrics = train_step(state, batch, rng_streams["dropout"], mean_streams["mean"], is_masked_loss)
        if debug_logging: 
            step_time = time.perf_counter() - step_start
        if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
            total_time = time.perf_counter() - batch_start
            logger.debug(
                "train_epoch: batch={} step_time={:.3f}s total_time={:.3f}s",
                batch_idx,
                step_time,
                total_time,
            )
            if step_time > 1.0:
                logger.debug(
                    "train_epoch: batch={} slow_step_warning step_time={:.3f}s",
                    batch_idx,
                    step_time,
                )
        metrics.append(batch_metrics)

    metrics = stack_forest(metrics)
    avg_metrics = jax.tree.map(jnp.mean, metrics)  # Log the average error of the epoch

    logger.info(f"Epoch {epoch + 1} -- Loss {avg_metrics['train_loss'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("train/loss", avg_metrics["train_loss"].item(), state.step)
        metric_writer.add_scalar("train/grad_min", avg_metrics["grad_min"].item(), state.step)
        metric_writer.add_scalar("train/grad_mean", avg_metrics["grad_mean"].item(), state.step)
        metric_writer.add_scalar("train/grad_median", avg_metrics["grad_median"].item(), state.step)
        metric_writer.add_scalar("train/grad_max", avg_metrics["grad_max"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics

def validation_epoch(
    state, epoch: int, val_ds, configs, rng_streams, metric_writer, ckpt_manager, mean_streams,
    is_masked_loss: bool = True,
):
    dynamic_mask = getattr(configs, "dynamic_mask", False)
    debug_logging = getattr(configs, "debug_logging", False)
    debug_log_every_batches = getattr(configs, "debug_log_every_batches", 1)
    mask_chunk_size = getattr(configs, "mask_chunk_size", 200)
    mask_chunk_min = getattr(configs, "mask_chunk_min", 2)
    mask_chunk_max = getattr(configs, "mask_chunk_max", 6)
    default_mask_value = getattr(configs, "default_mask_value", -1)
    wave_number_raw = jnp.asarray(val_ds["wave_number"].values)

    if debug_logging:
        logger.debug(
            "validation_epoch: epoch={} spectra={} wave_len={} batch_size={}",
            epoch,
            val_ds.sizes.get("spectra", "?"),
            wave_number_raw.shape[0],
            configs.batch_size,
        )

    if dynamic_mask:
        base_key = jax.random.PRNGKey(getattr(configs, "root_rng_seed", 0))
        mask_seed = int(jax.random.fold_in(base_key, epoch)[0])
        mask_rng = np.random.default_rng(mask_seed)
        mask_windows = _build_dynamic_mask_windows_np(
            wave_number_raw,
            mask_rng,
            chunk_size=mask_chunk_size,
            min_chunks=mask_chunk_min,
            max_chunks=mask_chunk_max,
        )
        mask_windows_for_loader = []
    else:
        mask_windows = list(
            zip(configs.masked_interval_starts, configs.masked_interval_ends)
        )
        mask_windows_for_loader = mask_windows
    if debug_logging:
        logger.debug(
            "validation_epoch: epoch={} dynamic_mask={} {}",
            epoch,
            dynamic_mask,
            _summarize_mask_windows(mask_windows),
        )
    if debug_logging:
        loader_start = time.perf_counter()
    data_loader = batch_sampler(
        val_ds,
        mask_windows_for_loader,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch_idx, batch in enumerate(data_loader):
        if debug_logging:
            batch_start = time.perf_counter()
        if debug_logging and batch_idx == 0:
            logger.debug(
                "validation_epoch: time_to_first_batch={:.3f}s",
                batch_start - loader_start,
            )
            _log_batch_summary(batch, batch_idx, "validation_epoch")
        if dynamic_mask:
            if debug_logging:
                mask_apply_start = time.perf_counter()
            batch = _apply_mask_to_batch(
                batch,
                wave_number_raw,
                mask_windows,
                default_mask_value=default_mask_value,
            )
            if debug_logging:
                mask_apply_time = time.perf_counter() - mask_apply_start
            if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
                logger.debug(
                    "validation_epoch: batch={} {}",
                    batch_idx,
                    _summarize_mask_windows(mask_windows),
                )
                logger.debug(
                    "validation_epoch: batch={} mask_apply={:.4f}s",
                    batch_idx,
                    mask_apply_time,
                )
        if debug_logging: 
            step_start = time.perf_counter()
        state, batch_metrics = validation_step(state, batch, rng_streams["dropout"], mean_streams["mean"], is_masked_loss)
        if debug_logging: 
            step_time = time.perf_counter() - step_start
        if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
            total_time = time.perf_counter() - batch_start
            logger.debug(
                "validation_epoch: batch={} step_time={:.3f}s total_time={:.3f}s",
                batch_idx,
                step_time,
                total_time,
            )
            if step_time > 1.0:
                logger.debug(
                    "validation_epoch: batch={} slow_step_warning step_time={:.3f}s",
                    batch_idx,
                    step_time,
                )
        metrics.append(batch_metrics)

    metrics = stack_forest(metrics)
    avg_metrics = jax.tree.map(jnp.mean, metrics)  # Log the average error of the epoch

    logger.info(f"Validation -- Epoch {epoch + 1} -- CorrGamma Loss {avg_metrics['val_corrected_gamma_loss'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("val/val_corrected_gamma_loss", avg_metrics["val_corrected_gamma_loss"].item(), state.step)
        metric_writer.add_scalar("val/MSE", avg_metrics["MSE"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics

# ==========================
#     MULTI-DEVICE TRAIN STEP WITH ARITHMETIC LOSS
# ==========================
@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(3,4,5)  # Add static arg
)
# @jax.jit
def train_step_pmap_arithmetic(
    state: TrainState,
    batch,
    dropout_key,
    num_devices: int,  # Passed explicitly
    loss_fn: str = "CorrGamma",  # Default loss function
    is_masked_loss: bool = True
):
    # Get device index for unique key folding
    device_idx = lax.axis_index('batch')
    
    # Fold key with device-specific index
    folded_key = jax.random.fold_in(dropout_key, device_idx)
    dropout_train_key = folded_key
    
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        local_loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction="Arithmetic",
            is_masked_loss=is_masked_loss,
        )
        
        return local_loss # scalar per device
    
    def mse_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        return _masked_mse_loss(batch["spectra"], pred_spectra, batch["mask"])
    
    match loss_fn:
        case "CorrGamma":
            local_loss, local_grads = jax.value_and_grad(corrected_gamma_loss_fn)(state.params)
        case "MSE":
            local_loss, local_grads = jax.value_and_grad(mse_loss_fn)(state.params)
        case _:
            raise ValueError(f"Unknown loss function: {loss_fn}. Supported: 'CorrGamma', 'MSE'.")
    
    # Average loss across devices
    global_loss = lax.psum(local_loss, axis_name="batch") / num_devices 
    # Average gradients across devices
    final_grads = jax.tree.map(lambda g: lax.psum(g, axis_name="batch") / num_devices, local_grads)
    
    # Check final_grads for NaNs, Infs
    flat_grads, _ = jax.tree_util.tree_flatten(final_grads)
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    
    nan_inf_check(all_grads)

    zeros_check = jnp.allclose(all_grads, 0)
    lax.cond(zeros_check, lambda _: jax.debug.print("Zero gradients warning!"), lambda _: None, operand=None)
    
    # Compute gradient parameters for logging
    grad_min = jnp.min(all_grads)
    grad_mean = jnp.mean(all_grads)
    grad_median = jnp.median(all_grads)
    grad_max = jnp.max(all_grads)
    
    # single parameter update
    new_state = state.apply_gradients(grads=final_grads)
    
    train_metrics = {
    "train_loss_step": global_loss,
    "grad_min": grad_min,
    "grad_mean": grad_mean,
    "grad_median": grad_median,
    "grad_max": grad_max
    }
    return new_state, train_metrics

# ==========================
#     MULTI-DEVICE VALIDATION STEP WITH ARITHMETIC LOSS
# ==========================
@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(3,4,5)  # Add static arg
)
# @jax.jit
def validation_step_pmap_arithmetic(
    state: TrainState, 
    batch, 
    dropout_key,
    num_devices: int,  # Passed explicitly
    loss_fn: str = "CorrGamma",  # Default loss function
    is_masked_loss: bool = True
):
    # Get device index for unique key folding
    device_idx = lax.axis_index('batch')
    
    # Fold key with device-specific index
    folded_key = jax.random.fold_in(dropout_key, device_idx)
    dropout_train_key = folded_key
    
    # Local loss per device function
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=False,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        local_loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction="Arithmetic",
            is_masked_loss=is_masked_loss,
        )
        
        return local_loss # scalar per device
    
    def mse_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=False,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        return _masked_mse_loss(batch["spectra"], pred_spectra, batch["mask"])
    
    match loss_fn:
        case "CorrGamma":
            local_loss = corrected_gamma_loss_fn(state.params)
        case "MSE":
            local_loss = mse_loss_fn(state.params)
        case _:
            raise ValueError(f"Unknown loss function: {loss_fn}. Supported: 'CorrGamma', 'MSE'.")
    
    # Average loss across devices
    global_loss = lax.psum(local_loss, axis_name="batch") / num_devices
    
    val_metrics = {
        "val_loss_step": global_loss
        }
    
    return state, val_metrics

# ==========================
#     MULTI-DEVICE TRAIN STEP WITH GEOMETRIC LOSS
# ==========================
@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(3,4)  # Add static arg
)
# @jax.jit
def train_step_pmap_geometric(
    state: TrainState,
    batch,
    dropout_key,
    num_devices: int,  # Passed explicitly
    is_masked_loss: bool = True
):
    # Get device index for unique key folding
    device_idx = lax.axis_index('batch')
    
    # Fold key with device-specific index
    folded_key = jax.random.fold_in(dropout_key, device_idx)
    dropout_train_key = folded_key
    
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        local_loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction="Geometric",
            is_masked_loss=is_masked_loss,
        )
        
        return local_loss # scalar per device
    
    
    local_loss, local_grads = jax.value_and_grad(corrected_gamma_loss_fn)(state.params)
    
    log_local = jnp.log(jnp.clip(local_loss, 1e-8))
    sum_log_local = lax.psum(log_local, axis_name="batch")
    mean_log_local = sum_log_local / num_devices
    global_loss = jnp.exp(mean_log_local)
    
    weights = (global_loss / local_loss) / num_devices  # shape: scalar per device
    
    weighted_grads = jax.tree.map(lambda g: g * weights, local_grads)
    final_grads = jax.tree.map(lambda wg: lax.psum(wg, axis_name="batch"), weighted_grads)
    
    flat_grads, _ = jax.tree_util.tree_flatten(final_grads)
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    
    # Check final_grads for NaNs, Infs
    nan_inf_check(all_grads)
    zeros_check = jnp.allclose(all_grads, 0)
    lax.cond(zeros_check, lambda _: jax.debug.print("Zero gradients warning!"), lambda _: None, operand=None)
    
    # Compute gradient parameters for logging
    grad_min = jnp.min(all_grads)
    grad_mean = jnp.mean(all_grads)
    grad_median = jnp.median(all_grads)
    grad_max = jnp.max(all_grads)
    
    # single parameter update
    new_state = state.apply_gradients(grads=final_grads)
    
    
    train_metrics = {
    "train_loss_step": global_loss,
    "grad_min": grad_min,
    "grad_mean": grad_mean,
    "grad_median": grad_median,
    "grad_max": grad_max
    }
    return new_state, train_metrics


# ==========================
#     MULTI-DEVICE VALIDATION STEP WITH GEOMETRIC LOSS
# ==========================
@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(3,4)  # Add static arg
)
# @jax.jit
def validation_step_pmap_geometric(
    state: TrainState, 
    batch, 
    dropout_key,
    num_devices: int,  # Passed explicitly
    is_masked_loss: bool = True
):
    # Get device index for unique key folding
    device_idx = lax.axis_index('batch')
    
    # Fold key with device-specific index
    folded_key = jax.random.fold_in(dropout_key, device_idx)
    dropout_train_key = folded_key
    # Local loss per device function
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=False,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_inf_check(pred_spectra)

        local_loss = _masked_gamma_loss(
            batch["spectra"],
            pred_spectra,
            batch["mask"],
            reduction="Geometric",
            is_masked_loss=is_masked_loss,
        )
        
        return local_loss # scalar per device
    
    
    local_loss = corrected_gamma_loss_fn(state.params)
    
    log_local = jnp.log(jnp.clip(local_loss, 1e-8))
    sum_log_local = lax.psum(log_local, axis_name="batch")
    mean_log_local = sum_log_local / num_devices
    global_loss = jnp.exp(mean_log_local)
    
    val_metrics = {
        "val_loss_step": global_loss
        }
    
    return state, val_metrics

def train_epoch_pmap(
    state, 
    epoch: int, 
    train_ds,
    configs,
    rng_streams, 
    metric_writer, 
    ckpt_manager, 
    window_RNG_key, 
    mean_streams,
    is_masked_loss: bool = True
):
    # Choosing the train step function WITHOUT pmapping scalar string
    match mean_streams["mean"]:
        case "Arithmetic":
            train_step_pmap = train_step_pmap_arithmetic
        case "Geometric":
            train_step_pmap = train_step_pmap_geometric
        case _:
            raise ValueError("Mean is incorrect.")
    
    # Get current devices
    devices = jax.devices()
    num_devices = len(devices)
    
    dynamic_mask = getattr(configs, "dynamic_mask", False)
    debug_logging = getattr(configs, "debug_logging", False)
    debug_log_every_batches = getattr(configs, "debug_log_every_batches", 1)
    mask_chunk_size = getattr(configs, "mask_chunk_size", 200)
    mask_chunk_min = getattr(configs, "mask_chunk_min", 2)
    mask_chunk_max = getattr(configs, "mask_chunk_max", 6)
    default_mask_value = getattr(configs, "default_mask_value", -1)
    wave_number_raw = jnp.asarray(train_ds["wave_number"].values)

    if debug_logging:
        logger.debug(
            "train_epoch_pmap: epoch={} spectra={} wave_len={} batch_size={} devices={}",
            epoch,
            train_ds.sizes.get("spectra", "?"),
            wave_number_raw.shape[0],
            configs.batch_size,
            num_devices,
        )

    if (not dynamic_mask) and configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item()
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        configs.masked_interval_starts[1] = window_end
        configs.masked_interval_ends[0] = window_start
    
    mask_windows = [] if dynamic_mask else list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    if debug_logging:
        logger.debug(
            "train_epoch_pmap: epoch={} dynamic_mask={} {}",
            epoch,
            dynamic_mask,
            _summarize_mask_windows(mask_windows),
        )

    if debug_logging:
        loader_start = time.perf_counter()
    data_loader = batch_sampler(
        train_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )

    metrics_list = []

    for batch_idx, batch in enumerate(data_loader):
        if debug_logging:
            batch_start = time.perf_counter()
        if debug_logging and batch_idx == 0:
            logger.debug(
                "train_epoch_pmap: time_to_first_batch={:.3f}s",
                batch_start - loader_start,
            )
            _log_batch_summary(batch, batch_idx, "train_epoch_pmap")
        if dynamic_mask:
            if debug_logging:
                mask_build_start = time.perf_counter()
            mask_seed = int(jax.random.fold_in(window_RNG_key, batch_idx)[0])
            mask_rng = np.random.default_rng(mask_seed)
            mask_windows = _build_dynamic_mask_windows_np(
                wave_number_raw,
                mask_rng,
                chunk_size=mask_chunk_size,
                min_chunks=mask_chunk_min,
                max_chunks=mask_chunk_max,
            )
            if debug_logging:
                mask_build_time = time.perf_counter() - mask_build_start
                mask_apply_start = time.perf_counter()
            batch = _apply_mask_to_batch(
                batch,
                wave_number_raw,
                mask_windows,
                default_mask_value=default_mask_value,
            )
            if debug_logging:
                mask_apply_time = time.perf_counter() - mask_apply_start
            if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
                logger.debug(
                    "train_epoch_pmap: batch={} {}",
                    batch_idx,
                    _summarize_mask_windows(mask_windows),
                )
                logger.debug(
                    "train_epoch_pmap: batch={} mask_build={:.4f}s mask_apply={:.4f}s",
                    batch_idx,
                    mask_build_time,
                    mask_apply_time,
                )
        # 1) Shard the batch so each device gets a sub-batch
        if debug_logging:
            shard_start = time.perf_counter()
        batch_sharded = shard_batch(batch)
        if debug_logging:
            shard_time = time.perf_counter() - shard_start
        
        # 2) Create a dropout key for each device
        if debug_logging:
            dropout_shard_start = time.perf_counter()
        dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
        dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]  # Convert to list
        dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)
        if debug_logging:
            dropout_shard_time = time.perf_counter() - dropout_shard_start
        
        loss_fn=configs.loss_fn
        # 3) Run pmapped train step
        if debug_logging: 
            step_start = time.perf_counter()
        state, batch_metrics = train_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded,
            num_devices,
            loss_fn,
            is_masked_loss
            )
        if debug_logging: 
            step_time = time.perf_counter() - step_start
        if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
            total_time = time.perf_counter() - batch_start
            logger.debug(
                "train_epoch_pmap: batch={} step_time={:.3f}s total_time={:.3f}s",
                batch_idx,
                step_time,
                total_time,
            )
            if step_time > 1.0:
                logger.debug(
                    "train_epoch_pmap: batch={} slow_step_warning step_time={:.3f}s",
                    batch_idx,
                    step_time,
                )
            logger.debug(
                "train_epoch_pmap: batch={} shard={:.4f}s dropout_shard={:.4f}s",
                batch_idx,
                shard_time,
                dropout_shard_time,
            )
        # batch_metrics is a PyTree with shape [num_devices, ...] for each metric

        metrics_list.append(batch_metrics)
    
    # PROPER AGGREGATION:
    avg_metrics = {
        k: jnp.mean(jnp.stack([m[k] for m in metrics_list]))
        for k in metrics_list[0].keys()
    }
    
    # Verify final shapes
    for k, v in avg_metrics.items():
        if v.ndim != 0:
            jax.debug.print("Warning: Metric {k} not scalar after aggregation!")    
    return state, avg_metrics

def validation_epoch_pmap(
    state, 
    epoch: int, 
    val_ds,
    configs, 
    rng_streams, 
    metric_writer, 
    ckpt_manager, 
    window_RNG_key, 
    mean_streams,
    is_masked_loss: bool = True
):
    # Choosing the train step function WITHOUT pmapping scalar string
    match mean_streams["mean"]:
        case "Arithmetic":
            validation_step_pmap = validation_step_pmap_arithmetic
        case "Geometric":
            validation_step_pmap = validation_step_pmap_geometric
        case _:
            raise ValueError("Mean is incorrect.")
    
    # Get current devices
    devices = jax.devices()
    num_devices = len(devices)
    dynamic_mask = getattr(configs, "dynamic_mask", False)
    debug_logging = getattr(configs, "debug_logging", False)
    debug_log_every_batches = getattr(configs, "debug_log_every_batches", 1)
    mask_chunk_size = getattr(configs, "mask_chunk_size", 200)
    mask_chunk_min = getattr(configs, "mask_chunk_min", 2)
    mask_chunk_max = getattr(configs, "mask_chunk_max", 6)
    default_mask_value = getattr(configs, "default_mask_value", -1)
    wave_number_raw = jnp.asarray(val_ds["wave_number"].values)

    if debug_logging:
        logger.debug(
            "validation_epoch_pmap: epoch={} spectra={} wave_len={} batch_size={} devices={}",
            epoch,
            val_ds.sizes.get("spectra", "?"),
            wave_number_raw.shape[0],
            configs.batch_size,
            num_devices,
        )

    if dynamic_mask:
        base_key = jax.random.PRNGKey(getattr(configs, "root_rng_seed", 0))
        mask_seed = int(jax.random.fold_in(base_key, epoch)[0])
        mask_rng = np.random.default_rng(mask_seed)
        mask_windows = _build_dynamic_mask_windows_np(
            wave_number_raw,
            mask_rng,
            chunk_size=mask_chunk_size,
            min_chunks=mask_chunk_min,
            max_chunks=mask_chunk_max,
        )
        mask_windows_for_loader = []
    else:
        mask_windows = list(
            zip(configs.masked_interval_starts, configs.masked_interval_ends)
        )
        mask_windows_for_loader = mask_windows
    if debug_logging:
        logger.debug(
            "validation_epoch_pmap: epoch={} dynamic_mask={} {}",
            epoch,
            dynamic_mask,
            _summarize_mask_windows(mask_windows),
        )

    if debug_logging:
        loader_start = time.perf_counter()
    data_loader = batch_sampler(
        val_ds,
        mask_windows_for_loader,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )

    metrics_list = []

    for batch_idx, batch in enumerate(data_loader):
        if debug_logging:
            batch_start = time.perf_counter()
        if debug_logging and batch_idx == 0:
            logger.debug(
                "validation_epoch_pmap: time_to_first_batch={:.3f}s",
                batch_start - loader_start,
            )
            _log_batch_summary(batch, batch_idx, "validation_epoch_pmap")
        if dynamic_mask:
            if debug_logging:
                mask_apply_start = time.perf_counter()
            batch = _apply_mask_to_batch(
                batch,
                wave_number_raw,
                mask_windows,
                default_mask_value=default_mask_value,
            )
            if debug_logging:
                mask_apply_time = time.perf_counter() - mask_apply_start
            if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
                logger.debug(
                    "validation_epoch_pmap: batch={} {}",
                    batch_idx,
                    _summarize_mask_windows(mask_windows),
                )
                logger.debug(
                    "validation_epoch_pmap: batch={} mask_apply={:.4f}s",
                    batch_idx,
                    mask_apply_time,
                )
        # 1) Shard the batch so each device gets a sub-batch
        if debug_logging:
            shard_start = time.perf_counter()
        batch_sharded = shard_batch(batch)
        if debug_logging:
            shard_time = time.perf_counter() - shard_start

        # 2) Create a dropout key for each device
        if debug_logging:
            dropout_shard_start = time.perf_counter()
        dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
        dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]  # Convert to list
        dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)
        if debug_logging:
            dropout_shard_time = time.perf_counter() - dropout_shard_start

        loss_fn=configs.loss_fn
        # 3) Run pmapped train step
        if debug_logging: 
            step_start = time.perf_counter()
        _, batch_metrics = validation_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded,
            num_devices,
            loss_fn,
            is_masked_loss
            )
        if debug_logging: 
            step_time = time.perf_counter() - step_start
        if debug_logging and _should_log_batch(batch_idx, debug_log_every_batches):
            total_time = time.perf_counter() - batch_start
            logger.debug(
                "validation_epoch_pmap: batch={} step_time={:.3f}s total_time={:.3f}s",
                batch_idx,
                step_time,
                total_time,
            )
            if step_time > 1.0:
                logger.debug(
                    "validation_epoch_pmap: batch={} slow_step_warning step_time={:.3f}s",
                    batch_idx,
                    step_time,
                )
            logger.debug(
                "validation_epoch_pmap: batch={} shard={:.4f}s dropout_shard={:.4f}s",
                batch_idx,
                shard_time,
                dropout_shard_time,
            )
        # batch_metrics is a PyTree with shape [num_devices, ...] for each metric

        metrics_list.append(batch_metrics)

    # PROPER AGGREGATION:
    avg_metrics = {
        k: jnp.mean(jnp.stack([m[k] for m in metrics_list]))
        for k in metrics_list[0].keys()
    }
    
    # Verify final shapes
    for k, v in avg_metrics.items():
        if v.ndim != 0:
            jax.debug.print("Warning: Metric {k} not scalar after aggregation!")
        
    return state, avg_metrics

