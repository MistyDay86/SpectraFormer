import gpustat
import jax
import jax.numpy as jnp
# import numpy as np
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


@partial(jax.jit, static_argnames=("configs_mean",))
def train_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean
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
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        match configs_mean:
            case 'Arithmetic':
                loss = loss_array.mean()
            case 'Geometric':
                loss = my_geometric_mean(loss_array)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        
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

@partial(jax.jit, static_argnames=("configs_mean",))
def validation_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean
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
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        match configs_mean:
            case 'Arithmetic':
                loss = loss_array.mean()
            case 'Geometric':
                loss = my_geometric_mean(loss_array)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        return loss

    corrected_gamma_loss = val_corrected_gamma_fn(state.params)
    mse = optax.losses.squared_error(pred_spectra, batch["spectra"]).mean()             # Mean square error - normalized L2 loss - scalar value that evaluates the overall prediction accuracy of a model across the dataset
    
    val_metrics = {
        "val_corrected_gamma_loss": corrected_gamma_loss,
        "MSE": mse
        }
    return state, val_metrics

def train_epoch(
    state, epoch: int, train_ds, configs, rng_streams, metric_writer, ckpt_manager, window_RNG_key, mean_streams
):
    masked_interval_starts_config = configs.masked_interval_starts
    masked_interval_ends_config = configs.masked_interval_ends
    
    ######################################################################################
    if configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item() # from 10 to 100 percent of a half of a spectra lenght
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        masked_interval_starts_config[1] = window_end
        masked_interval_ends_config[0] = window_start
    ######################################################################################
    
    mask_windows = list(
        zip(masked_interval_starts_config, masked_interval_ends_config)
    )
    
    
    data_loader = batch_sampler(
        train_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch in data_loader:
        state, batch_metrics = train_step(state, batch, rng_streams["dropout"], mean_streams["mean"])
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
    state, epoch: int, val_ds, configs, rng_streams, metric_writer, ckpt_manager, mean_streams
):
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    data_loader = batch_sampler(
        val_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch in data_loader:
        state, batch_metrics = validation_step(state, batch, rng_streams["dropout"], mean_streams["mean"])
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
    static_broadcasted_argnums=(3,4)  # Add static arg
)
# @jax.jit
def train_step_pmap_arithmetic(
    state: TrainState,
    batch,
    dropout_key,
    num_devices: int,  # Passed explicitly
    loss_fn: str = "CorrGamma"  # Default loss function
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
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        local_loss = jnp.mean(loss_array)
        
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
        
        loss = jnp.mean( (pred_spectra - batch["spectra"]) ** 2 )
        return loss
    
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
    static_broadcasted_argnums=(3,4)  # Add static arg
)
# @jax.jit
def validation_step_pmap_arithmetic(
    state: TrainState, 
    batch, 
    dropout_key,
    num_devices: int,  # Passed explicitly
    loss_fn: str = "CorrGamma"  # Default loss function
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
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        local_loss = jnp.mean(loss_array)
        
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
        
        loss = jnp.mean( (pred_spectra - batch["spectra"]) ** 2 )
        return loss
    
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
    static_broadcasted_argnums=(3,)  # Add static arg
)
# @jax.jit
def train_step_pmap_geometric(
    state: TrainState,
    batch,
    dropout_key,
    num_devices: int  # Passed explicitly
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
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        local_loss = my_geometric_mean(loss_array)
        
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
    static_broadcasted_argnums=(3,)  # Add static arg
)
# @jax.jit
def validation_step_pmap_geometric(
    state: TrainState, 
    batch, 
    dropout_key,
    num_devices: int  # Passed explicitly
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
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        local_loss = my_geometric_mean(loss_array)
        
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
    mean_streams
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
    
    if configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item()
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        configs.masked_interval_starts[1] = window_end
        configs.masked_interval_ends[0] = window_start
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )

    data_loader = batch_sampler(
        train_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )

    metrics_list = []

    for batch in data_loader:
        # 1) Shard the batch so each device gets a sub-batch
        batch_sharded = shard_batch(batch)
        
        # 2) Create a dropout key for each device
        dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
        dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]  # Convert to list
        dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)
        
        loss_fn=configs.loss_fn
        # 3) Run pmapped train step
        state, batch_metrics = train_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded,
            num_devices,
            loss_fn
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
    mean_streams
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
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )

    data_loader = batch_sampler(
        val_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )

    metrics_list = []

    for batch in data_loader:
        # 1) Shard the batch so each device gets a sub-batch
        batch_sharded = shard_batch(batch)

        # 2) Create a dropout key for each device
        dropout_device_keys = jax.random.split(rng_streams["dropout"], num_devices)
        dropout_device_keys = [dropout_device_keys[i] for i in range(num_devices)]  # Convert to list
        dropout_sharded = jax.device_put_sharded(dropout_device_keys, devices)

        loss_fn=configs.loss_fn
        # 3) Run pmapped train step
        _, batch_metrics = validation_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded,
            num_devices,
            loss_fn
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

