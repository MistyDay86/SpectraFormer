"""
SpectraFormer training script.

Usage:
    python train_script.py --model-tag min70_highf --material SiC-high-f
    python train_script.py --model-tag min70_highf --material SiC-high-f --regime multi-gpu
"""

import sys
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


def main(args: TrainArgs) -> None:
    """Run training with the given arguments."""
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
    from spectraformer.train import (
        train_epoch, validation_epoch,
        train_epoch_pmap, validation_epoch_pmap,
        log_gpu_usage
    )
    from spectraformer.inference import plot_results_train, predict, plot_loss

    jax.config.update("jax_debug_nans", args.debug_nans)

    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"JAX devices: {devices} ({num_devices} total)")

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
    configs.tabulate()
    is_early_stop = True if not hasattr(configs, 'is_early_stop') else configs.is_early_stop # turning on early stopping process
    min_delta = 1e-4 if not hasattr(configs, 'early_stop_min_delta') else configs.early_stop_min_delta
    patience = 5 if not hasattr(configs, 'early_stop_patience') else configs.early_stop_patience
    
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
    
    # New automatic dataset loading
    datasets = []
    dataset_names = []
    
    num_spectra_train = 0
    num_spectra_train_accepted = 0
    num_spectra_val = 0
    num_spectra_val_accepted = 0
    
    # No filtering load
    for nc_file in nc_files:
        # Get relative path from datadir (e.g., "SiC/subdir/filename.nc")
        relative_path = nc_file.relative_to(parsed_datadir)
        
        train_ds, val_ds = dataset_loader(
            datadir=parsed_datadir,
            file_location_with_name=str(relative_path),
            shuffle_rng_seed=configs.root_rng_seed,
            is_filter=False,
            option='whitaker_hayes'
        )
        # Update the total number of spectra
        num_spectra_train += train_ds.sizes['spectra']
        num_spectra_val += val_ds.sizes['spectra']
        # Load only those, who is large enough to be treated in parallel
        if train_ds.sizes['spectra'] >= configs.batch_size and val_ds.sizes['spectra']>=configs.batch_size:
            datasets.append((train_ds, val_ds))
            dataset_names.append(nc_file.name)
            num_spectra_train_accepted += train_ds.sizes['spectra']
            num_spectra_val_accepted += val_ds.sizes['spectra']
    
    #  Filtering load - only if filtering is set (to not double-load same data)
    if hasattr(configs, 'is_filter') and configs.is_filter:
        for nc_file in nc_files:
            # Get relative path from datadir (e.g., "SiC/subdir/filename.nc")
            relative_path = nc_file.relative_to(parsed_datadir)
            
            train_ds, val_ds = dataset_loader(
                datadir=parsed_datadir,
                file_location_with_name=str(relative_path),
                shuffle_rng_seed=configs.root_rng_seed,
                is_filter=configs.is_filter if hasattr(configs, 'is_filter') else False,
                option='whitaker_hayes'
            )
            # Load only those, who is large enough to be treated in parallel
            if train_ds.sizes['spectra'] >= configs.batch_size and val_ds.sizes['spectra']>=configs.batch_size:
                datasets.append((train_ds, val_ds))

    logger.info(f"Loaded {len(datasets)}/{len(nc_files)} datasets from {material_dir}")
    logger.info(f"Total spectra: train={num_spectra_train}, val={num_spectra_val}, total={num_spectra_train + num_spectra_val}")
    logger.info(f"Accepted spectra: train={num_spectra_train_accepted}, val={num_spectra_val_accepted}, total={num_spectra_train_accepted + num_spectra_val_accepted}")

    if hasattr(configs, 'is_filter') and configs.is_filter:
        logger.info("Filtering enabled: doubling data")
    
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    dummy_example = next(batch_sampler(datasets[0][0], mask_windows, batch_size=1))
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
        max_to_keep=patience+1 # this is for having best model in case of training process only worsens the loss
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

    for epoch in range(restored_epoch + 1, restored_epoch + configs.num_epochs + 1):
        window_RNG_key = jax.random.split(window_RNG_key, num=1)[0]

        epoch_train_metrics = []
        epoch_val_metrics = []

        dataset_order = jax.random.permutation(window_RNG_key, len(datasets))
        
        for ds_idx in dataset_order:
            train_ds, val_ds = datasets[ds_idx]
            logger.debug(f"Dataset {ds_idx}: train={train_ds.shape[1]}, val={val_ds.shape[1]}")
            
            match training_regime:
                case "One device":
                    
                    # Training
                    state, train_metrics_ds = train_epoch(
                        state, epoch, train_ds, configs, rng_streams,
                        metric_writer, ckpt_manager, window_RNG_key, mean_streams
                    )
                    # Validation
                    state, val_metrics_ds = validation_epoch(
                        state, epoch, val_ds, configs, rng_streams, 
                        metric_writer, ckpt_manager, mean_streams
                    )
                case "All devices":
                    
                    state, train_metrics_ds = train_epoch_pmap(
                        state=state, epoch=epoch, train_ds=train_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager, 
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams
                        )
                    state, val_metrics_ds = validation_epoch_pmap(
                        state=state, epoch=epoch, val_ds=val_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager,
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams
                        )
                case _:
                    raise Exception(f"Specify training_regime correctly!")
            
            epoch_train_metrics.append(train_metrics_ds)
            epoch_val_metrics.append(val_metrics_ds)

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
            ckpt_manager.save(
                step=int(single_state.step),
                args=ocp.args.StandardSave(single_state),
            )

        early_stop = early_stop.update(val_metrics[-1]["val_loss_step"])
        best_metric_value = min(metric["val_loss_step"] for metric in val_metrics)
        metrics_diff = val_metrics[-1]["val_loss_step"] - best_metric_value
        if is_early_stop:
            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_metrics[-1]['train_loss_step']:.4e} | "
                f"val_loss={val_metrics[-1]['val_loss_step']:.4e} | "
                f"patience={early_stop.patience_count}/{patience}"
            )
        else:
            logger.info(
                f"Epoch {epoch} | "
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
