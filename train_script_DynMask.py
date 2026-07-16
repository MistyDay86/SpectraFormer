"""SpectraFormer GammaNLL training script."""

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


logger.remove()
logger.add(
    sys.stderr,
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
)


@dataclass
class TrainArgs:
    model_tag: str = "min70_highf_filelevel_gammanll"
    material: str = "SiC-high-f"
    regime: Literal["single-gpu", "multi-gpu"] = "single-gpu"
    debug_nans: bool = True
    debug_logging: bool = False
    debug_compile_logging: bool = False


def main(args: TrainArgs) -> None:
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
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG" if args.debug_logging else "INFO",
    )

    Path("temp").mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.add(f"temp/{args.model_tag}_{timestamp}.log")

    if args.debug_compile_logging:
        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING, force=True)
        logging.getLogger("jax").setLevel(logging.WARNING)
        logging.getLogger("jaxlib").setLevel(logging.WARNING)

    import jax
    import jax.numpy as jnp
    import ml_confs
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    from flax.training.train_state import TrainState
    from tensorboardX import SummaryWriter

    from spectraformer.input_pipeline import batch_sampler, dataset_loader
    from spectraformer.inference import plot_loss, plot_results_train
    from spectraformer.model import CustomTrainState, SpectraFormer
    from spectraformer.train_DynMask import (
        _apply_mask_to_batch,
        _build_dynamic_mask_windows_np,
        train_step,
        validation_step,
    )

    jax.config.update("jax_debug_nans", args.debug_nans)

    maindir = Path(__file__).parent.resolve()
    logdir = maindir / "logs"
    ckptdir = maindir / "checkpoints"
    datadir = maindir / "data"
    configsdir = maindir / "configs"
    parsed_datadir = datadir / "parsed_data_spatial"
    material_dir = parsed_datadir / args.material

    logdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)

    config_file_path = configsdir / f"configs_{args.model_tag}.yaml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()

    devices = jax.devices()
    logger.info(f"JAX devices: {devices} ({len(devices)} total)")

    nc_files = sorted(material_dir.rglob("*.nc"))
    if not nc_files:
        raise ValueError(f"No .nc files found in {material_dir}")

    file_validation_fraction = float(getattr(configs, "file_validation_fraction", 0.20))
    file_split_seed = int(getattr(configs, "file_split_seed", getattr(configs, "root_rng_seed", 0)))

    rng = np.random.default_rng(file_split_seed)
    indices = rng.permutation(len(nc_files))
    n_val = max(1, int(round(len(nc_files) * file_validation_fraction)))
    n_val = min(n_val, len(nc_files) - 1)
    val_idx = set(indices[:n_val].tolist())

    train_files = [f for i, f in enumerate(nc_files) if i not in val_idx]
    val_files = [f for i, f in enumerate(nc_files) if i in val_idx]

    logger.info(f"Found {len(nc_files)} maps in {material_dir}")
    logger.info(f"Train maps: {len(train_files)}")
    logger.info(f"Validation maps: {len(val_files)}")
    logger.info("Validation map list:")
    for f in val_files:
        logger.info(f"  - {f.relative_to(material_dir)}")

    is_filter = bool(getattr(configs, "is_filter", False))
    filter_options = [False, True] if is_filter else [False]

    def load_full_map(nc_file: Path, use_filter: bool, split_fraction: float):
        relative_path = nc_file.relative_to(parsed_datadir)
        train_ds, val_ds = dataset_loader(
            datadir=parsed_datadir,
            file_location_with_name=str(relative_path),
            shuffle_rng_seed=getattr(configs, "root_rng_seed", 0),
            split_fraction=split_fraction,
            is_filter=use_filter,
            option="whitaker_hayes",
        )
        return train_ds if split_fraction == 1.0 else val_ds

    train_parts = []
    val_parts = []

    for nc_file in train_files:
        for use_filter in filter_options:
            ds = load_full_map(nc_file, use_filter, 1.0)
            if ds.sizes["spectra"] >= configs.batch_size:
                train_parts.append((nc_file.name, use_filter, ds))
            else:
                logger.warning(f"Skipping train map {nc_file.name}: {ds.sizes['spectra']} spectra")

    for nc_file in val_files:
        for use_filter in filter_options:
            ds = load_full_map(nc_file, use_filter, 0.0)
            if ds.sizes["spectra"] >= configs.batch_size:
                val_parts.append((nc_file.name, use_filter, ds))
            else:
                logger.warning(f"Skipping validation map {nc_file.name}: {ds.sizes['spectra']} spectra")

    if not train_parts:
        raise ValueError("No train map has enough spectra for the current batch size.")
    if not val_parts:
        raise ValueError("No validation map has enough spectra for the current batch size.")

    train_spectra = sum(ds.sizes["spectra"] for _, _, ds in train_parts)
    val_spectra = sum(ds.sizes["spectra"] for _, _, ds in val_parts)
    logger.info(f"Usable train maps: {len(train_parts)}")
    logger.info(f"Usable validation maps: {len(val_parts)}")
    logger.info(f"Train spectra: {train_spectra}")
    logger.info(f"Validation spectra: {val_spectra}")

    mask_windows_static = list(zip(configs.masked_interval_starts, configs.masked_interval_ends))
    mask_windows_for_loader = [] if getattr(configs, "dynamic_mask", False) else mask_windows_static

    dummy_ds = train_parts[0][2]
    dummy_example = next(batch_sampler(dummy_ds, mask_windows_for_loader, batch_size=1, shuffle=False))

    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
        dropout_rate=configs.dropout_rate,
    )

    root_key = jax.random.PRNGKey(seed=configs.root_rng_seed)
    main_key, params_key, dropout_key = jax.random.split(root_key, 3)
    window_key = main_key

    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=True,
    )

    learning_rate_decay = getattr(configs, "learning_rate_decay", "Constant")
    if learning_rate_decay == "Constant":
        tx = optax.adam(learning_rate=configs.learning_rate)
    elif learning_rate_decay == "Multiple cosine decay cycles":
        cosine_kwargs = []
        init_value = getattr(configs, "warmup_coeff", 0.1) * configs.learning_rate
        peak_value = configs.learning_rate
        warmup_steps = getattr(configs, "warmup_steps", 1000)
        decay_steps = getattr(configs, "decay_steps", 2000)
        decline_coeff = getattr(configs, "decline_coeff", 1)
        for _ in range(getattr(configs, "num_cycles", 20)):
            end_value = decline_coeff * init_value
            cosine_kwargs.append(
                {
                    "init_value": init_value,
                    "peak_value": peak_value,
                    "warmup_steps": warmup_steps,
                    "decay_steps": decay_steps,
                    "end_value": end_value,
                }
            )
            init_value = end_value
            peak_value *= decline_coeff
        tx = optax.adam(learning_rate=optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs))
    else:
        raise ValueError(f"Unsupported learning_rate_decay: {learning_rate_decay}")

    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )

    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=int(getattr(configs, "checkpoint_max_to_keep", 1)),
        enable_async_checkpointing=False,
    )
    ckpt_path = ckptdir / configs.tag
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ckpt_manager = ocp.CheckpointManager(
        ckpt_path,
        options=ckpt_options,
        metadata=configs.to_dict(),
    )

    if len(ckpt_manager.all_steps()) > 0:
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(),
            args=ocp.args.StandardRestore(state),
        )
        logger.info(f"Resuming from epoch {state.epoch}, step {state.step}")
    else:
        logger.info(f"No checkpoint found for '{configs.tag}', training from scratch")

    metric_writer = SummaryWriter(logdir / configs.tag)
    metric_writer.add_custom_scalars(
        {
            "my_layout": {
                "loss_step": ["Multiline", ["train/train_loss_step", "val/val_loss_step"]],
                "loss_epoch": ["Multiline", ["train/train_loss_epoch", "val/val_loss_epoch"]],
                "checkpoint_mse": ["Multiline", ["val/val_checkpoint_mse_epoch"]],
            }
        }
    )

    is_masked_loss = bool(getattr(configs, "is_masked_loss", True))
    best_val_mse = float("inf")
    restored_epoch = int(np.asarray(state.epoch))

    def apply_dynamic_mask(batch, wave_number_raw, rng_key, batch_index):
        if not getattr(configs, "dynamic_mask", False):
            return batch
        seed = int(jax.random.fold_in(rng_key, batch_index)[0])
        mask_rng = np.random.default_rng(seed)
        windows = _build_dynamic_mask_windows_np(
            wave_number_raw,
            mask_rng,
            chunk_size=getattr(configs, "mask_chunk_size", 200),
            min_chunks=getattr(configs, "mask_chunk_min", 2),
            max_chunks=getattr(configs, "mask_chunk_max", 6),
        )
        return _apply_mask_to_batch(
            batch,
            wave_number_raw,
            windows,
            default_mask_value=getattr(configs, "default_mask_value", -1),
        )

    def scalar(x):
        return float(np.asarray(jax.device_get(x)).reshape(-1)[0])

    for epoch in range(restored_epoch + 1, restored_epoch + configs.num_epochs + 1):
        epoch_start = time.perf_counter()
        window_key = jax.random.fold_in(window_key, epoch)

        train_losses = []
        train_batch_index = 0

        for name, use_filter, train_ds in train_parts:
            for batch in batch_sampler(
                train_ds,
                mask_windows_for_loader,
                batch_size=configs.batch_size,
                rng_seed=epoch,
                shuffle=True,
                drop_last=True,
                default_mask_value=getattr(configs, "default_mask_value", -1),
            ):
                batch = apply_dynamic_mask(
                    batch,
                    jnp.asarray(train_ds["wave_number"].values),
                    window_key,
                    train_batch_index,
                )
                state, metrics = train_step(
                    state,
                    batch,
                    dropout_key,
                    "Not specified",
                    is_masked_loss=is_masked_loss,
                )
                loss_value = scalar(metrics["train_loss"])
                train_losses.append(loss_value)
                metric_writer.add_scalar("train/train_loss_step", loss_value, int(np.asarray(state.step)))
                train_batch_index += 1

        val_losses = []
        val_mses = []
        val_batch_index = 0

        for name, use_filter, val_ds in val_parts:
            for batch in batch_sampler(
                val_ds,
                mask_windows_for_loader,
                batch_size=configs.batch_size,
                rng_seed=epoch,
                shuffle=False,
                drop_last=True,
                default_mask_value=getattr(configs, "default_mask_value", -1),
            ):
                batch = apply_dynamic_mask(
                    batch,
                    jnp.asarray(val_ds["wave_number"].values),
                    jax.random.PRNGKey(configs.root_rng_seed + epoch),
                    val_batch_index,
                )
                state, metrics = validation_step(
                    state,
                    batch,
                    dropout_key,
                    "Not specified",
                    is_masked_loss=is_masked_loss,
                )
                val_loss = scalar(metrics["val_gamma_nll_loss"])
                val_mse = scalar(metrics["MSE"])
                val_losses.append(val_loss)
                val_mses.append(val_mse)
                metric_writer.add_scalar("val/val_loss_step", val_loss, int(np.asarray(state.step)))
                metric_writer.add_scalar("val/val_checkpoint_mse_step", val_mse, int(np.asarray(state.step)))
                val_batch_index += 1

        if not train_losses or not val_losses:
            raise RuntimeError("No batches were processed in this epoch.")

        train_loss_epoch = float(np.nanmean(train_losses))
        val_loss_epoch = float(np.nanmean(val_losses))
        val_mse_epoch = float(np.nanmean(val_mses))

        metric_writer.add_scalar("train/train_loss_epoch", train_loss_epoch, epoch)
        metric_writer.add_scalar("val/val_loss_epoch", val_loss_epoch, epoch)
        metric_writer.add_scalar("val/val_checkpoint_mse_epoch", val_mse_epoch, epoch)

        logger.info(
            f"Epoch {epoch} -- train GammaNLL {train_loss_epoch:.3e} -- "
            f"val GammaNLL {val_loss_epoch:.3e} -- val MSE {val_mse_epoch:.3e}"
        )

        if val_mse_epoch < best_val_mse:
            old_best = best_val_mse
            best_val_mse = val_mse_epoch
            state = state.replace(epoch=jnp.array(epoch, dtype=jnp.int32))
            step_value = int(np.asarray(state.step))
            ckpt_manager.save(step_value, args=ocp.args.StandardSave(state))
            logger.info(
                f"Saved best checkpoint at epoch {epoch}, step {step_value}: "
                f"MSE {val_mse_epoch:.3e} previous {old_best:.3e}"
            )

        state = state.replace(epoch=jnp.array(epoch, dtype=jnp.int32))
        metric_writer.flush()
        gc.collect()

        logger.info(f"Epoch time: {time.perf_counter() - epoch_start:.1f}s")

    final_batch = next(
        batch_sampler(
            val_parts[0][2],
            mask_windows_for_loader,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            default_mask_value=getattr(configs, "default_mask_value", -1),
        )
    )
    final_batch = apply_dynamic_mask(
        final_batch,
        jnp.asarray(val_parts[0][2]["wave_number"].values),
        jax.random.PRNGKey(configs.root_rng_seed),
        0,
    )

    res = plot_results_train(
        apply_fn=state.apply_fn,
        variables={"params": state.params},
        batch=final_batch,
        raman_shift=val_parts[0][2].wave_number.values,
    )
    fig = plot_loss(res)
    metric_writer.add_figure("final_loss_on_example", fig)
    metric_writer.close()
    ckpt_manager.close()


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
