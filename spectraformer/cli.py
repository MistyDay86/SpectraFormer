"""Command line inference for the GammaNLL SpectraFormer pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass
class InferenceArgs:
    checkpoint: Path
    config: Path
    input: Path
    output: Path
    material: str = "SiC-high-f"
    tau: float = 0.0
    device: Literal["auto", "cpu", "gpu"] = "auto"


def main(args: InferenceArgs) -> None:
    import os

    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import ml_confs
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    import xarray as xr
    from scipy.stats import gamma

    from spectraformer.input_pipeline import batch_sampler, dataset_loader
    from spectraformer.model import CustomTrainState, SpectraFormer
    from spectraformer.train_DynMask import _apply_mask_to_batch, _build_dynamic_mask_windows_np

    configs = ml_confs.from_file(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.input.is_file():
        input_files = [args.input]
        input_root = args.input.parent
    else:
        input_files = sorted(args.input.rglob("*.nc"))
        input_root = args.input

    if not input_files:
        raise FileNotFoundError(f"No .nc files found in {args.input}")

    mask_windows_static = list(zip(configs.masked_interval_starts, configs.masked_interval_ends))
    mask_windows_for_loader = [] if getattr(configs, "dynamic_mask", False) else mask_windows_static

    first_file = input_files[0]
    first_ds, _ = dataset_loader(
        datadir=first_file.parent,
        file_location_with_name=first_file.name,
        shuffle_rng_seed=getattr(configs, "root_rng_seed", 0),
        split_fraction=1.0,
        is_filter=False,
        option="whitaker_hayes",
    )
    dummy = next(batch_sampler(first_ds, mask_windows_for_loader, batch_size=1, shuffle=False))

    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
        dropout_rate=configs.dropout_rate,
    )
    variables = model.init(
        jax.random.PRNGKey(configs.root_rng_seed),
        dummy["masked_spectra"][0],
        dummy["wave_number"],
        dummy["mask"],
        training=False,
    )

    tx = optax.adam(learning_rate=configs.learning_rate)
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )

    ckpt_manager = ocp.CheckpointManager(args.checkpoint)
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint}")

    try:
        restored = ckpt_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state, partial_restore=True),
        )
    except (TypeError, ValueError):
        restored = ckpt_manager.restore(latest_step)

    def _extract_params(obj):
        if hasattr(obj, "params"):
            return obj.params
        if isinstance(obj, dict):
            if "params" in obj:
                return obj["params"]
            if "default" in obj:
                return _extract_params(obj["default"])
        raise TypeError(f"Cannot extract params from checkpoint object of type {type(obj)}")

    state = state.replace(params=_extract_params(restored))

    def apply_dynamic_mask(batch, wave_number_raw, batch_index):
        if not getattr(configs, "dynamic_mask", False):
            return batch
        mask_rng = np.random.default_rng(int(configs.root_rng_seed) + int(batch_index))
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

    def as_np(x):
        return np.asarray(jax.device_get(x))

    for input_file in input_files:
        ds, _ = dataset_loader(
            datadir=input_file.parent,
            file_location_with_name=input_file.name,
            shuffle_rng_seed=getattr(configs, "root_rng_seed", 0),
            split_fraction=1.0,
            is_filter=False,
            option="whitaker_hayes",
        )

        spectra_list = []
        masked_list = []
        mask_list = []
        mu_list = []
        alpha_list = []
        batch_index = 0

        for batch in batch_sampler(
            ds,
            mask_windows_for_loader,
            batch_size=configs.batch_size,
            shuffle=False,
            drop_last=False,
            default_mask_value=getattr(configs, "default_mask_value", -1),
        ):
            batch = apply_dynamic_mask(
                batch,
                jnp.asarray(ds["wave_number"].values),
                batch_index,
            )
            pred_mu, pred_alpha = state.apply_fn(
                {"params": state.params},
                batch["masked_spectra"],
                batch["wave_number"],
                batch["mask"],
                training=False,
            )

            spectra_list.append(as_np(batch["spectra"]))
            masked_list.append(as_np(batch["masked_spectra"]))
            mask_list.append(as_np(batch["mask"]))
            mu_list.append(as_np(pred_mu))
            alpha_list.append(as_np(pred_alpha))
            batch_index += 1

        spectra = np.squeeze(np.concatenate(spectra_list, axis=0))
        masked_spectra = np.squeeze(np.concatenate(masked_list, axis=0))
        mask = np.squeeze(np.concatenate(mask_list, axis=0))
        mu = np.squeeze(np.concatenate(mu_list, axis=0))
        alpha = np.squeeze(np.concatenate(alpha_list, axis=0))

        predicted_spectra = mu
        predicted_difference = spectra - predicted_spectra

        if args.tau and args.tau > 0:
            mu_clip = np.clip(mu, 1e-6, None)
            alpha_clip = np.clip(alpha, 1e-6, None)
            q_tau = gamma.ppf(args.tau, a=alpha_clip, scale=mu_clip / alpha_clip)
            predicted_spectra_quantile = np.where(mask.astype(bool), q_tau, mu)
        else:
            predicted_spectra_quantile = mu

        predicted_difference_quantile = spectra - predicted_spectra_quantile

        wave_number = np.asarray(ds["wave_number"].values)
        if np.nanmax(wave_number) < 10:
            wave_number = wave_number * 800 + 2000

        out = xr.Dataset(
            data_vars={
                "spectra": (("sample", "wave_number"), spectra),
                "masked_spectra": (("sample", "wave_number"), masked_spectra),
                "mask": (("sample", "wave_number"), (np.asarray(mask).squeeze() if np.asarray(mask).squeeze().ndim == 2 else np.broadcast_to(np.asarray(mask).squeeze()[None, :], np.asarray(spectra).squeeze().shape))),
                "predicted_spectra": (("sample", "wave_number"), predicted_spectra),
                "predicted_mu": (("sample", "wave_number"), mu),
                "predicted_alpha": (("sample", "wave_number"), alpha),
                "predicted_difference": (("sample", "wave_number"), predicted_difference),
                "predicted_spectra_quantile": (("sample", "wave_number"), predicted_spectra_quantile),
                "predicted_difference_quantile": (("sample", "wave_number"), predicted_difference_quantile),
            },
            coords={
                "sample": np.arange(spectra.shape[0]),
                "wave_number": wave_number,
            },
            attrs={
                "checkpoint": str(args.checkpoint),
                "checkpoint_step": int(latest_step),
                "config": str(args.config),
                "tau": float(args.tau),
            },
        )

        relative = input_file.relative_to(input_root) if input_file.is_relative_to(input_root) else Path(input_file.name)
        output_file = args.output / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_file.with_suffix(".nc")
        out.to_netcdf(output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main(tyro.cli(InferenceArgs))
