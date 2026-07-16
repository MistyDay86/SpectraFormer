"""
SpectraFormer: Transformer-based Raman spectra unmixing.

This package provides a transformer model for unmixing Raman spectra,
particularly for graphene buffer layers on SiC substrates.

Example usage (CLI):
    spectraformer-unmix --checkpoint checkpoints/model --input data.nc --output unmixed.nc

Example usage (Python):
    from spectraformer.model import SpectraFormer
    from spectraformer.inference import predict
"""

__version__ = "1.0.0"

from spectraformer.model import SpectraFormer, CustomTrainState
from spectraformer.inference import predict
from spectraformer.input_pipeline import preprocess_dataset, batch_sampler

__all__ = [
    "SpectraFormer",
    "CustomTrainState",
    "predict",
    "preprocess_dataset",
    "batch_sampler",
]
