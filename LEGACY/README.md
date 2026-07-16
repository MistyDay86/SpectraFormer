# SpectaFormer

SpectraFormer is a transformer-based Machine Learning model aimed for Raman spectra unmixing for graphene buffer layer on SiC substrate.

See more: [arXiv](https://arxiv.org/abs/2601.04445) paper

## Installation

```bash
git clone --depth 1 https://github.com/pietronvll/SpectraFormer.git
cd SpectraFormer
uv sync  # or: pip install -e .
```

## Usage

Check out the [guide](https://pietronvll.github.io/SpectraFormer/).

## Tips

### Tensorboard

To use tensorboard, type in terminal

```console
tensorboard --logdir=logs --samples_per_plugin images=1000
```

### Streamlit dashboard app

To use dashboard app, type in terminal

```console
streamlit run dashboard.py
```

### GPU usage in terminal

Useful command (especially during training) to watch gpu load in real time:

```console
watch -n 1 nvidia-smi
```

where 1 is update time in seconds.
