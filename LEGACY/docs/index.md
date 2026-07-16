---
layout: home

hero:
  name: "SpectraFormer"
  text: "Raman Spectra Unmixing"
  tagline: Transformer-based Machine Learning model for graphene buffer layer on SiC substrate
  actions:
    - theme: brand
      text: 🚀 Get Started
      link: /installation
    - theme: alt
      text: 📦 GitHub
      link: https://github.com/pietronvll/SpectraFormer
    - theme: alt
      text: 📄 arXiv Paper
      link: https://arxiv.org/abs/2601.04445
---

# What is SpectraFormer?

SpectraFormer is a transformer-based model for **spectral unmixing** of Raman data from epitaxial graphene on SiC. It learns to predict and subtract the SiC substrate contribution, isolating the buffer layer signal.

![Pipeline](/pipeline_scheme_v6_lowres.png)

The model masks the SiC spectral region, predicts the missing signal using self-attention, and subtracts it from the original spectrum to extract the graphene/buffer layer contribution.

![Results](/fig1_lowres.png)

Ready to try it? Check out the [Installation guide](/installation).
