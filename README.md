# 🌌 SkySimulation

**A Python toolkit for simulating CMB temperature and polarisation maps with standard and non-standard primordial power spectra**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![CAMB](https://img.shields.io/badge/Powered%20by-CAMB-orange)](https://camb.info/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13829665.svg)](https://doi.org/10.5281/zenodo.19445834)
[![arXiv](https://img.shields.io/badge/arXiv-2604.05290-red)](https://arxiv.org/abs/2604.05290)

</div>

---

## 📖 Overview

**SkySimulation** is a modular Python package for generating realistic CMB sky maps in temperature and polarisation (**T, Q, U**) for use in cosmological analyses. It supports both standard **ΛCDM** cosmologies and models with **oscillatory features in the primordial power spectrum**, enabling systematic studies of their imprints on CMB observables.

The package uses [CAMB](https://camb.info/) to compute angular power spectra, applies official **Planck galactic masks** and **Planck-derived noise uncertainties**, and estimates full noise covariance matrices via MCMC sampling.

This code was developed as part of the analysis pipeline for [*Explaining Neural Networks on the Sky: Machine Learning Interpretability for CMB Maps*](https://arxiv.org/abs/XXXX.XXXXX).

---

## ✨ Features

- 🔭 **Full-sky CMB map simulation** — Generate T, Q, and U maps from any input power spectrum using HEALPix pixelisation via CAMB
- 🌀 **ΛCDM & beyond** — Supports standard cosmological parameters as well as a non-standard oscillating primordial power spectrum of the form:

$$P_{\mathcal{R}}(k)=P_{\mathcal{R}, 0}(k)\left[1+A_{\rm lin} \sin \left(\omega_{\rm lin} \frac{k}{k_{\star}}+\phi\right)\right]$$

- 🎛️ **Tunable feature parameters** — Freely adjust the **amplitude** ($A_{\rm lin}$) and **log-frequency** ($\omega_{\rm lin}$) of the primordial oscillatory feature. While also being able to explore other cosmological parameters ($\omega_{\rm cdm}, \omega_{\rm b}, A_{\rm s}, n_{\rm s}\$).
- 🗺️ **Planck galactic mask** — Applies the official Planck galactic mask to exclude contaminated sky regions, reproducing realistic sky coverage
- 🛰️ **Planck-like noise** — Adds realistic noise realisations calibrated on Planck instrument specifications
- 📊 **Noise covariance matrix** — Constructs the full pixel-pixel noise covariance matrix $C_{ij}$ by sampling Planck-derived asymmetric uncertainties via MCMC

---

## 📦 Installation

### Requirements

- Python ≥ 3.9
- [CAMB](https://camb.info/)
- [healpy](https://healpy.readthedocs.io/)
- numpy, scipy, matplotlib
- emcee (for covariance MCMC)

All dependencies are listed in `requirements.txt`.

### Install

```bash
git clone https://github.com/skyexplain/SkySimulation.git
cd SkySimulation
pip install -e .
```

---

## 🚀 Quick Start

See the `examples/` directory for ready-to-run scripts.
Also see **[Sandbox](https://github.com/skyexplain/Sandbox)** for Tutorials and more examples.

---

## 🔗 Related Packages

This repository is part of a three-package ecosystem:

| Package | Description |
|---|---|
| [**SkySimulation**](https://github.com/skyexplain/SkySimulation) | CMB map simulation (T, Q, U) with ΛCDM and oscillatory features |
| [**SkyNeuralNets**](https://github.com/skyexplain/SkyNeuralNets) | NN-based model selection on CMB maps *(this repo)* |
| [**SkyInterpret**](https://github.com/skyexplain/SkyInterpret) | Interpretability analysis of the trained neural networks |

---

## 🔗 External Data

This package relies on publicly available Planck data products:
- **Galactic mask**: Planck 2018 confidence masks ([Planck Legacy Archive](https://pla.esac.esa.int/))
- **Noise uncertainties**: Planck 2018 noise characterisation, used to build the asymmetric error distributions from ([Planck Legacy Archive](https://pla.esac.esa.int/)), sampled in the $C_{ij}$ estimation

---

## 📄 Citation

If you use **SkySimulation** in your research, please cite:

```bibtex
@article{Ocampo2026,
  author        = {Indira Ocampo and Guadalupe Cañas-Herrera},
  title         = {Explaining Neural Networks on the Sky: Machine Learning Interpretability for CMB Maps},
  journal       = {JCAP},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

---

## 📬 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/skyexplain/SkySimulation/issues) or contact [indira.ocampo@csic.es](mailto:indira.ocampo@csic.es).
