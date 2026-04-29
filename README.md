# DeepHeliosat

Deep learning models for Global Horizontal Irradiance (GHI) estimation from GOES-16 satellite imagery.

## Overview

This repository contains the training code for a three-stream fusion CNN that estimates GHI from GOES-16 satellite image patches combined with tabular meteorological and geometric features.

Two model architectures are provided:

- **YuanModel** (`train_cnn_cropsize.py`) — CNN with residual blocks. Evaluated across four spatial footprints: 31 px (~50 km), 17 px (~27 km), 9 px (~14 km), 3 px (~5 km). Each crop size is trained over 5 independent runs.
- **FCN** (`train_fcn.py`) — Fully-connected baseline using the full 33×33 px patch (~50 km) with random rotation augmentation.

## Architecture

Each model fuses three streams:
- **Image stream**: GOES-16 channels (M6C01–M6C06, M6C13) + MODIS albedo (ws/bs) → 9 channels total, processed by residual blocks
- **Tabular stream**: 29 features (location, solar angles, clear-sky irradiances, channel calibration coefficients)
- **Cross stream**: concatenation of image and tabular embeddings

Targets: `GHI_corrected` and `kc_corrected` (clearness index), predicted jointly.

## Data

The dataset and results are hosted on Hugging Face: [vadimbec/DeepHeliosat](https://huggingface.co/datasets/vadimbec/DeepHeliosat)

- `FullDataset_all_corrected_roundagg.csv` — main tabular dataset (GOES-16 + BSRN/SOLRAD/NREL ground measurements, 2019–2022)
- `results/` — per-run model predictions on the test set

**Spatiotemporal split:**
- Train: all stations except test IDs, years ≤ 2021
- Test: `NREL_MIDC-UOSMRL`, `NREL_MIDC-UTPASRL`, `BSRN-LRC`, `NREL_MIDC-UAT`, `SOLRAD-BIS`, `SOLRAD-STE`, year 2022

## Requirements

```
torch
torchvision
pandas
numpy
scikit-learn
tqdm
```

## Usage

Update the dataset path in the training script, then:

```bash
python train_cnn_cropsize.py   # CNN across crop sizes
python train_fcn.py            # FCN baseline
```
