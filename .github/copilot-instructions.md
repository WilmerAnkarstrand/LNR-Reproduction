# LNR Codebase Guide

## Project Overview
LNR (Label Noise Rebalancing) is a PyTorch-based framework for long-tailed imbalanced learning. It implements a two-stage training process: Stage 1 uses mixup augmentation on imbalanced data, Stage 2 applies MiSLAS with LNR's label flipping to enhance minority classes.

## Architecture
- **Models**: Backbone (ResNet variants from `models/`) + Classifier + LearnableWeightScaling (LWS) for class-wise scaling
- **Data Flow**: Datasets load imbalanced data with `imb_factor` (e.g., 0.01 for extreme imbalance), compute `cls_num_list` for class frequencies
- **Training Stages**:
  - Stage 1: Mixup training (e.g., `train_stage1.py`)
  - Stage 2: Freeze backbone, train classifier + LWS with label flipping (e.g., `train_stage2_lnr.py`)

## Key Methods (`methods.py`)
- `LabelAwareSmoothing`: Applies smoothing based on class frequency (higher for rare classes)
- `mixup_data` + `mixup_criterion`: Data augmentation with label rebalancing logic (avoids mixing rare↔common if imbalance ratio >3)
- `LearnableWeightScaling`: Trainable per-class scaling for logits

## Configuration & Workflows
- **Configs**: YAML files in `config/` specify dataset, paths, resume checkpoints, hyperparameters (e.g., `smooth_head=0.4`, `lr_factor=0.2`)
- **Training Command**: `python train_stage2_lnr.py --cfg config/cifar100/cifar100_imb001_stage2_mislas.yaml`
- **Evaluation**: Tracks overall/top5 accuracy, head/medium/tail class accuracy, Expected Calibration Error (ECE)
- **Checkpoints**: Saved in `saved/modelname_date/ckps/` with `current.pth.tar` and `model_best.pth.tar`

## Conventions
- Use `cls_num_list` (from dataset) for frequency-aware operations
- Label flipping in Stage 2 uses prediction z-scores and class priors to flip majority samples to minority classes
- Distributed training supported via `config.distributed`
- Logging via `utils/logger.py`, metrics via `utils/metric.py`

## Adding Components
- **New Dataset**: Extend `datasets/base.py` pattern, add LT sampler in `datasets/sampler.py`, update config YAML
- **New Method**: Add to `methods.py`, integrate in training loop (e.g., modify `train_lnr` in `train_stage2_lnr.py`)
- **New Model**: Add to `models/`, update config `backbone` field

## Dependencies
- PyTorch (distributed training)
- NumPy, YACS (config parsing)
- Standard CV libs (PIL, etc. for datasets)

Reference: `README.md` for setup, `config/cifar100/cifar100_imb001_stage2_mislas.yaml` for config example.