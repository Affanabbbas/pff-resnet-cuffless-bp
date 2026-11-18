# MIMIC-BP ResNet1D and PhysFusion Blood Pressure Estimation

This repository contains a PyTorch research prototype for cuffless, calibration-free blood pressure estimation from ECG and PPG waveforms using the MIMIC-BP dataset. The codebase started as a controlled ResNet1D baseline and was extended with derivative channels, subject-wise leakage checks, alternative regression losses, CNN-Mamba variants, physiological feature fusion, multi-seed evaluation, group cross-validation, bootstrap analysis, and ensemble comparisons.

The main objective is to estimate systolic blood pressure (SBP) and diastolic blood pressure (DBP) in mmHg for unseen subjects. The project emphasizes reproducibility, subject-independent evaluation, and transparent reporting over single-run leaderboard-style claims.

## Project Overview

Cuffless blood pressure estimation is a difficult physiological time-series regression task. Models can appear strong when train, validation, and test splits share subjects or when preprocessing statistics leak information across splits. This repository therefore treats the data protocol as a first-class part of the experiment:

- ECG and PPG waveforms are used as model inputs.
- ABP is used only to derive SBP/DBP labels, not as an input signal.
- Splits are subject-wise, and overlap checks run before training.
- Normalization statistics are computed from training subjects only when global normalization is used.
- Evaluation metrics are reported after converting predictions back to real mmHg units.
- Segment-level and subject-level summaries are both supported.

## Repository Structure

```text
.
├── train.py                         # Main training entry point
├── evaluate.py                      # Checkpoint evaluation and prediction export
├── train_with_computational_profiling.py
├── config*.yaml                     # Experiment configurations
├── datasets/
│   └── mimic_bp_dataset.py          # Subject-wise MIMIC-BP dataset loader
├── models/
│   ├── resnet1d.py                  # ResNet1D, dual-head, and PhysFusion variants
│   └── cnn_mamba_attention.py       # CNN-Mamba-Attention model
├── utils/
│   ├── leakage_check.py             # Split overlap checks
│   ├── metrics.py                   # SBP/DBP/average MAE, RMSE, Pearson metrics
│   ├── reproducibility.py           # Deterministic seeding helpers
│   └── computational_report.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── unzip_mimic_bp.py
│   └── make_paper_plots.py
├── outputs/
│   └── final_results_table.md       # Curated result summary
├── figures/
│   └── paper_plots/                 # Curated publication-style plots
└── METHODS_NOTES.md                 # Detailed methodological notes
```

Large files such as checkpoints, raw prediction dumps, logs, cached Python files, TIFF exports, and sharing bundles are intentionally excluded from Git by `.gitignore`.

## Dataset

This project uses the [MIMIC-BP dataset](https://doi.org/10.7910/DVN/DBM1NF). The data must be downloaded separately from the dataset repository and prepared according to its access and use terms; it is not included with this code. The expected local layout contains ECG, PPG, label files, and subject split files. The loader accepts several common capitalization and folder-name variants.

Expected layout:

```text
mimic_bp/
├── ecg/
├── ppg/
├── labels/
├── train_subjects.txt
├── val_subjects.txt or validation_subjects.txt
└── test_subjects.txt
```

Each subject contributes fixed-length waveform segments. The current protocol assumes 30-second windows sampled at 125 Hz.

Dataset inspection:

```bash
python scripts/inspect_dataset.py --data_dir /path/to/mimic_bp
```

If the dataset is archived locally:

```bash
python scripts/unzip_mimic_bp.py --data_dir /path/to/mimic_bp
```

## Experimental Protocol

The default task is subject-independent blood pressure estimation:

- Input: ECG and PPG waveforms, optionally augmented with dPPG and ddPPG.
- Output: SBP and DBP in mmHg.
- Split unit: subject, not waveform segment.
- Test condition: unseen subjects only.
- Primary metric: average of SBP MAE and DBP MAE.
- Additional metrics: SBP/DBP RMSE, SBP/DBP Pearson correlation, subject-level MAE.

The training script stops if subject leakage is detected:

```text
train vs val: overlap = 0
train vs test: overlap = 0
val vs test: overlap = 0
```

## Signal Processing

The dataset pipeline supports several preprocessing choices:

- NaN and infinity replacement
- spike winsorization
- optional ECG/PPG filtering
- optional quality filtering
- derivative-channel construction
- global train-only normalization
- per-segment z-score normalization
- optional label standardization during training

The strongest experiments in this repository use four signal channels:

```text
[ECG, PPG, dPPG, ddPPG]
```

Label standardization is applied only inside the training objective. Validation and test metrics are computed after inversion back to mmHg.

## Models

### ResNet1D Baseline

The baseline model uses a one-dimensional residual convolutional encoder for waveform regression. It includes:

- convolutional stem
- residual temporal blocks
- attention pooling
- final SBP/DBP regression head

This model is intentionally simple enough to function as a sanity-check baseline while still being competitive for waveform-based BP estimation.

### Loss Variants

Several regression objectives are implemented:

- Huber loss
- MSE loss
- MAE loss
- MAE + Pearson loss
- MAE + Pearson + standard deviation matching loss

The Pearson term was introduced to reduce mean-regression behavior, where predictions minimize absolute error but fail to preserve subject-to-subject or segment-to-segment ranking. The standard deviation term further encourages realistic prediction spread.

### CNN-Mamba-Attention

The repository includes a CNN-Mamba-Attention variant for temporal sequence modeling. This branch explores whether state-space-style sequence modeling can improve long-range waveform representation beyond local convolutional features.

### PhysFusion

PhysFusion extends the ResNet1D signal encoder with a parallel physiological feature branch. The model combines learned waveform embeddings with handcrafted physiological descriptors computed before per-segment normalization.

The current feature set includes:

- ECG mean, standard deviation, and peak-to-peak amplitude
- PPG mean, standard deviation, and peak-to-peak amplitude
- mean absolute dPPG
- mean absolute ddPPG
- ECG-derived heart-rate proxy
- PPG-derived heart-rate proxy
- ECG-to-PPG delay / PTT-like feature
- PPG peak count

These features are not intended as a substitute for waveform learning. They provide auxiliary physiological summaries that may help the model represent amplitude, morphology, and timing information.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mimic-bp-resnet-physfusion.git
cd mimic-bp-resnet-physfusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if GPU training is required.

## Training

Train using a YAML config:

```bash
python train.py --config config_resnet_mae_seed2026.yaml
```

Override key options from the command line:

```bash
python train.py \
  --config config_resnet_mae_seed2026.yaml \
  --data_dir /path/to/mimic_bp \
  --batch_size 64 \
  --epochs 50 \
  --device cuda
```

Quick debugging run:

```bash
python train.py \
  --config config_resnet_mae_seed2026.yaml \
  --data_dir /path/to/mimic_bp \
  --batch_size 16 \
  --epochs 2 \
  --num_workers 0 \
  --device cpu
```

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/resnet_mae_seed2026/best_resnet1d.pt \
  --data_dir /path/to/mimic_bp \
  --split test \
  --out_csv outputs/test_predictions.csv \
  --metrics_json outputs/test_metrics.json \
  --subject_csv outputs/test_subject_predictions.csv
```

Evaluation exports:

- segment-level predictions
- aggregate metrics JSON
- optional subject-level averaged predictions

## Current Result Summary

The table below is copied from the curated local result summary in `outputs/final_results_table.md`. These values should be treated as project-internal experimental results until independently reproduced in a fresh environment.

| Rank | Method | Type | SBP MAE | DBP MAE | AVG MAE | SBP RMSE | DBP RMSE | SBP Pearson | DBP Pearson | Subject AVG MAE | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3-model ensemble | Ensemble | 12.8685 | 8.3914 | 10.6300 | 16.1725 | 10.8916 | 0.4005 | 0.4320 | 7.7151 | Best overall |
| 2 | 2-model ensemble | Ensemble | 12.8968 | 8.4207 | 10.6588 | 16.2505 | 10.9416 | 0.3910 | 0.4220 | n/a | Previous best ensemble |
| 3 | ResNet1D + MAE | Single model | 12.9860 | 8.6038 | 10.7949 | 16.4650 | 11.2163 | 0.3925 | 0.4058 | 7.8154 | Best single model |
| 4 | ResNet1D + MAE-Pearson alpha=0.05 | Single model | 13.0546 | 8.5813 | 10.8179 | 16.3614 | 11.0883 | 0.3841 | 0.4144 | 7.6906 | Best subject-level single model |
| 5 | ResNet1D + MAE-Pearson alpha=0.10 | Single model | 13.0856 | 8.7117 | 10.8987 | 16.5874 | 11.2965 | 0.3845 | 0.4052 | 7.9125 | Rejected |
| 6 | Original ResNet1D | Single model | 13.0979 | 8.7827 | 10.9403 | 16.4454 | 11.2582 | 0.3728 | 0.3697 | 7.9352 | Baseline |
| 7 | CNN-Mamba-Attention | Single model | 13.2746 | 8.6139 | 10.9442 | 16.7588 | 11.1086 | 0.3501 | 0.3889 | 7.9917 | Baseline |
| 8 | Small CNN-Mamba | Single model | 13.4054 | 8.6013 | 11.0034 | 16.8062 | 11.1137 | 0.3059 | 0.3784 | 8.0294 | Useful for ensemble |
| 9 | ResNet1D + Global Norm + Huber | Single model | 13.6063 | 8.9258 | 11.2661 | 17.0847 | 11.4180 | 0.2594 | 0.3056 | 8.1209 | Rejected |

## Reproducibility Notes

The repository includes:

- deterministic seeding helpers
- saved split metadata under `reproducibility/`
- train/evaluation terminal logs in local runs
- configuration files for multiple seeds
- group five-fold configurations
- bootstrap comparison artifacts
- computational profiling utilities

For a clean reproduction, use the checked-in configs and write new artifacts into `outputs/` and `checkpoints/`. Those directories are ignored by default because they can become very large.

## Reproducing the Paper

The following scripts reproduce the paper's tables and reported analyses when the required dataset and run outputs are available:

- `make_table6_quartiles.py`
- `check_ensemble.py`
- `print_paper_numbers.py`
- `bootstrap_five_seed.py`
- `run_ecg_ppg_ablation.sh`
- `descriptors_only_baseline.py`

Figures 9–11 require ensemble prediction files that are not distributed with this repository. Those figures cannot be regenerated from the repository contents alone.

## Suggested Research Questions

This codebase is organized around the following questions:

- Does derivative augmentation improve waveform-based BP estimation?
- Is MAE more stable than Huber or MSE for this dataset?
- Can Pearson-aware objectives reduce mean-regression?
- Do physiological summary features complement deep waveform features?
- Are improvements consistent across random seeds?
- Are improvements preserved under subject-level group cross-validation?
- Does ensembling improve segment-level and subject-level performance?

## Limitations

This is a research prototype, not a clinical device. The results depend on dataset quality, preprocessing choices, subject splits, and the exact evaluation protocol. The project does not claim clinical-grade performance or regulatory readiness. Any medical interpretation requires separate validation under appropriate clinical and ethical review.

Known limitations:

- MIMIC-derived waveform data may not represent all patient populations.
- Subject-level generalization is difficult and may vary by split.
- Ensemble performance can overstate deployability if individual model uncertainty is not considered.
- Handcrafted physiological features are approximate and depend on signal quality.
- Further external validation is required before drawing broad conclusions.

## GitHub Upload Checklist

Before pushing publicly:

1. Confirm that the MIMIC-BP license permits the way you describe and use the dataset.
2. Do not upload raw MIMIC-BP data.
3. Do not upload large checkpoint files unless you intentionally use Git LFS.
4. Keep only curated result summaries and lightweight figures in Git.
5. Review the included MIT license and confirm it suits your intended release.
6. Re-run one small train/evaluation command after cloning into a fresh environment.

Recommended first commit:

```bash
git init
git add .gitignore README.md METHODS_NOTES.md requirements.txt \
  train.py evaluate.py train_with_computational_profiling.py \
  datasets models utils scripts configs reproducibility \
  config*.yaml outputs/final_results_table.md outputs/final_results_table.csv \
  figures/paper_plots
git commit -m "Initial research code release"
```

Then create a repository on GitHub and push:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mimic-bp-resnet-physfusion.git
git push -u origin main
```

## Citation and Acknowledgment

If you use this repository, cite the original MIMIC-BP dataset source according to its official instructions. Also cite any model or library papers that are central to your final report if you publish results derived from this code.
