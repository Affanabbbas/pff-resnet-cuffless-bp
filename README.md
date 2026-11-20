# MIMIC-BP: Cuffless Blood Pressure Estimation with ResNet1D and PhysFusion

PyTorch code for subject-independent, calibration-free estimation of systolic (SBP) and diastolic (DBP) blood pressure in mmHg from ECG and PPG waveforms, using the [MIMIC-BP dataset](https://doi.org/10.7910/DVN/DBM1NF). It accompanies the paper *"Subject-Independent Cuffless Blood Pressure Estimation from ECG and PPG: A Multi-Seed Evaluation of Derivative Channels and Physiological Feature Fusion"*. The feature-fusion model is called PFF ResNet in the paper and PhysFusion in the code.

## Evaluation protocol

- **Inputs:** ECG and PPG, optionally with first and second PPG derivatives (dPPG, ddPPG). ABP is used only to derive SBP/DBP labels, never as an input.
- **Windows:** 30-second segments sampled at 125 Hz.
- **Subject-wise splits:** training, validation, and test sets share no subjects. Training stops if any overlap is detected.
- **Normalization:** global statistics are computed from training subjects only; per-segment z-scoring is also supported. Label standardization is applied only inside the training loss.
- **Metrics:** reported in mmHg after inverting label standardization: SBP/DBP MAE, RMSE, Pearson correlation, and subject-level MAE. The primary metric is the mean of SBP MAE and DBP MAE.

## Models

- **ResNet1D:** convolutional stem, residual temporal blocks, attention pooling, and an SBP/DBP regression head. It serves as the baseline.
- **Loss variants:** Huber, MSE, MAE, MAE + Pearson, and MAE + Pearson + standard-deviation matching. The last two reduce mean-regression behavior.
- **CNN-Mamba-Attention:** a state-space sequence-modeling variant.
- **PhysFusion:** the ResNet1D encoder plus a parallel branch of handcrafted descriptors computed before per-segment normalization: ECG and PPG amplitude statistics, mean absolute dPPG and ddPPG, ECG- and PPG-derived heart-rate proxies, an ECG-to-PPG delay (PTT-like) feature, and PPG peak count.

## Repository layout

```text
train.py                              # training entry point
evaluate.py                           # checkpoint evaluation and prediction export
train_with_computational_profiling.py # training with cost reporting
config*.yaml                          # experiment configurations
datasets/mimic_bp_dataset.py          # subject-wise dataset loader
models/                               # resnet1d.py, cnn_mamba_attention.py
utils/                                # leakage check, metrics, seeding, cost report
scripts/                              # dataset inspection, unzip, paper plots
outputs/final_results_table.md        # curated result summary
figures/paper_plots/                  # publication plots
METHODS_NOTES.md                      # detailed methodological notes
```

Checkpoints, raw prediction dumps, logs, and TIFF exports are excluded from Git by `.gitignore`.

## Setup

```bash
git clone https://github.com/Affanabbbas/mimic-bp-resnet-physfusion.git
cd mimic-bp-resnet-physfusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build for GPU training.

**Data.** The MIMIC-BP data are not included. Download them from the dataset page and follow its access terms, then arrange them as:

```text
mimic_bp/
├── ecg/  ppg/  labels/
├── train_subjects.txt
├── val_subjects.txt      # or validation_subjects.txt
└── test_subjects.txt
```

Check the layout with `python scripts/inspect_dataset.py --data_dir /path/to/mimic_bp`.

## Training and evaluation

```bash
python train.py --config config_resnet_mae_seed2026.yaml --data_dir /path/to/mimic_bp --device cuda

python evaluate.py \
  --checkpoint checkpoints/resnet_mae_seed2026/best_resnet1d.pt \
  --data_dir /path/to/mimic_bp --split test \
  --out_csv outputs/test_predictions.csv \
  --metrics_json outputs/test_metrics.json \
  --subject_csv outputs/test_subject_predictions.csv
```

Command-line options override the YAML config. Evaluation exports segment-level predictions, aggregate metrics, and optional subject-level averages.

## Results

Test-set errors in mmHg on unseen subjects (lower is better), from `outputs/final_results_table.md`, which lists all variants with full precision. The paper reports the multi-seed, group cross-validation, and bootstrap analyses.

| Method | SBP MAE | DBP MAE | Avg MAE | Subject Avg MAE |
|---|---:|---:|---:|---:|
| 3-model ensemble | 12.87 | 8.39 | 10.63 | 7.72 |
| ResNet1D + MAE | 12.99 | 8.60 | 10.79 | 7.82 |
| ResNet1D + MAE-Pearson (alpha = 0.05) | 13.05 | 8.58 | 10.82 | 7.69 |
| Original ResNet1D | 13.10 | 8.78 | 10.94 | 7.94 |
| CNN-Mamba-Attention | 13.27 | 8.61 | 10.94 | 7.99 |
| Small CNN-Mamba | 13.41 | 8.60 | 11.00 | 8.03 |

## Reproducing the paper

With the dataset and run outputs in place, these scripts regenerate the reported tables and analyses: `make_table6_quartiles.py`, `check_ensemble.py`, `print_paper_numbers.py`, `bootstrap_five_seed.py`, `run_ecg_ppg_ablation.sh`, and `descriptors_only_baseline.py`. Figures 9-11 need ensemble prediction files that are not distributed, so they cannot be regenerated from this repository alone.

## Limitations

This is research code, not a clinical device, and it makes no claim of clinical-grade accuracy. Results depend on the dataset, preprocessing, subject splits, and evaluation protocol. MIMIC-derived data may not represent all populations, subject-level generalization varies by split, and the handcrafted features depend on signal quality. External validation is required before broader conclusions.

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please cite the paper above and the [MIMIC-BP dataset](https://doi.org/10.7910/DVN/DBM1NF) according to its official instructions.
