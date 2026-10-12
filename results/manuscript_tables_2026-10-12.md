# Manuscript tables regenerated 2026-10-12

NSTDB windows: clean control = 264, active noise = 264

## Table 1

| Model Architecture | AUROC [95% CI] | AP | Brier | Fixed τ = 0.50 [TP / FP / TN / FN] | Balanced Acc. [95% CI] | Sensitivity [95% CI] | False Reject Rate [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | 0.989 [0.964 – 0.999] | 0.975 | 0.058 | [228 / 5 / 745 / 69] | 88.1% [71.7 – 98.4%] | 76.8% [43.7 – 97.5%] | 0.7% [0.0 – 1.6%] |
| **2D CNN (Rendered Image)** | 0.989 [0.967 – 0.999] | 0.971 | 0.052 | [234 / 7 / 743 / 63] | 88.9% [75.5 – 99.5%] | 78.8% [51.8 – 100.0%] | 0.9% [0.3 – 1.7%] |
| **Rule-Based SQI Heuristic** | 0.977 [0.926 – 0.998] | 0.948 | 0.105 | [196 / 9 / 741 / 101] | 82.4% [57.2 – 96.1%] | 66.0% [15.5 – 92.9%] | 1.2% [0.3 – 2.4%] |
| **SinglePass Multimodal Net** | 0.967 [0.926 – 0.999] | 0.928 | 0.094 | [189 / 14 / 736 / 108] | 80.9% [63.4 – 98.5%] | 63.6% [28.1 – 98.9%] | 1.9% [0.7 – 3.3%] |
| **LightGBM (SQI)** | 0.964 [0.893 – 0.997] | 0.928 | 0.074 | [218 / 31 / 719 / 79] | 84.6% [64.8 – 96.1%] | 73.4% [33.9 – 94.6%] | 4.1% [1.6 – 7.3%] |
| **Independent SQI MLP** | 0.939 [0.834 – 0.990] | 0.880 | 0.088 | [231 / 65 / 685 / 66] | 84.6% [68.2 – 95.6%] | 77.8% [48.8 – 96.5%] | 8.7% [2.4 – 18.7%] |
| **Logistic Regression (SQI)** | 0.938 [0.838 – 0.994] | 0.888 | 0.084 | [227 / 47 / 703 / 70] | 85.1% [67.6 – 96.4%] | 76.4% [44.0 – 96.8%] | 6.3% [1.7 – 13.5%] |
| **LiquidAI d1-3B (Zero-Shot)** | 0.852 [0.717 – 0.944] | 0.740 | 0.154 | [158 / 30 / 720 / 139] | 74.6% [59.0 – 87.2%] | 53.2% [21.3 – 77.4%] | 4.0% [1.5 – 6.9%] |

## Table 1b

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 94.28% (FPR 4.67%) | 0.05569 | 0.93% (Sens 80.13%) | 0.48918 |
| **1D CNN (Raw Signal)** | 95.96% (FPR 3.47%) | 0.00794 | 1.33% (Sens 80.13%) | 0.33178 |
| **Rule-Based SQI Heuristic** | 84.51% (FPR 4.93%) | 0.37622 | 4.13% (Sens 80.47%) | 0.40701 |
| **SinglePass Multimodal Net** | 69.02% (FPR 4.67%) | 0.17462 | 6.53% (Sens 80.13%) | 0.10541 |
| **LightGBM (SQI)** | 80.47% (FPR 4.80%) | 0.36019 | 4.80% (Sens 80.47%) | 0.36019 |
| **Independent SQI MLP** | 70.03% (FPR 4.93%) | 0.72503 | 9.33% (Sens 80.47%) | 0.43368 |
| **Logistic Regression (SQI)** | 74.07% (FPR 4.80%) | 0.56444 | 8.27% (Sens 80.13%) | 0.39292 |
| **LiquidAI d1-3B (Zero-Shot)** | 53.20% (FPR 4.00%) | 0.50000 | 32.67% (Sens 80.13%) | 0.37754 |

## Table 2

| Model Architecture | Transfer AUROC | Transfer AP | Brier Score | Fixed τ = 0.50 [TP / FP / TN / FN] | Sensitivity | False Reject Rate | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.746 | 0.748 | 0.356 | [13 / 3 / 72 / 62] | 17.3% | 4.0% | 56.7% |
| **LightGBM (SQI)** | 0.731 | 0.792 | 0.320 | [7 / 1 / 74 / 68] | 9.3% | 1.3% | 54.0% |
| **1D CNN (Raw Signal)** | 0.726 | 0.683 | 0.266 | [46 / 11 / 64 / 29] | 61.3% | 14.7% | 73.3% |
| **Rule-Based SQI Heuristic** | 0.701 | 0.761 | 0.238 | [72 / 63 / 12 / 3] | 96.0% | 84.0% | 56.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.711 | 0.389 | [75 / 75 / 0 / 0] | 100.0% | 100.0% | 50.0% |
| **SinglePass Multimodal Net** | 0.627 | 0.627 | 0.373 | [19 / 0 / 75 / 56] | 25.3% | 0.0% | 62.7% |
| **Independent SQI MLP** | 0.617 | 0.627 | 0.500 | [0 / 0 / 75 / 75] | 0.0% | 0.0% | 50.0% |
| **Logistic Regression (SQI)** | 0.383 | 0.451 | 0.627 | [56 / 75 / 0 / 19] | 74.7% | 100.0% | 37.3% |

## Table 2b

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 17.33% (FPR 4.00%) | 0.57338 | 46.67% (Sens 81.33%) | 1.74e-04 |
| **1D CNN (Raw Signal)** | 9.33% (FPR 4.00%) | 0.99977 | 37.33% (Sens 80.00%) | 4.44e-04 |
| **Rule-Based SQI Heuristic** | 32.00% (FPR 4.00%) | 0.71544 | 68.00% (Sens 81.33%) | 0.53974 |
| **SinglePass Multimodal Net** | 25.33% (FPR 0.00%) | 1.00000 | trivial (all predicted unusable) | 0.00000 |
| **LightGBM (SQI)** | 41.33% (FPR 1.33%) | 0.36021 | 70.67% (Sens 84.00%) | 0.01955 |
| **Independent SQI MLP** | 25.33% (FPR 0.00%) | 2.98e-34 | trivial (all predicted unusable) | 0.00000 |
| **Logistic Regression (SQI)** | not attainable (scores saturate) | — | trivial (all predicted unusable) | 0.00000 |
| **LiquidAI d1-3B (Zero-Shot)** | 30.67% (FPR 2.67%) | 0.95532 | 73.33% (Sens 80.00%) | 0.85195 |

## Table 2c

| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |
| :--- | :---: | :---: | :---: | :---: |
| All 13 features | 0.383 / 0.451 | 0.377 / 0.442 | 0.749 / 0.791 | 0.617 / 0.627 |
| Drop signal_std | 0.431 / 0.529 | 0.370 / 0.441 | 0.723 / 0.775 | 0.627 / 0.627 |
| Drop saturation, flatline, constants | 0.746 / 0.796 | 0.746 / 0.805 | 0.751 / 0.792 | 0.750 / 0.749 |
| Drop signal_std, saturation, flatline, constants | 0.713 / 0.770 | 0.721 / 0.781 | 0.723 / 0.774 | 0.729 / 0.762 |

## Table 3

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Control (Baseline)** | 0.942 ± 0.072 | 0.936 ± 0.093 | 0.697 ± 0.390 | 0.639 ± 0.375 | 0.786 ± 0.302 | 0.729 ± 0.020 |
| **+24 dB (Low Noise)** | 0.899 ± 0.136 | 0.867 ± 0.217 | 0.787 ± 0.314 | 0.575 ± 0.390 | 0.763 ± 0.311 | 0.717 ± 0.026 |
| **+18 dB** | 0.812 ± 0.159 | 0.538 ± 0.334 | 0.742 ± 0.311 | 0.367 ± 0.390 | 0.391 ± 0.409 | 0.695 ± 0.025 |
| **+12 dB** | 0.500 ± 0.226 | 0.163 ± 0.169 | 0.521 ± 0.349 | 0.083 ± 0.181 | 0.082 ± 0.187 | 0.653 ± 0.038 |
| **+6 dB** | 0.176 ± 0.143 | 0.074 ± 0.051 | 0.231 ± 0.307 | 0.019 ± 0.078 | 0.017 ± 0.057 | 0.606 ± 0.044 |
| **0 dB (Equal Power)** | 0.037 ± 0.046 | 0.072 ± 0.036 | 0.105 ± 0.188 | 0.017 ± 0.064 | 0.006 ± 0.018 | 0.550 ± 0.063 |
| **-6 dB (Severe Noise)** | 0.011 ± 0.013 | 0.081 ± 0.060 | 0.075 ± 0.139 | 0.115 ± 0.109 | 0.053 ± 0.066 | 0.561 ± 0.072 |

## Table 4

| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Preproc Share (Median of Ratios) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based SQI Heuristic** | 0 | 0.00 MiB | CPU | 2.79 ms (3.28 ms) | 0.23 ms (0.25 ms) | 3.03 ms (3.51 ms) | 92.3% |
| **Logistic Regression (SQI)** | 14 | 0.01 MiB | CPU | 1.40 ms (1.52 ms) | 0.19 ms (0.25 ms) | 1.59 ms (1.74 ms) | 87.9% |
| **LightGBM (SQI)** | 1,092 | 0.14 MiB | CPU | 1.45 ms (1.78 ms) | 0.22 ms (0.35 ms) | 1.68 ms (2.03 ms) | 87.4% |
| **Independent SQI MLP** | 3,201 | 0.01 MiB | Apple M2 (MPS) | 1.44 ms (1.87 ms) | 1.18 ms (1.51 ms) | 2.64 ms (3.22 ms) | 55.7% |
| **1D CNN (Raw Signal)** | 167,489 | 0.64 MiB | Apple M2 (MPS) | 0.00 ms (0.00 ms) | 1.50 ms (2.10 ms) | 1.50 ms (2.10 ms) | 0.0% |
| **2D CNN (Rendered Image)** | 245,697 | 0.94 MiB | Apple M2 (MPS) | 66.49 ms (152.10 ms) | 4.86 ms (7.31 ms) | 71.43 ms (156.88 ms) | 93.2% |
| **SinglePass Multimodal Net** | 131,969 | 0.47 MiB | Apple M2 (MPS) | 72.15 ms (153.65 ms) | 5.38 ms (6.04 ms) | 77.23 ms (159.15 ms) | 93.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 3,120,000,000 | 5958.00 MiB | Apple M2 (MPS) | 70.72 ms (201.70 ms) | 845.10 ms (857.15 ms) | 916.30 ms (1048.48 ms) | 7.7% |

## Table 5

| Model | Metric | Seed 42 (reported) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD (seeds 1–4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | BUT QDB pooled AUROC | 0.989 | 0.971 | 0.979 | 0.979 | 0.977 | 0.976 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC AUROC | 0.726 | 0.715 | 0.713 | 0.719 | 0.710 | 0.714 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC sensitivity at τ = 0.50 | 61.3% | 62.7% | 61.3% | 61.3% | 62.7% | 62.0 ± 0.8% |
| **1D CNN (Raw Signal)** | CinC false reject at τ = 0.50 | 14.7% | 20.0% | 20.0% | 16.0% | 21.3% | 19.3 ± 2.3% |
| **2D CNN (Rendered Image)** | BUT QDB pooled AUROC | 0.989 | 0.984 | 0.983 | 0.984 | 0.993 | 0.986 ± 0.005 |
| **2D CNN (Rendered Image)** | CinC AUROC | 0.746 | 0.762 | 0.763 | 0.766 | 0.747 | 0.760 ± 0.008 |
| **2D CNN (Rendered Image)** | CinC sensitivity at τ = 0.50 | 17.3% | 41.3% | 41.3% | 40.0% | 42.7% | 41.3 ± 1.1% |
| **2D CNN (Rendered Image)** | CinC false reject at τ = 0.50 | 4.0% | 4.0% | 2.7% | 2.7% | 4.0% | 3.3 ± 0.8% |
| **SinglePass Multimodal Net** | BUT QDB pooled AUROC | 0.967 | 0.967 | 0.975 | 0.967 | 0.967 | 0.969 ± 0.004 |
| **SinglePass Multimodal Net** | CinC AUROC | 0.627 | 0.500 | 0.500 | 0.500 | 0.627 | 0.532 ± 0.063 |
| **SinglePass Multimodal Net** | CinC sensitivity at τ = 0.50 | 25.3% | 100.0% | 100.0% | 100.0% | 25.3% | 81.3 ± 37.3% |
| **SinglePass Multimodal Net** | CinC false reject at τ = 0.50 | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 75.0 ± 50.0% |

