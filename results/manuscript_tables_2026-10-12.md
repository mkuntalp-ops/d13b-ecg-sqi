# Manuscript tables regenerated 2026-10-12

NSTDB windows: clean control = 44, active noise = 264

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
| **Rule-Based SQI Heuristic** | 0.700 | 0.759 | 0.239 | [72 / 63 / 12 / 3] | 96.0% | 84.0% | 56.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.706 | 0.388 | [75 / 75 / 0 / 0] | 100.0% | 100.0% | 50.0% |
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
| **LiquidAI d1-3B (Zero-Shot)** | 30.67% (FPR 4.00%) | 0.95257 | 73.33% (Sens 81.33%) | 0.84389 |

## Table 2c

| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |
| :--- | :---: | :---: | :---: | :---: |
| All 13 features | 0.383 / 0.451 | 0.377 / 0.442 | 0.749 / 0.791 | 0.617 / 0.627 |
| Drop signal_std | 0.431 / 0.529 | 0.370 / 0.441 | 0.723 / 0.775 | 0.627 / 0.627 |
| Drop saturation, flatline, constants | 0.746 / 0.796 | 0.746 / 0.805 | 0.751 / 0.792 | 0.750 / 0.749 |
| Drop signal_std, saturation, flatline, constants | 0.713 / 0.770 | 0.721 / 0.781 | 0.723 / 0.774 | 0.729 / 0.762 |

## Table 3

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline | d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean reference (MIT-BIH base)** | 0.967 ± 0.028 | 0.975 ± 0.082 | 0.788 ± 0.326 | 0.685 ± 0.358 | 0.857 ± 0.271 | 0.740 ± 0.011 | 0.777 ± 0.104 |
| **+24 dB (Low Noise)** | 0.900 ± 0.136 | 0.867 ± 0.217 | 0.787 ± 0.314 | 0.575 ± 0.390 | 0.764 ± 0.310 | 0.717 ± 0.026 | 0.759 ± 0.100 |
| **+18 dB** | 0.812 ± 0.159 | 0.538 ± 0.334 | 0.742 ± 0.311 | 0.367 ± 0.390 | 0.391 ± 0.409 | 0.695 ± 0.025 | 0.740 ± 0.074 |
| **+12 dB** | 0.501 ± 0.225 | 0.163 ± 0.169 | 0.521 ± 0.349 | 0.083 ± 0.181 | 0.082 ± 0.187 | 0.653 ± 0.038 | 0.704 ± 0.052 |
| **+6 dB** | 0.176 ± 0.143 | 0.074 ± 0.051 | 0.231 ± 0.307 | 0.019 ± 0.078 | 0.017 ± 0.057 | 0.606 ± 0.044 | 0.685 ± 0.049 |
| **0 dB (Equal Power)** | 0.038 ± 0.046 | 0.072 ± 0.036 | 0.105 ± 0.188 | 0.017 ± 0.064 | 0.006 ± 0.018 | 0.550 ± 0.063 | 0.685 ± 0.051 |
| **-6 dB (Severe Noise)** | 0.011 ± 0.013 | 0.081 ± 0.060 | 0.075 ± 0.139 | 0.012 ± 0.041 | 0.006 ± 0.020 | 0.555 ± 0.070 | 0.699 ± 0.059 |

## Table 3b

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | d1-3B (zero-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24 dB (Low Noise)** | -0.067 ± 0.117 | -0.107 ± 0.240 | -0.002 ± 0.073 | -0.110 ± 0.252 | -0.094 ± 0.183 | -0.018 ± 0.058 |
| **+18 dB** | -0.155 ± 0.141 | -0.436 ± 0.344 | -0.046 ± 0.159 | -0.318 ± 0.336 | -0.466 ± 0.402 | -0.037 ± 0.059 |
| **+12 dB** | -0.467 ± 0.214 | -0.811 ± 0.187 | -0.267 ± 0.364 | -0.602 ± 0.354 | -0.776 ± 0.307 | -0.072 ± 0.092 |
| **+6 dB** | -0.791 ± 0.139 | -0.901 ± 0.093 | -0.557 ± 0.417 | -0.666 ± 0.366 | -0.840 ± 0.276 | -0.092 ± 0.103 |
| **0 dB (Equal Power)** | -0.930 ± 0.055 | -0.902 ± 0.087 | -0.683 ± 0.373 | -0.668 ± 0.365 | -0.852 ± 0.271 | -0.092 ± 0.119 |
| **-6 dB (Severe Noise)** | -0.956 ± 0.032 | -0.893 ± 0.098 | -0.713 ± 0.356 | -0.673 ± 0.363 | -0.851 ± 0.273 | -0.078 ± 0.113 |

## Table 4

| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Total Median, Range over Runs | Preproc Share (Median of Ratios) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based SQI Heuristic** | 0 | 0.00 MiB | CPU | 2.58 ms (3.04 ms) | 0.22 ms (0.24 ms) | 2.81 ms (3.27 ms) | 2.60–2.83 ms | 92.3% |
| **Logistic Regression (SQI)** | 14 | 0.01 MiB | CPU | 1.34 ms (1.52 ms) | 0.18 ms (0.22 ms) | 1.52 ms (1.73 ms) | 1.51–1.54 ms | 88.4% |
| **LightGBM (SQI)** | 1,092 | 0.14 MiB | CPU | 1.32 ms (1.49 ms) | 0.17 ms (0.25 ms) | 1.48 ms (1.72 ms) | 1.48–1.55 ms | 88.4% |
| **Independent SQI MLP** | 3,201 | 0.01 MiB | Apple M4 (MPS) | 1.38 ms (1.76 ms) | 1.13 ms (1.77 ms) | 2.53 ms (3.62 ms) | 2.49–2.54 ms | 54.5% |
| **1D CNN (Raw Signal)** | 167,489 | 0.64 MiB | Apple M4 (MPS) | 0.00 ms (0.00 ms) | 1.47 ms (2.28 ms) | 1.47 ms (2.28 ms) | 1.36–1.49 ms | 0.0% |
| **2D CNN (Rendered Image)** | 245,697 | 0.94 MiB | Apple M4 (MPS) | 60.23 ms (137.53 ms) | 5.12 ms (5.63 ms) | 65.24 ms (142.53 ms) | 64.74–66.39 ms | 92.4% |
| **SinglePass Multimodal Net** | 131,969 | 0.47 MiB | Apple M4 (MPS) | 61.41 ms (128.75 ms) | 5.56 ms (6.10 ms) | 67.09 ms (134.26 ms) | 66.59–69.04 ms | 92.4% |
| **LiquidAI d1-3B (Zero-Shot)** | 3,120,000,000 | 5958.00 MiB | Apple M4 (MPS) | 63.37 ms (155.48 ms) | 828.47 ms (834.71 ms) | 891.19 ms (984.62 ms) | 891.06–893.27 ms | 7.1% |

## Table 5

| Model | Metric | Seed 42 (reported) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD (seeds 1–4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | BUT QDB pooled AUROC | 0.989 | 0.971 | 0.979 | 0.979 | 0.977 | 0.976 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC AUROC | 0.726 | 0.715 | 0.713 | 0.719 | 0.710 | 0.714 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC sensitivity at τ = 0.50 | 61.3% | 62.7% | 61.3% | 61.3% | 62.7% | 62.0 ± 0.8% |
| **1D CNN (Raw Signal)** | CinC false reject at τ = 0.50 | 14.7% | 20.0% | 20.0% | 16.0% | 21.3% | 19.3 ± 2.3% |
| **2D CNN (Rendered Image)** | BUT QDB pooled AUROC | 0.989 | 0.984 | 0.983 | 0.984 | 0.993 | 0.986 ± 0.005 |
| **2D CNN (Rendered Image)** | CinC AUROC | 0.746 | 0.763 | 0.763 | 0.767 | 0.748 | 0.760 ± 0.008 |
| **2D CNN (Rendered Image)** | CinC sensitivity at τ = 0.50 | 17.3% | 41.3% | 42.7% | 20.0% | 42.7% | 36.7 ± 11.1% |
| **2D CNN (Rendered Image)** | CinC false reject at τ = 0.50 | 4.0% | 4.0% | 2.7% | 2.7% | 4.0% | 3.3 ± 0.8% |
| **SinglePass Multimodal Net** | BUT QDB pooled AUROC | 0.967 | 0.967 | 0.975 | 0.967 | 0.967 | 0.969 ± 0.004 |
| **SinglePass Multimodal Net** | CinC AUROC | 0.627 | 0.500 | 0.500 | 0.500 | 0.627 | 0.532 ± 0.063 |
| **SinglePass Multimodal Net** | CinC sensitivity at τ = 0.50 | 25.3% | 100.0% | 100.0% | 100.0% | 25.3% | 81.3 ± 37.3% |
| **SinglePass Multimodal Net** | CinC false reject at τ = 0.50 | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 75.0 ± 50.0% |

## Table 1d

| Model | AUROC, class 3 unusable (297) [95% CI] | AUROC, classes 2–3 unusable (490), seed 42 [95% CI] | AUROC, classes 2–3 unusable, seeds 1–4 (range) | Sens. / FR at τ = 0.50, class 3 unusable | Sens. / FR at τ = 0.50, classes 2–3 unusable |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.938 [0.838 – 0.994] | 0.955 [0.920 – 0.976] | 0.955–0.955 | 76.4% / 6.3% | 85.3% / 8.6% |
| **LightGBM (SQI)** | 0.964 [0.893 – 0.997] | 0.951 [0.918 – 0.972] | 0.951–0.951 | 73.4% / 4.1% | 89.0% / 13.1% |
| **Independent SQI MLP** | 0.939 [0.834 – 0.990] | 0.837 [0.594 – 0.966] | 0.825–0.942 | 77.8% / 8.7% | 75.9% / 14.7% |
| **1D CNN (Raw Signal)** | 0.989 [0.964 – 0.999] | 0.961 [0.902 – 0.994] | 0.959–0.964 | 76.8% / 0.7% | 78.2% / 6.6% |
| **2D CNN (Rendered Image)** | 0.989 [0.967 – 0.999] | 0.984 [0.960 – 0.998] | 0.984–0.986 | 78.8% / 0.9% | 90.6% / 5.6% |
| **SinglePass Multimodal Net** | 0.967 [0.926 – 0.999] | 0.865 [0.611 – 0.993] | 0.867–0.984 | 63.6% / 1.9% | 78.6% / 6.5% |
| **Rule-Based SQI Heuristic** | 0.977 [0.926 – 0.998] | 0.973 [0.948 – 0.987] | — | 66.0% / 1.2% | 41.8% / 0.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.852 [0.717 – 0.944] | 0.785 [0.655 – 0.875] | — | 53.2% / 4.0% | 35.3% / 2.7% |

## Table 2d

| Model | AUROC: first-N (reported) | AUROC: random draws, mean ± SD (min–max) | Sensitivity at τ = 0.50: first-N | Sensitivity at τ = 0.50: random draws, mean ± SD | False reject at τ = 0.50: random draws, mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.383 | 0.323 ± 0.032 (0.291–0.372) | 74.7% | 67.5 ± 6.8% | 100.0 ± 0.0% |
| **LightGBM (SQI)** | 0.731 | 0.750 ± 0.049 (0.700–0.821) | 9.3% | 16.0 ± 4.7% | 4.3 ± 1.7% |
| **Independent SQI MLP** | 0.617 | 0.670 ± 0.030 (0.628–0.703) | 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| **1D CNN (Raw Signal)** | 0.726 | 0.707 ± 0.062 (0.660–0.816) | 61.3% | 66.7 ± 6.3% | 22.7 ± 3.7% |
| **2D CNN (Rendered Image)** | 0.746 | 0.781 ± 0.064 (0.713–0.884) | 17.3% | 21.6 ± 4.8% | 6.7 ± 3.0% |
| **SinglePass Multimodal Net** | 0.627 | 0.668 ± 0.038 (0.620–0.707) | 25.3% | 33.3 ± 6.9% | 0.3 ± 0.6% |
| **Rule-Based SQI Heuristic** | 0.700 | 0.683 ± 0.046 (0.639–0.737) | 96.0% | 92.5 ± 2.9% | 88.0 ± 2.5% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.650 ± 0.059 (0.583–0.723) | 100.0% | 99.7 ± 0.6% | 100.0 ± 0.0% |

## Table 6

| Model | AUROC range over seeds × subsets | SD across training seeds (mean over subsets) | SD across subsets (mean over seeds) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.291–0.372 | 0.000 | 0.032 |
| **LightGBM (SQI)** | 0.700–0.821 | 0.000 | 0.049 |
| **Independent SQI MLP** | 0.613–0.707 | 0.007 | 0.033 |
| **1D CNN (Raw Signal)** | 0.643–0.825 | 0.009 | 0.066 |
| **2D CNN (Rendered Image)** | 0.713–0.904 | 0.007 | 0.064 |
| **SinglePass Multimodal Net** | 0.500–0.707 | 0.091 | 0.015 |

## Table 7

| Configuration | BUT QDB AUROC [95% CI] | BUT QDB Sens. / FR at τ = 0.50 | CinC AUROC | CinC Sens. / FR at τ = 0.50 | NSTDB clean P(usable) | NSTDB paired change, +24 dB | NSTDB paired change, 0 dB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary (SQI text + image)** | 0.852 [0.717 – 0.944] | 53.2% / 4.0% | 0.636 | 100.0% / 100.0% | 0.777 | -0.018 | -0.092 |
| **Image only (no SQI text)** | 0.913 [0.855 – 0.972] | 99.7% / 81.3% | 0.680 | 81.3% / 78.7% | 0.459 | -0.037 | -0.039 |
| **Reworded question (SQI text + image)** | 0.869 [0.732 – 0.955] | 17.5% / 0.5% | 0.627 | 100.0% / 98.7% | 0.870 | -0.017 | -0.073 |

## Table 1c

| Model Architecture | Target 1 (FPR ≤ 5%): nested Sens. / FPR | Target 1: post-hoc Sens. / FPR | Target 2 (Sens. ≥ 80%): nested Sens. / FPR | Target 2: post-hoc Sens. / FPR | Youden: nested Sens. / FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 77.4% / 6.7% | 74.1% / 4.8% | 87.5% / 13.1% | 80.1% / 8.3% | 73.1% / 8.7% |
| **LightGBM (SQI)** | 70.4% / 3.7% | 80.5% / 4.8% | 92.9% / 9.5% | 80.5% / 4.8% | 62.6% / 5.7% |
| **Independent SQI MLP** | 65.3% / 4.8% | 70.0% / 4.9% | 88.6% / 12.3% | 80.5% / 9.3% | 66.3% / 9.1% |
| **1D CNN (Raw Signal)** | 97.3% / 3.5% | 96.0% / 3.5% | 81.5% / 1.2% | 80.1% / 1.3% | 97.6% / 3.5% |
| **2D CNN (Rendered Image)** | 94.6% / 4.7% | 94.3% / 4.7% | 76.8% / 0.9% | 80.1% / 0.9% | 91.9% / 5.2% |
| **SinglePass Multimodal Net** | 70.0% / 5.3% | 69.0% / 4.7% | 78.1% / 11.1% | 80.1% / 6.5% | 85.5% / 7.9% |
| **Rule-Based SQI Heuristic** | 84.8% / 5.3% | 84.5% / 4.9% | 69.0% / 3.2% | 80.5% / 4.1% | 90.2% / 8.5% |
| **LiquidAI d1-3B (Zero-Shot)** | 53.2% / 4.0% | 53.2% / 4.0% | 76.1% / 34.3% | 80.1% / 32.7% | 63.0% / 14.7% |

## Table 1e

| Model Architecture | AP, quota (28.4%) | AP, natural (16.0%) | AP, natural excl. 105001 (2.8%) | PPV at τ = 0.50, 16.0% | PPV at τ = 0.50, 2.8% | NPV at τ = 0.50, 2.8% | False rejections per hour of usable signal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.888 | 0.823 | 0.587 | 0.700 | 0.260 | 0.993 | 22.6 |
| **LightGBM (SQI)** | 0.928 | 0.882 | 0.728 | 0.772 | 0.339 | 0.992 | 14.9 |
| **Independent SQI MLP** | 0.880 | 0.806 | 0.551 | 0.632 | 0.205 | 0.993 | 31.2 |
| **1D CNN (Raw Signal)** | 0.975 | 0.954 | 0.831 | 0.957 | 0.768 | 0.993 | 2.4 |
| **2D CNN (Rendered Image)** | 0.971 | 0.947 | 0.809 | 0.942 | 0.709 | 0.994 | 3.4 |
| **SinglePass Multimodal Net** | 0.928 | 0.878 | 0.700 | 0.867 | 0.496 | 0.989 | 6.7 |
| **Rule-Based SQI Heuristic** | 0.948 | 0.910 | 0.765 | 0.913 | 0.613 | 0.990 | 4.3 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.740 | 0.630 | 0.351 | 0.718 | 0.277 | 0.986 | 14.4 |

## Table 1f

| Held-out subject (fold) | Unusable / usable windows | 1D CNN (Raw Signal) | 2D CNN (Rendered Image) | SinglePass Multimodal Net | LightGBM (SQI) | Rule-Based SQI Heuristic | LiquidAI d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 (4)** | 22 / 50 | 14% | 86% | 64% | 5% | 9% | 5% |
| **105 (3)** | 60 / 50 | 100% | 100% | 100% | 100% | 100% | 65% |
| **111 (2)** | 60 / 50 | 32% | 27% | 35% | 37% | 12% | 53% |
| **113 (0)** | 4 / 50 | 100% | 100% | 75% | 100% | 50% | 100% |
| **114 (1)** | 60 / 50 | 98% | 100% | 100% | 100% | 98% | 100% |
| **122 (0)** | 60 / 50 | 100% | 73% | 0% | 100% | 100% | 33% |
| **124 (4)** | 31 / 50 | 74% | 100% | 100% | 35% | 19% | 6% |
| **False reject, range over 15 subjects** | — / 750 | 0–6% | 0–4% | 0–10% | 0–22% | 0–8% | 0–16% |

## Table 2e

| Model Architecture | Reported subset: lead II | Reported subset: max over 12 leads | Reported subset: mean over 12 leads | Random subsets: lead II, mean (min–max) | Random subsets: max over 12, mean (min–max) | Random subsets: mean over 12, mean (min–max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.383 | 0.393 | 0.226 | 0.323 (0.291–0.372) | 0.352 (0.320–0.407) | 0.157 (0.121–0.222) |
| **LightGBM (SQI)** | 0.731 | 0.729 | 0.858 | 0.750 (0.700–0.821) | 0.667 (0.624–0.726) | 0.877 (0.845–0.932) |
| **Independent SQI MLP** | 0.617 | 0.735 | 0.735 | 0.670 (0.628–0.703) | 0.807 (0.763–0.832) | 0.809 (0.766–0.834) |
| **1D CNN (Raw Signal)** | 0.726 | 0.330 | 0.803 | 0.707 (0.660–0.816) | 0.214 (0.153–0.251) | 0.801 (0.753–0.879) |
| **2D CNN (Rendered Image)** | 0.746 | 0.761 | 0.870 | 0.781 (0.713–0.884) | 0.692 (0.675–0.713) | 0.852 (0.822–0.910) |
| **SinglePass Multimodal Net** | 0.627 | 0.787 | 0.787 | 0.668 (0.620–0.707) | 0.837 (0.775–0.868) | 0.841 (0.783–0.872) |
| **Rule-Based SQI Heuristic** | 0.700 | 0.835 | 0.715 | 0.683 (0.639–0.737) | 0.860 (0.821–0.900) | 0.743 (0.701–0.778) |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.800 | 0.630 | — | — | — |

## Table 2f

| Model Architecture | Rule | Threshold (from BUT QDB) | BUT QDB Sens. / FPR | CinC 2011 reported Sens. / FPR | CinC 2011 random subsets Sens. range | CinC 2017 Sens. / FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | Target 1 (FPR ≤ 5%) | 0.5644 | 74.1% / 4.8% | 74.7% / 100.0% | 61.3–77.3% | 74.5% / 15.7% |
| **Logistic Regression (SQI)** | Target 2 (Sens. ≥ 80%) | 0.3929 | 80.1% / 8.3% | 74.7% / 100.0% | 61.3–77.3% | 83.5% / 25.5% |
| **LightGBM (SQI)** | Target 1 (FPR ≤ 5%) | 0.3602 | 80.5% / 4.8% | 41.3% / 1.3% | 42.7–58.7% | 79.5% / 19.2% |
| **LightGBM (SQI)** | Target 2 (Sens. ≥ 80%) | 0.3602 | 80.5% / 4.8% | 41.3% / 1.3% | 42.7–58.7% | 79.5% / 19.2% |
| **Independent SQI MLP** | Target 1 (FPR ≤ 5%) | 0.725 | 70.0% / 4.9% | 0.0% / 0.0% | 0.0–0.0% | 74.5% / 16.5% |
| **Independent SQI MLP** | Target 2 (Sens. ≥ 80%) | 0.4337 | 80.5% / 9.3% | 0.0% / 0.0% | 0.0–0.0% | 88.1% / 26.2% |
| **1D CNN (Raw Signal)** | Target 1 (FPR ≤ 5%) | 0.007939 | 96.0% / 3.5% | 70.7% / 28.0% | 70.7–86.7% | 99.6% / 54.5% |
| **1D CNN (Raw Signal)** | Target 2 (Sens. ≥ 80%) | 0.3318 | 80.1% / 1.3% | 62.7% / 16.0% | 62.7–78.7% | 93.5% / 36.9% |
| **2D CNN (Rendered Image)** | Target 1 (FPR ≤ 5%) | 0.05569 | 94.3% / 4.7% | 52.0% / 10.7% | 54.7–70.7% | 96.4% / 50.4% |
| **2D CNN (Rendered Image)** | Target 2 (Sens. ≥ 80%) | 0.4892 | 80.1% / 0.9% | 17.3% / 4.0% | 17.3–28.0% | 89.9% / 34.0% |
| **SinglePass Multimodal Net** | Target 1 (FPR ≤ 5%) | 0.1746 | 69.0% / 4.7% | 25.3% / 0.0% | 24.0–40.0% | 92.8% / 43.1% |
| **SinglePass Multimodal Net** | Target 2 (Sens. ≥ 80%) | 0.1054 | 80.1% / 6.5% | 25.3% / 0.0% | 24.0–40.0% | 94.6% / 46.2% |
| **Rule-Based SQI Heuristic** | Target 1 (FPR ≤ 5%) | 0.3762 | 84.5% / 4.9% | 100.0% / 100.0% | 100.0–100.0% | 52.9% / 6.2% |
| **Rule-Based SQI Heuristic** | Target 2 (Sens. ≥ 80%) | 0.407 | 80.5% / 4.1% | 98.7% / 98.7% | 97.3–98.7% | 43.5% / 5.2% |
| **LiquidAI d1-3B (Zero-Shot)** | Target 1 (FPR ≤ 5%) | 0.5 | 53.2% / 4.0% | 100.0% / 100.0% | 98.7–100.0% | 19.8% / 4.0% † |
| **LiquidAI d1-3B (Zero-Shot)** | Target 2 (Sens. ≥ 80%) | 0.3775 | 80.1% / 32.7% | 100.0% / 100.0% | 100.0–100.0% | 40.6% / 14.4% † |

## Table 2g

| SQI feature | BUT QDB (1,047 windows) | CinC 2011 lead II (565 records) | CinC 2011 lead II (reported 150) | CinC 2017 (first window, all recordings) |
| :--- | :---: | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.030 | 0.308 | 0.305 | 0.124 |
| `baseline_wander_ratio` | 0.421 | 0.437 | 0.435 | 0.595 |
| `high_frequency_noise_ratio` | 0.777 | 0.307 | 0.329 | 0.414 |
| `saturation_fraction` | 0.501 | 0.634 | 0.593 | 0.505 |
| `flatline_fraction` | 0.419 | 0.645 | 0.632 | 0.281 |
| `qrs_energy_ratio` | 0.132 | 0.226 | 0.263 | 0.231 |
| `kurtosis_ksqi` | 0.164 | 0.269 | 0.289 | 0.401 |
| `skewness_ssqi` | 0.093 | 0.338 | 0.358 | 0.342 |
| `zero_crossing_rate` | 0.727 | 0.248 | 0.311 | 0.470 |
| `powerline_noise_ratio` | 0.773 | 0.312 | 0.324 | 0.426 |
| `signal_std` | 0.385 | 0.402 | 0.385 | 0.786 |

## Table 4b

| Step | Median (three runs) | Range of run medians | p95 (median over runs) |
| :--- | :---: | :---: | :---: |
| Unit conversion (µV to mV) | 0.01 ms | 0.01–0.01 ms | 0.02 ms |
| Resampling 1,000 → 500 Hz | 0.10 ms | 0.09–0.10 ms | 0.12 ms |
| SQI extraction (13 features) | 1.72 ms | 1.70–1.76 ms | 1.95 ms |
| Rendering, Matplotlib (used by all image models) | 58.41 ms | 58.18–59.91 ms | 121.16 ms |
| Rendering, direct raster (PIL; timing only) | 0.63 ms | 0.62–0.68 ms | 0.72 ms |
| 1D CNN forward, CPU | 1.30 ms | 1.27–1.35 ms | 1.49 ms |
| 1D CNN raw-to-decision, CPU | 1.41 ms | 1.39–1.46 ms | 1.61 ms |
| 2D CNN forward, CPU | 5.11 ms | 5.10–7.13 ms | 5.61 ms |
| Multimodal net forward, CPU | 4.39 ms | 4.32–5.58 ms | 5.10 ms |
| 1D CNN forward, MPS | 2.80 ms | 2.75–2.80 ms | 3.24 ms |
| 1D CNN raw-to-decision, MPS | 2.90 ms | 2.85–2.94 ms | 3.36 ms |
| 2D CNN forward, MPS | 2.92 ms | 2.90–3.75 ms | 3.87 ms |
| Multimodal net forward, MPS | 3.17 ms | 3.15–3.98 ms | 4.48 ms |

## Table 7b

| d1-3B representation | BUT QDB AUROC [95% CI] (5-fold subject CV) | CinC 2011 reported subset AUROC | CinC 2011 random subsets AUROC, mean (min–max) | CinC 2017 balanced subset AUROC (recording max) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-shot choice probability (primary configuration)** | 0.852 [0.717 – 0.944] | 0.636 | 0.650 (0.583–0.723) | 0.739 |
| **Linear probe on the decision state (final norm, answer position)** | 0.974 [0.930 – 0.997] | 0.475 | 0.443 (0.411–0.482) | 0.883 |
| **Linear probe on image tokens (projector output, mean)** | 0.979 [0.941 – 0.996] | 0.777 | 0.825 (0.756–0.907) | 0.910 |

## Table 8

| Model Architecture | AUROC [95% CI], recording max | AP [95% CI], recording max (prevalence 3.3%) | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.868 [0.848 – 0.886] | 0.173 [0.146 – 0.207] | 80.2% / 18.9% | 0.126 / 0.992 | 0.916 |
| **LightGBM (SQI)** | 0.880 [0.863 – 0.895] | 0.234 [0.187 – 0.285] | 67.6% / 14.6% | 0.135 / 0.987 | 0.930 |
| **Independent SQI MLP** | 0.868 [0.852 – 0.883] | 0.140 [0.120 – 0.171] | 86.3% / 24.0% | 0.108 / 0.994 | 0.934 |
| **1D CNN (Raw Signal)** | 0.773 [0.760 – 0.786] | 0.064 [0.057 – 0.073] | 92.4% / 34.5% | 0.083 / 0.996 | 0.801 |
| **2D CNN (Rendered Image)** | 0.850 [0.832 – 0.867] | 0.137 [0.115 – 0.172] | 89.9% / 33.8% | 0.082 / 0.995 | 0.879 |
| **SinglePass Multimodal Net** | 0.835 [0.818 – 0.852] | 0.107 [0.092 – 0.128] | 88.8% / 35.7% | 0.078 / 0.994 | 0.888 |
| **Rule-Based SQI Heuristic** | 0.887 [0.867 – 0.905] | 0.182 [0.155 – 0.217] | 18.3% / 2.5% | 0.199 / 0.973 | 0.933 |

## Table 8b

| Model Architecture | AUROC [95% CI], recording max (278 noisy + 278 non-noisy) | Sens. / FR at τ = 0.50 |
| :--- | :---: | :---: |
| **Logistic Regression (SQI)** | 0.860 [0.826 – 0.890] | 80.2% / 18.3% |
| **LightGBM (SQI)** | 0.880 [0.849 – 0.905] | 67.6% / 15.5% |
| **Independent SQI MLP** | 0.866 [0.835 – 0.896] | 86.3% / 23.0% |
| **1D CNN (Raw Signal)** | 0.767 [0.720 – 0.814] | 92.4% / 35.3% |
| **2D CNN (Rendered Image)** | 0.859 [0.825 – 0.889] | 89.9% / 32.4% |
| **SinglePass Multimodal Net** | 0.848 [0.812 – 0.880] | 88.8% / 32.0% |
| **Rule-Based SQI Heuristic** | 0.876 [0.843 – 0.905] | 18.3% / 3.2% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.739 [0.697 – 0.778] | 19.8% / 4.0% |

