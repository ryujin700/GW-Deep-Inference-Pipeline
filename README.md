# Deep Learning Pipelines for Gravitational Wave Detection & Parameter Estimation

A high-performance PyTorch implementation benchmarking **1D-CNNs (George & Huerta exact architecture)**, **Deep 1D-ResNets**, and **1D-Vision Transformers (with [CLS] token masking)** for real-time Binary Black Hole (BBH) gravitational wave detection and mass inference from multi-interferometer strain data ($`H_1, L_1, V_1`$).

---

## Pipeline & Physical Modeling

* **Strain Simulation & Conditioning:** 1.0-second time-series sampled at $`f_s = 8192\,\mathrm{Hz}`$. Injects post-Newtonian inspiral-merger chirps across primary masses $`m_1, m_2 \in [10, 50]\,M_\odot`$ into colored Gaussian noise shaped by analytic **LIGO Power Spectral Density (PSD)**.

* **Whitening & Bandpass Filtering:** Frequency-domain noise whitening using zero-phase $`6^{\mathrm{th}}`$-order Butterworth bandpass filtering ($`20\,\mathrm{Hz} - 800\,\mathrm{Hz}`$).

* **Joint Multi-Task Objective:** Unified classification (BCE loss) and parameter regression (MSE loss) with dynamic learning rate warmup scheduling:

 $$Loss = BCE(P_{sig}, y_{cls}) + \lambda(t) \cdot MSE(\hat{m}_1, \hat{m}_2; m_1, m_2)$$

## Model Architectures

1. **George & Huerta Exact 1D-CNN:** 4-stage convolutional hierarchy (kernel sizes 16 to 63) with Batch Normalization, Max Pooling, and Adaptive Global Average Pooling.

2. **Deep 1D-ResNet:** Residual convolutional blocks with identity skip connections and stem downsampling.

3. **1D Transformer with Detector Masking:** 1D patch tokenizer ($`8192 \rightarrow 128`$ tokens) prepended with a learnable `[CLS]` token and sinusoidal positional encodings. Trained with dynamic detector dropout ($`p=0.3`$) for multi-interferometer failure resilience.

---

## Benchmark Results

### Detection & Mass Estimation Performance (Test Set: $`N=1000`$)

| Architecture | Detection AUC | Accuracy | MAE ($`m_1`$) | MAE ($`m_2`$) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **George & Huerta CNN** | **1.0000** | **100.0%** | $`6.05\,M_\odot`$ | $`3.88\,M_\odot`$ | 753s |
| **Deep 1D-ResNet** | **1.0000** | **100.0%** | $`6.22\,M_\odot`$ | $`3.93\,M_\odot`$ | 545s |
| **1D-Transformer (Masked)** | **1.0000** | **100.0%** | **$`5.99\,M_\odot`$** | **$`3.84\,M_\odot`$** | 666s |

### Detector Dropout Robustness (Interferometer Configurations)

| Model Variant | Full Array ($`H_1, L_1, V_1`$) | Masked $`V_1`$ ($`H_1, L_1`$) | Single Detector $`H_1`$ ($`L_1, V_1`$ off) |
| :--- | :---: | :---: | :---: |
| **Baseline Transformer (No Masking)** | AUC: 1.0000 | AUC: 1.0000 | AUC: 1.0000 |
| **Mask-Trained Transformer** | **AUC: 1.0000** | **AUC: 1.0000** | **AUC: 1.0000** |

---

## Real LIGO Event Inference

Validated against catalog parameters from real/injected events:

* **GW150914 (BBH):** $`P(\text{signal}) = 1.000`$, Inferred Chirp Mass $`\mathcal{M}_c \approx 21.65\,M_\odot`$.

* **GW170814 (3-Detector BBH):** $`P(\text{signal}) = 1.000`$, Inferred Chirp Mass $`\mathcal{M}_c \approx 21.57\,M_\odot`$.

* **GW170817 (BNS Out-of-Distribution):** Accurately flags signal presence while demonstrating regression boundary saturation.

---

## How to Run

```bash
git clone https://github.com/ryujin700/GW-Deep-Inference-Pipeline.git
cd GW-Deep-Inference-Pipeline
pip install -r requirements.txt
jupyter notebook notebooks/gravitational_wave_inference.ipynb
