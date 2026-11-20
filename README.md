# Can You Fool an Industrial Anomaly Detector?
### Adversarial Evasion Attacks and Defense-in-Depth on Cyber-Physical Water Circulation Telemetry (SKAB)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%2F%20CUDA-ee4c2c.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-f7931e.svg)
![Domain](https://img.shields.io/badge/Domain-CPS%20%2F%20ICS%20Security-critical.svg)
![License](https://img.shields.io/badge/License-GPL--3.0-green.svg)

---

## 1. Executive Summary & Cyber-Physical Threat Model

Modern Industrial Control Systems (ICS) and Supervisory Control and Data Acquisition (SCADA) networks increasingly rely on deep neural anomaly detectors—such as Long Short-Term Memory Autoencoders (**LSTM-AE**)—to identify physical equipment faults, valve blockages, cavitation, and cyber-physical sabotage in real time. While deep temporal models excel at capturing nonlinear cross-sensor dynamics under benign operating conditions, their reliance on differentiable reconstruction residuals introduces a critical attack surface.

### The Man-in-the-Middle (MitM) Sensor Spoofing Scenario
In this project, I investigate whether a sophisticated adversary can **hide an active physical fault** (such as a partially closed circulation valve on a water pump testbed) from an industrial anomaly detector. I model an attacker who has compromised the Programmable Logic Controller (PLC) or fieldbus communication layer (e.g., Modbus TCP / OPC UA) between the physical sensor array and the historian/IDS:

1. **Physical State:** An actual mechanical anomaly is occurring on the water loop (`valve1` or `valve2` flow restriction), causing genuine deviations in fluid flow rate (`RateRMS`), loop pressure (`Pressure`), motor current (`Current`), and vibration (`Accelerometer1RMS`, `Accelerometer2RMS`).
2. **Adversarial Objective:** The attacker intercepts the 60-second telemetry window $\mathbf{X} \in \mathbb{R}^{60 \times 8}$ before it reaches the anomaly detector and injects a bounded additive perturbation $\boldsymbol{\delta} \in \mathbb{R}^{60 \times 8}$ such that the spoofed telemetry $\mathbf{X}_{\text{adv}} = \mathbf{X} + \boldsymbol{\delta}$ suppresses the LSTM Autoencoder's Mean Squared Error (MSE) reconstruction loss below the 99th-percentile alarm threshold $\tau_{99}$.
3. **Physical Plausibility Constraint:** To avoid triggering hard-coded SCADA range alarms or human operator suspicion on the HMI console, the attacker constrains $\boldsymbol{\delta}$ within an $L_\infty$ ball proportional to each sensor's natural anomaly-free standard deviation $\sigma_j$:
   $$\|\boldsymbol{\delta}_{:, j}\|_\infty \le \epsilon \cdot \sigma_j, \quad \epsilon \in [0.00, 0.50]$$

---

## 2. Dataset & Zero-Leakage Preprocessing Pipeline

I evaluate our threat model on the **[Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB)**, recorded from a physical water circulation testbed equipped with 8 heterogeneous industrial sensors sampled at $1\text{ Hz}$:

| Sensor Feature | Physical Quantity Measured | Unit | Role in Coupled Loop Dynamics |
| :--- | :--- | :--- | :--- |
| `Accelerometer1RMS` | Vibration acceleration (Sensor 1) | $g$ | Detects mechanical cavitation and pump bearing stress |
| `Accelerometer2RMS` | Vibration acceleration (Sensor 2) | $g$ | Secondary housing vibration signature |
| `Current` | Electric motor amperage | $\text{A}$ | Reflects hydraulic load on the impeller |
| `Pressure` | Loop pressure downstream of water pump | $\text{Bar}$ | Spikes or drops sharply during valve closure (`valve1`/`valve2`) |
| `Temperature` | Electric motor body temperature | $^\circ\text{C}$ | Thermal inertia indicator across extended runs |
| `Thermocouple` | Circulation fluid temperature | $^\circ\text{C}$ | Tracks heat dissipation within the closed water loop |
| `Voltage` | Electric motor supply voltage | $\text{V}$ | Monitors power supply stability |
| `RateRMS` | Fluid circulation flow rate | $\text{L/min}$ | Primary hydraulic throughput indicator |

### Strict Zero-Leakage Protocol
To ensure realistic industrial evaluation without future-to-past or cross-experiment leakage:
1. **Normal-Only Global Standardization:** `StandardScaler` ($\mu_{\text{norm}}, \sigma_{\text{norm}}$) is fit **exclusively** on the `anomaly-free/` corpus.
2. **Per-Experiment Ambient Alignment:** Because separate testbed runs exhibit slow ambient thermal drift (e.g., water and motor starting temperatures varying across days), each test file in `valve1/` and `valve2/` is zero-centered using its own initial 60-second healthy calibration prefix (`anomaly == 0`) before scaling by $\sigma_{\text{norm}}$.
3. **File-Isolated Causal Windowing:** Sliding windows of length $W = 60\text{ s}$ (stride $S = 1\text{ s}$) are extracted **strictly inside individual CSV files** so no window ever bridges discontinuous timestamps across separate experiments.
   - **Training Tensor (`X_train`):** `(18250, 60, 8)` — 80% of chronological normal windows (`anomaly-free`).
   - **Validation Tensor (`X_val`):** `(4562, 60, 8)` — 20% of chronological normal windows used exclusively to calibrate the 99th-percentile false-alarm threshold ($\text{FAR} \approx 1\%$).
   - **Test Tensor (`X_test`):** `(21292, 60, 8)` — 20 experiment runs from `valve1/` and `valve2/` containing `36.76%` ground-truth anomalous windows.

---

## 3. Exploratory Telemetry & Cross-Sensor Physics

### Multi-Panel Sensor Telemetry with Ground-Truth Anomaly Regions
During valve closure experiments (`valve1` and `valve2`), restricting the water intake or outlet produces immediate hydraulic transients—most notably sharp step changes in `RateRMS` and `Pressure` accompanied by secondary shifts in vibrational energy (`Accelerometer1RMS`, `Accelerometer2RMS`).

![Multi-Panel Sensor Time Series with Shaded Anomaly Regions](figures/01_sensor_timeseries_shaded.png)
*Figure 1: Full 8-sensor telemetry trace from a representative `valve1` experiment. Crimson shaded regions denote physical valve closure intervals (`anomaly == 1`).*

---

### Cross-Sensor Correlation Structure
Under normal closed-loop operation (`anomaly-free`), thermal and hydraulic sensors exhibit structured physical coupling. An adversary attempting to spoof a single sensor in isolation risks breaking these natural cross-sensor correlations, whereas a white-box gradient attack perturbs all 8 channels simultaneously to mimic normal latent manifold projections.

![Cross-Sensor Correlation Matrix Heatmap](figures/02_sensor_correlation_heatmap.png)
*Figure 2: Pearson correlation matrix across all 8 SKAB sensors computed strictly on the `anomaly-free` baseline corpus.*

---

## 4. Baseline Anomaly Detectors & Unattacked Performance

I benchmark two distinct paradigms trained exclusively on normal operating data (`X_train`):
1. **Isolation Forest (`IsolationForest`):** A non-differentiable, tree-based ensemble trained on summary statistics of each 60-second window (per-sensor window mean $\mu_W$, window standard deviation $\sigma_W$, and terminal causal reading $x_T$, yielding $24$ features per window).
2. **LSTM Autoencoder (`LSTMAutoencoder`):** A sequence-to-sequence PyTorch neural network (`8 -> 32 -> 16 [latent bottleneck] -> 32 -> 8`) trained for 25 epochs to minimize Mean Squared Error (MSE) across the full $60 \times 8$ temporal window (final train MSE: `0.261`).

For both detectors, the decision threshold $\tau_{99}$ is locked at the **99th percentile** of anomaly scores on the held-out healthy validation split (`X_val`), giving a ~1% false-alarm rate on normal data (LSTM-AE: `0.3704`, Isolation Forest: `0.5375`).

| Detector Architecture | F1 Score | Precision | Recall (TPR) | ROC-AUC | Mean Detection Delay (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (Tree Ensemble)** | `0.023` | `0.958` | `0.012` | `0.807` | `119.1 s` |
| **LSTM Autoencoder (Deep Sequence Model)** | `0.011` | `0.538` | `0.006` | `0.520` | `12.0 s` |

At the strict 1% false-alarm threshold, **both detectors catch very few faults** (recall around 1%). Isolation Forest is far more precise and separates faults from normal operation much better (ROC-AUC `0.81`), while the LSTM Autoencoder is close to chance (ROC-AUC `0.52`).

### Temporal Reconstruction Error vs. Alarm Threshold
On clean test data, the LSTM Autoencoder's reconstruction error **rarely crosses** the $\tau_{99}$ threshold. The mean MSE on anomalous windows (`0.248`) is below the threshold (`0.370`) and even slightly below the model's own training MSE on normal data (`0.261`), so the autoencoder flags only `0.55%` of anomalous windows. A likely cause is the per-experiment alignment step, which zero-centers each test run and may remove part of the fault signature before it reaches the model (see Limitations).

![LSTM-AE Reconstruction Error Over Time](figures/03_error_over_time.png)
*Figure 3: LSTM Autoencoder reconstruction MSE over time across test experiments. The dashed horizontal line marks the 99th-percentile validation threshold $\tau_{99}$, and shaded crimson bands mark true physical anomalies.*

---

### Threshold-Independent Discrimination (ROC & Precision-Recall Curves)
Across all possible thresholds, **Isolation Forest separates faults from normal operation much better** than the LSTM Autoencoder (ROC-AUC `0.81` vs. `0.52`). Simple window statistics carry more fault information here than the sequence model's reconstruction error.

![ROC and Precision-Recall Curves](figures/04_roc_pr_curves.png)
*Figure 4: Receiver Operating Characteristic (ROC, left) and Precision-Recall (PR, right) curves comparing the LSTM Autoencoder and Isolation Forest on clean `valve1` and `valve2` test data.*

---

## 5. White-Box Evasion Attack ($L_\infty$ Projected Gradient Descent)

### Attack Formulation
To test whether the LSTM Autoencoder can be fooled during an active physical anomaly, I formulate a white-box evasion attack using **Projected Gradient Descent (PGD)**. Given a genuinely anomalous window $\mathbf{X}_{\text{anom}}$ that triggers an alarm ($f_{\text{AE}}(\mathbf{X}_{\text{anom}}) > \tau_{99}$), the adversary iteratively updates $\mathbf{X}^{(k)}$ over $K = 25$ steps (step size $\alpha = \epsilon / 6$ in standardized units) to **minimize** the autoencoder's reconstruction error:

$$\mathbf{X}^{(k+1)} = \Pi_{\mathcal{B}_\infty(\mathbf{X}_{\text{anom}}, \epsilon \boldsymbol{\sigma})} \left( \mathbf{X}^{(k)} - \alpha \cdot \text{sign}\left( \nabla_{\mathbf{X}^{(k)}} \left\| \text{AE}(\mathbf{X}^{(k)}) - \mathbf{X}^{(k)} \right\|_2^2 \right) \right)$$

where $\Pi_{\mathcal{B}_\infty}$ projects the perturbation back onto the per-sensor physical budget $[-\epsilon \sigma_j, +\epsilon \sigma_j]$ at every step.

**Attack set:** the attack targets only anomalous windows the LSTM-AE detected on clean data. Because its clean recall is low, this is a small set of **43 windows**, so the attack percentages below move in steps of about 2.3 points.

### Waveform Spoofing in Physical Engineering Units
At a budget of $\epsilon = 0.20\sigma$ (at most 20% of one normal standard deviation per sensor), the spoofed signal stays close to the true anomalous waveform in physical units (`L/min`, `Bar`, `A`, `g`). Across all anomalous test windows, the attack lowers the mean reconstruction MSE from `0.248` to `0.175` and drops the LSTM-AE detection rate from `0.55%` to `0.00%`.

![Clean vs Adversarially Perturbed Sensor Signal Overlay](figures/05_clean_vs_perturbed_overlay.png)
*Figure 5: Comparison of original anomalous telemetry (solid navy) vs. PGD-spoofed telemetry at $\epsilon = 0.20\sigma$ (dashed crimson) in physical engineering units alongside the resulting drop in LSTM-AE reconstruction error.*

---

## 6. Defense-in-Depth: Hardening CPS Anomaly Detection

The LSTM Autoencoder's anomaly score is a smooth, differentiable function $\left\|\text{AE}(\mathbf{X}) - \mathbf{X}\right\|_2^2$, so PGD can follow its gradient to lower the reconstruction error without restoring a healthy physical state.

I evaluate two layered defenses, plus Isolation Forest on its own as a transfer check:
1. **Transfer Check (Isolation Forest alone):** Isolation Forest uses non-differentiable, axis-aligned splits on window statistics ($\mu_W, \sigma_W, x_T$), so the PGD attack cannot optimize against it. In practice the opposite of evasion happens: the sign-based PGD noise changes the window statistics enough that Isolation Forest **flags the spoofed windows more often**, rising from `2.3%` on the clean attack set to `100%` at $\epsilon \ge 0.15\sigma$. The attack is not stealthy to a second model.
2. **Defense 1: Hybrid Ensemble (`LSTM-AE + Isolation Forest`):** A logical OR of the two detectors. An alarm fires if either model flags the window, so an attacker must evade both.
3. **Defense 2: Temporal Residual Smoothing + Hybrid Ensemble:** Averages LSTM-AE errors over 5 consecutive windows, compares the result to a slightly lower threshold ($0.85\,\tau_{99}$), then ORs it with Isolation Forest. Smoothing suppresses window-level adversarial jitter.

### Post-Attack Detection Rate vs. Perturbation Budget ($\epsilon$)

| Perturbation Budget ($\epsilon \cdot \sigma$) | Undefended LSTM-AE (%) | Isolation Forest (%) | Defense 1: Hybrid Ensemble (%) | Defense 2: Smoothing + Ensemble (%) |
| :---: | :---: | :---: | :---: | :---: |
| **`0.00 σ` (Clean)** | `100.0` | `2.3` | `100.0` | `100.0` |
| **`0.05 σ`** | `95.3` | `90.7` | `100.0` | `100.0` |
| **`0.10 σ`** | `7.0` | `97.7` | `97.7` | `100.0` |
| **`0.15 σ`** | `0.0` | `100.0` | `100.0` | `100.0` |
| **`0.20 σ`** | `0.0` | `100.0` | `100.0` | `100.0` |
| **`0.30 σ`** | `0.0` | `100.0` | `100.0` | `100.0` |
| **`0.40 σ`** | `0.0` | `100.0` | `100.0` | `100.0` |
| **`0.50 σ`** | `0.0` | `100.0` | `100.0` | `100.0` |

*(Note: Rates are measured on the 43 anomalous windows the LSTM-AE detected on clean data. The undefended row is 100% at $\epsilon = 0$ by construction.)*

![Post-Attack Detection Rate vs Perturbation Budget](figures/06_detection_rate_vs_budget.png)
*Figure 6: Detection rate across perturbation budgets $\epsilon \in [0.00, 0.50]\sigma$. The undefended LSTM-AE drops to 7% at $\epsilon = 0.10\sigma$ and 0% from $0.15\sigma$, while Isolation Forest and both ensemble defenses keep detection near 100%.*

---

## 7. Interactive CPS Security Dashboard

For interactive inspection of individual sensor channels, adversarial perturbation envelopes, and detector alarm thresholds across a full valve-fault experiment, open the standalone Plotly dashboard generated in `figures/07_cps_interactive_dashboard.html`:

* **Interactive Artifact:** [`figures/07_cps_interactive_dashboard.html`](figures/07_cps_interactive_dashboard.html)
  * **Panel 1:** Zoomable physical telemetry (`RateRMS` and `Pressure`) comparing clean vs. PGD-spoofed signals.
  * **Panel 2:** Unattacked vs. attacked LSTM-AE reconstruction MSE relative to the $\tau_{99}$ threshold.
  * **Panel 3:** Hybrid Ensemble + Residual Smoothing score during the spoofed valve fault.

---

## 8. Repository Structure & Reproducibility

```text
├── README.md                                    # Project documentation and CPS security analysis
├── config.yaml                                  # Deterministic hyperparameters & paths
├── cps_skab_security_notebook.py                # Script version of the notebook
├── notebooks/
│   └── cps_skab_security_notebook.ipynb         # Complete end-to-end executable notebook
├── data/
│   └── SKAB/                                    # Cloned Skoltech Anomaly Benchmark repository (created on run)
│       ├── anomaly-free/                        # Normal water circulation runs (training/calibration)
│       ├── valve1/                              # Outlet valve restriction experiments (test)
│       └── valve2/                              # Intake valve restriction experiments (test)
└── figures/
    ├── 01_sensor_timeseries_shaded.png          # 8-sensor telemetry overview with shaded faults
    ├── 02_sensor_correlation_heatmap.png        # Normal baseline cross-sensor correlation matrix
    ├── 03_error_over_time.png                   # LSTM-AE reconstruction error & 99th-pct threshold
    ├── 04_roc_pr_curves.png                     # Clean ROC and Precision-Recall benchmark curves
    ├── 05_clean_vs_perturbed_overlay.png        # Physical-unit overlay of 0.20σ PGD sensor spoofing
    ├── 06_detection_rate_vs_budget.png          # Evasion robustness curve (Undefended vs Defenses)
    └── 07_cps_interactive_dashboard.html        # Standalone 3-panel interactive Plotly explorer
```

### Quickstart Instructions
```bash
# 1. Clone this repository
git clone https://github.com/farianishat/cps-skab-adversarial-security.git
cd cps-skab-adversarial-security

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/cps_skab_security_notebook.ipynb
```
All random seeds (`Python`, `NumPy`, `PyTorch`) are locked to `SEED = 42` in `config.yaml`, and the notebook automatically clones the SKAB dataset and regenerates all 7 artifacts in `figures/` on execution.

---

## 9. Key Takeaways for Industrial AI Security
1. **A deep detector can be weaker than a simple one.** On this testbed, Isolation Forest on window statistics separated faults far better than the LSTM Autoencoder (ROC-AUC `0.81` vs. `0.52`). Model complexity alone does not buy detection quality.
2. **Differentiable detectors are easy to evade.** For every fault the LSTM Autoencoder caught, a white-box PGD perturbation of only `0.15σ` per sensor hid it completely.
3. **Gradient attacks are not stealthy to a second model.** The same perturbations made Isolation Forest flag `100%` of spoofed windows, and an ensemble with temporal smoothing kept `100%` detection at every budget. Model diversity is a cheap, effective defense, though here most of the ensemble's strength comes from Isolation Forest.

---

## 10. Limitations & Next Steps
1. **Weak clean baseline.** The LSTM Autoencoder is close to chance on clean data, and its training loss barely decreases (`0.272` to `0.261` over 25 epochs). Next steps: a larger model, more epochs, and testing whether per-experiment zero-centering removes the fault signature.
2. **Small attack set.** The attack is evaluated on only 43 windows, because the attack set is limited to faults the autoencoder already detects. A stronger baseline or a looser threshold would give a larger, more reliable attack evaluation.
3. **Low recall at 1% false-alarm rate.** Both detectors catch about 1% of faults at this threshold. Reporting results at several thresholds would give a fuller picture.
4. **Attacker knowledge.** The attack is white-box against the LSTM-AE only. An adaptive attacker who targets the ensemble directly is a natural next test.

---

## Citation & Acknowledgments
* **Dataset:** Katser, I. D., & Kozitsin, V. O. (2020). *Skoltech Anomaly Benchmark (SKAB)*. Kaggle. DOI: `10.34740/KAGGLE/DSV/1693952` — [https://github.com/waico/SKAB](https://github.com/waico/SKAB)
