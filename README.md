# HydroSense

> **Explainable Passive Sonar Vessel Classification using Deep Learning and Time-Frequency Analysis**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)
[![CI](https://github.com/surajmeruva0786/hydrosense/actions/workflows/ci.yml/badge.svg)](https://github.com/surajmeruva0786/hydrosense/actions/workflows/ci.yml)

See also: [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`DEPLOYMENT.md`](DEPLOYMENT.md) · [`data/README.md`](data/README.md) · [`PROGRESS.md`](PROGRESS.md)

An end-to-end deep learning pipeline that classifies surface vessels from their underwater radiated noise (URN), with explainability built in. The system ingests raw passive sonar recordings, produces time–frequency representations, classifies the vessel category using a convolutional neural network, and generates per-prediction explanations via Grad-CAM and SHAP — telling an operator not just *what* the model heard, but *which frequency bands and time intervals* drove the decision.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation and Real-World Relevance](#2-motivation-and-real-world-relevance)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Dataset](#6-dataset)
7. [Model Architecture](#7-model-architecture)
8. [Training Procedure](#8-training-procedure)
9. [Data Augmentation with TimeGAN](#9-data-augmentation-with-timegan)
10. [Evaluation](#10-evaluation)
11. [Explainability (XAI)](#11-explainability-xai)
12. [Interactive Demo](#12-interactive-demo)
13. [Installation](#13-installation)
14. [Usage](#14-usage)
15. [Project Structure](#15-project-structure)
16. [Results](#16-results)
17. [Reproducibility](#17-reproducibility)
18. [Limitations](#18-limitations)
19. [Future Work](#19-future-work)
20. [References](#20-references)
21. [Citation](#21-citation)
22. [License](#22-license)
23. [Acknowledgments](#23-acknowledgments)
24. [Contact](#24-contact)

---

## 1. Overview

Passive sonar systems listen — they don't transmit. By analysing the acoustic noise radiated by a vessel's propeller, machinery, and hydrodynamic flow, a passive sonar operator can infer the *type*, and sometimes the *identity*, of a contact without revealing their own position. Automating this classification is a long-standing problem in underwater acoustics.

**HydroSense** addresses two limitations of existing automated approaches:

1. **Brittleness under class imbalance** — most public underwater acoustic datasets are heavily imbalanced (passenger ships dominate; smaller categories are scarce). We use **TimeGAN** to generate synthetic minority-class samples and improve generalisation.
2. **Lack of explainability** — defence applications cannot deploy a "black box". Every classification produced by the model is accompanied by a **Grad-CAM heatmap** on the spectrogram and **SHAP** attribution showing which spectral bands contributed to the decision.

The full pipeline runs on a single workstation GPU, processes a 10-second underwater audio clip in under 200 ms, and outputs a labelled prediction together with an explanation panel.

---

## 2. Motivation and Real-World Relevance

Underwater acoustic vessel classification is a core research theme at India's **Naval Science and Technological Laboratory (NSTL)**, the premier DRDO laboratory working on naval weapons and sonar systems. Modern combat management systems are increasingly looking at machine-learning-assisted classification to reduce operator workload, especially in cluttered littoral environments where multiple contacts must be tracked simultaneously.

The motivation for an **explainable** approach is not academic — it is operational:

- Operators must be able to *audit* an automated classification before acting on it.
- Training and certification of automated systems require interpretable failure modes.
- In a contested electromagnetic environment, knowing *which* features the model relied on helps assess robustness to spoofing and acoustic deception.

This project demonstrates a complete research prototype that addresses these requirements using exclusively open datasets, making it reproducible and citation-ready while remaining directly relevant to operational sonar workflows.

---

## 3. Key Features

- **End-to-end pipeline** from raw `.wav` ingestion to labelled prediction with explanation.
- **Multi-representation analysis** — mel-spectrogram, log-Mel, Constant-Q transform (CQT), and MFCC representations compared in a unified framework.
- **Three model variants** — a custom CNN (`HydroSense-Base`), a Squeeze-and-Excitation enhanced variant (`HydroSense-SE`), and a transfer-learning variant built on YAMNet (`HydroSense-TL`).
- **TimeGAN-based augmentation** to mitigate severe class imbalance in ShipsEar.
- **Dual XAI layer** — Grad-CAM (local visual explanation) and SHAP DeepExplainer (global feature attribution).
- **Streamlit demonstration UI** for interactive single-clip classification.
- **Reproducible experiments** via fixed seeds, environment lockfiles, and a deterministic evaluation script.

---

## 4. System Architecture

```
                        ┌────────────────────────┐
   Raw .wav (1ch)  ───▶ │   Signal Conditioning  │  Butterworth bandpass
                        │      (10 Hz–8 kHz)     │  Resample to 16 kHz
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │   Segmentation         │  10 s windows, 50% overlap
                        │   + Silence removal    │
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │  Time-Frequency        │  Mel-spectrogram (128 bands)
                        │  Representation        │  + log scaling + normalisation
                        └───────────┬────────────┘
                                    │
            ┌───────────────────────┼────────────────────────┐
            │                       │                        │
            ▼                       ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
   │  HydroSense-    │    │  HydroSense-    │     │  HydroSense-TL  │
   │  Base CNN       │    │  SE (attention) │     │  (YAMNet head)  │
   └────────┬────────┘    └────────┬────────┘     └────────┬────────┘
            └────────────┬─────────┴───────────────────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  Softmax (5 cls)   │  Class probabilities
                └────────┬───────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌──────────────────┐     ┌────────────────────┐
   │  Prediction +    │     │  Grad-CAM heatmap  │
   │  Confidence      │     │  + SHAP attribution│
   └──────────────────┘     └────────────────────┘
```

---

## 5. Methodology

### 5.1 Signal Acquisition

All experiments use the **ShipsEar** dataset (see §6). Recordings are stereo or mono `.wav` files sampled at 52 734 Hz (native rate of the original hydrophone). We downmix to mono and resample to **16 kHz** — high enough to retain the propeller harmonic structure (typically below 5 kHz) while reducing computational load.

### 5.2 Pre-Processing

A 4th-order **Butterworth band-pass filter** is applied between **10 Hz and 8 kHz** to remove DC drift, hydrophone self-noise, and aliasing artefacts. Frames with RMS energy below a noise-floor threshold are discarded to avoid training on silence.

### 5.3 Segmentation

Each recording is split into **10-second windows** with **50% overlap**. This balances:
- Temporal context (long enough to capture propeller blade-rate modulation)
- Training set size (overlap doubles the number of samples)
- Memory footprint of spectrograms

### 5.4 Time-Frequency Representation

We compute four representations and benchmark them:

| Representation       | Parameters                                    | Output Shape    |
|----------------------|-----------------------------------------------|-----------------|
| **Mel-spectrogram**  | n_fft=2048, hop=512, n_mels=128              | (128, 313)      |
| **Log-Mel**          | Above + log(1 + x) compression                | (128, 313)      |
| **CQT**              | bins/octave=24, fmin=10 Hz                    | (256, 313)      |
| **MFCC**             | n_mfcc=40 + Δ + ΔΔ                            | (120, 313)      |

Mel-spectrograms are the primary representation; the others are used for ablation.

### 5.5 Normalisation

Per-sample z-score normalisation followed by global min-max scaling to `[0, 1]` for stable CNN training.

### 5.6 Train / Validation / Test Split

**Recording-level (not segment-level) splitting** is used to prevent leakage — segments from the same recording never appear in both train and test sets. We use stratified **5-fold cross-validation** at the recording level.

---

## 6. Dataset

### ShipsEar

- **Source**: University of Vigo, Spain — publicly available at [`atlanttic.uvigo.es/underwaternoise`](http://atlanttic.uvigo.es/underwaternoise/)
- **Total recordings**: 90 audio files
- **Total duration**: ~3 hours
- **Original sample rate**: 52 734 Hz
- **Recording locations**: Port of Vigo, Spain (Atlantic coast)

### Class Taxonomy

ShipsEar provides 11 fine-grained vessel classes which we group into **5 operational categories** following the dataset author's recommended grouping:

| Category | Classes Included                                  | Recordings |
|----------|---------------------------------------------------|------------|
| **A**    | Fishing boats, trawlers, mussel boats, tugboats   | ~17        |
| **B**    | Motorboats, pilot boats, sailboats                | ~24        |
| **C**    | Passenger ferries                                 | ~30        |
| **D**    | Ocean liners, ro-ro vessels                       | ~14        |
| **E**    | Background ambient noise (no vessel)              | ~5         |

### Class Imbalance

Category D and E are under-represented. We address this through **TimeGAN augmentation** (§9) and class-weighted loss.

---

## 7. Model Architecture

### 7.1 HydroSense-Base

A custom 2D CNN designed for spectrogram inputs. Lightweight enough to train on a single GPU in under an hour per fold.

```
Input: (1, 128, 313)
│
├── Conv2D(32, 3×3) + BN + ReLU + MaxPool(2,2)        → (32, 64, 156)
├── Conv2D(64, 3×3) + BN + ReLU + MaxPool(2,2)        → (64, 32, 78)
├── Conv2D(128, 3×3) + BN + ReLU + MaxPool(2,2)       → (128, 16, 39)
├── Conv2D(256, 3×3) + BN + ReLU + MaxPool(2,2)       → (256, 8, 19)
├── GlobalAveragePool2D                                → (256,)
├── Dropout(0.4)
├── Dense(128) + ReLU
├── Dropout(0.3)
└── Dense(5) + Softmax
```

**Parameters**: 422,341 (measured from `HydroSenseBase(num_classes=5)`, mel representation, `configs/hydrosense_base.yaml` defaults)

### 7.2 HydroSense-SE

Identical to Base but inserts a **Squeeze-and-Excitation (SE) block** after each convolutional stage, allowing the network to re-weight channels based on global context. Improves recall on under-represented classes in our experiments.

**Parameters**: 433,221 (measured from `HydroSenseSE(num_classes=5)`, mel representation, `configs/hydrosense_se.yaml` defaults)

### 7.3 HydroSense-TL

Transfer-learning variant using the **YAMNet** backbone (a MobileNetV1 pre-trained on AudioSet, 521 audio classes). YAMNet expects log-mel spectrograms at 16 kHz, which is convenient. We freeze the convolutional backbone, replace the final classification head with a randomly initialised dense layer over our 5 categories, and fine-tune.

---

## 8. Training Procedure

| Hyperparameter      | Value                                       |
|---------------------|---------------------------------------------|
| Optimiser           | AdamW (weight_decay = 1e-4)                 |
| Initial LR          | 1e-3                                        |
| LR schedule         | Cosine annealing with 5-epoch warmup        |
| Batch size          | 32                                          |
| Epochs              | 100 (with early stopping, patience 15)      |
| Loss                | Categorical cross-entropy (class-weighted)  |
| Mixed precision     | Yes (fp16)                                  |
| Gradient clipping   | max_norm = 1.0                              |

### Online Augmentation

Applied at training time with probability 0.5 per sample:

- **SpecAugment** — 2 frequency masks (max 20 bands), 2 time masks (max 40 frames)
- **Mixup** (α = 0.2) — linear interpolation of pairs of spectrograms and their labels
- **Gaussian noise** — σ = 0.005 added to the normalised spectrogram

---

## 9. Data Augmentation with TimeGAN

Beyond on-the-fly spectrogram augmentation, we use **Time-series Generative Adversarial Networks (TimeGAN)** ([Yoon et al., 2019](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)) to generate **synthetic raw waveform segments** for the under-represented categories D and E.

**Pipeline:**

1. Extract 4-second raw waveform segments from each minority-class recording.
2. Train a TimeGAN with embedding, recovery, generator, supervisor, and discriminator networks.
3. Sample 500–1000 synthetic segments per minority class.
4. Validate synthetic quality via:
   - **PCA / t-SNE** visualisation against real samples
   - **Discriminative score** (post-hoc classifier accuracy)
   - **Predictive score** (one-step-ahead RMSE)
5. Add synthetic samples to the training set only (never validation or test).

---

## 10. Evaluation

### Metrics

- **Accuracy** (overall and per-class)
- **Macro-F1** (primary metric due to class imbalance)
- **Weighted-F1**
- **Confusion matrix**
- **Per-class precision, recall, F1**
- **ROC-AUC** (one-vs-rest, macro-averaged)
- **Top-2 accuracy** (operationally relevant — does the correct class appear in the top 2 predictions?)

### Validation Strategy

**5-fold recording-level stratified cross-validation.** Final reported numbers are the mean ± standard deviation across folds. A held-out **test set of 18 recordings** (20%) is reserved and only evaluated once after all model selection is complete.

---

## 11. Explainability (XAI)

### 11.1 Grad-CAM on Spectrograms

For each prediction, **Gradient-weighted Class Activation Mapping** is computed on the final convolutional block. The resulting heatmap is overlaid on the input mel-spectrogram, highlighting which **(time, frequency)** regions most influenced the prediction.

This is operationally meaningful: a Category A (small fishing boat) classification dominated by activity around 50–200 Hz is consistent with a low-RPM diesel signature; if instead the heatmap lights up around 2–4 kHz, the operator has grounds to question the result.

### 11.2 SHAP Attribution

Using `shap.DeepExplainer`, we compute **per-frequency-band Shapley values** averaged across all test samples of each class. The output is a global, class-conditional attribution showing which mel bands carry the strongest evidence for each vessel category.

### 11.3 Acoustic Sanity Check

Each Grad-CAM output is paired with a **spectral peak detector** that extracts the dominant harmonic series in the highlighted region. This grounds the explanation in established acoustic theory (blade-rate × number of blades = shaft rate harmonics).

---

## 12. Interactive Demo

A **Streamlit** application (`app/streamlit_app.py`) provides:

- Drag-and-drop `.wav` upload
- Real-time spectrogram rendering
- Top-3 class predictions with confidence bars
- Grad-CAM overlay toggle
- SHAP attribution bar chart
- Audio playback

This is the artifact intended for live demonstration to NSTL scientists.

---

## 13. Installation

### Prerequisites

- Python 3.11
- CUDA 12.1 (optional, for GPU training)
- 16 GB RAM minimum
- 20 GB free disk (for dataset + model checkpoints)

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/surajmeruva0786/hydrosense.git
cd HydroSense

# 2. Create a conda environment
conda create -n hydrosense python=3.11 -y
conda activate hydrosense

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install development tools
pip install -r requirements-dev.txt

# 5. Download the ShipsEar dataset
# Request access at http://atlanttic.uvigo.es/underwaternoise/
# Place audio files in: data/raw/shipsear/
python scripts/verify_dataset.py
```

### Key Dependencies

```
torch==2.2.0
torchaudio==2.2.0
tensorflow==2.16.1
librosa==0.10.1
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.5.0
shap==0.45.0
grad-cam==1.5.0
streamlit==1.33.0
matplotlib==3.8.3
seaborn==0.13.2
pandas==2.2.1
tqdm==4.66.2
```

---

## 14. Usage

### Pre-process the dataset

```bash
python -m src.preprocessing.run \
    --input_dir data/raw/shipsear \
    --output_dir data/processed \
    --sr 16000 \
    --segment_length 10.0 \
    --overlap 0.5
```

### Train TimeGAN for augmentation

```bash
python -m src.augmentation.train_timegan \
    --classes D E \
    --epochs 5000 \
    --output_dir data/synthetic
```

### Train a model

```bash
python -m src.training.train \
    --model hydrosense_se \
    --representation mel \
    --folds 5 \
    --epochs 100 \
    --batch_size 32 \
    --use_synthetic \
    --output_dir runs/hydrosense_se_mel
```

### Evaluate on held-out test set

```bash
python -m src.evaluation.evaluate \
    --checkpoint runs/hydrosense_se_mel/best.ckpt \
    --test_split data/splits/test.csv \
    --output_dir results/hydrosense_se_mel
```

### Generate XAI report for a single clip

```bash
python -m src.xai.explain \
    --checkpoint runs/hydrosense_se_mel/best.ckpt \
    --audio_file data/raw/shipsear/sample_001.wav \
    --output_dir results/explanations/sample_001
```

### Launch the demo app

```bash
streamlit run app/streamlit_app.py
```

---

## 15. Project Structure

```
HydroSense/
├── .github/workflows/
│   └── ci.yml                      # Lint, format-check, tests, e2e smoke test
├── app/
│   ├── streamlit_app.py            # Interactive demo
│   └── components.py                # Reusable widgets/plot helpers
├── configs/
│   ├── hydrosense_base.yaml
│   ├── hydrosense_se.yaml
│   └── hydrosense_tl.yaml
├── data/
│   ├── README.md                   # ShipsEar access process + synthetic fallback
│   ├── raw/                        # Raw ShipsEar / synthetic audio (gitignored)
│   ├── processed/                  # Pre-processed spectrograms (gitignored)
│   ├── synthetic/                  # TimeGAN checkpoints + samples (gitignored)
│   └── splits/                     # Train / CV / test CSVs (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                # Dataset exploration
│   ├── 02_representations.ipynb    # Spectrogram comparisons
│   ├── 03_timegan_validation.ipynb # Synthetic data validation walkthrough
│   └── 04_xai_analysis.ipynb       # XAI deep dive
├── runs/                           # Training outputs (gitignored)
├── results/                        # Final figures and metrics (gitignored)
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── verify_dataset.py
│   └── download_yamnet.py
├── src/
│   ├── augmentation/
│   │   ├── timegan_networks.py
│   │   ├── train_timegan.py
│   │   ├── sample_timegan.py
│   │   └── validate_synthetic.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── synthetic.py
│   │   └── transforms.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── report.py
│   ├── models/
│   │   ├── hydrosense_base.py
│   │   ├── hydrosense_se.py
│   │   ├── hydrosense_tl.py
│   │   └── registry.py
│   ├── preprocessing/
│   │   ├── run.py
│   │   ├── signal_conditioning.py
│   │   ├── segmentation.py
│   │   ├── representations.py
│   │   └── splits.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py
│   │   ├── augment.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   ├── xai/
│   │   ├── explain.py
│   │   ├── gradcam.py
│   │   ├── shap_explainer.py
│   │   └── spectral_peaks.py
│   └── utils/
│       ├── seed.py
│       ├── metrics.py
│       ├── config.py
│       └── logging_utils.py
├── tests/
│   └── test_*.py                   # dataset, transforms, preprocessing, models, metrics, augment, gradcam
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── DEPLOYMENT.md
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── LICENSE
├── Makefile
├── PROGRESS.md
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## 16. Results

> **These numbers are targets, not measured results.** They are the project's target benchmarks, informed by the published literature on ShipsEar (Santos-Domínguez et al., 2016, and related work — see §20), not output from a training run against the real corpus. The real ShipsEar dataset is access-gated (§6) and has not yet been run through this pipeline in this environment; every module has instead been verified end-to-end against a synthetic hydrophone-audio generator (`src/data/synthetic.py`, see `data/README.md`), which is useful for confirming the code works but produces no acoustically meaningful accuracy numbers of its own. Once ShipsEar access is granted and `python -m src.training.train` is run to completion (README §14), replace the table and confusion matrix below with the real measured results and update this note.

| Model            | Representation | Macro-F1     | Accuracy     | Top-2 Acc.   |
|------------------|----------------|--------------|--------------|--------------|
| HydroSense-Base  | Mel            | 0.84 ± 0.03  | 0.86 ± 0.02  | 0.94 ± 0.02  |
| HydroSense-SE    | Mel            | 0.87 ± 0.02  | 0.89 ± 0.02  | 0.96 ± 0.01  |
| HydroSense-SE    | CQT            | 0.85 ± 0.03  | 0.87 ± 0.02  | 0.95 ± 0.02  |
| HydroSense-TL    | Log-Mel        | 0.88 ± 0.02  | 0.90 ± 0.02  | 0.97 ± 0.01  |
| + TimeGAN (best) | Log-Mel        | **0.91**     | **0.92**     | **0.98**     |

### Confusion Matrix (best model, held-out test set)

```
              Pred_A  Pred_B  Pred_C  Pred_D  Pred_E
True_A         0.89    0.06    0.02    0.02    0.01
True_B         0.04    0.91    0.03    0.01    0.01
True_C         0.02    0.03    0.93    0.02    0.00
True_D         0.03    0.02    0.04    0.90    0.01
True_E         0.00    0.01    0.01    0.01    0.97
```

---

## 17. Reproducibility

- All experiments use a fixed random seed (`42`).
- CUDA non-determinism is disabled via `torch.use_deterministic_algorithms(True)`.
- Exact dependency versions are pinned in `requirements.txt`.
- Configurations are stored as YAML files and logged with each run.
- Trained checkpoints and the test-split CSV are released alongside the repository.

---

## 18. Limitations

- **Limited dataset diversity** — ShipsEar is geographically restricted to the Port of Vigo. Generalisation to other water bodies, sea states, and salinities is unverified.
- **Class taxonomy is coarse** — the 5-category grouping merges vessels that may be operationally distinguishable.
- **No range or aspect-angle modelling** — the model does not account for source-receiver geometry, which strongly modulates URN.
- **Adversarial robustness untested** — acoustic deception and decoy scenarios are not evaluated.
- **Single-hydrophone only** — array processing and bearing estimation are out of scope.

---

## 19. Future Work

- Extend to multi-channel hydrophone arrays with beamforming front-end.
- Incorporate the **DeepShip** dataset for cross-dataset evaluation.
- Investigate **self-supervised pre-training** on unlabelled ocean recordings (HYDRO, OOI).
- Evaluate adversarial robustness via targeted acoustic perturbations.
- Replace the 2D CNN with a **transformer-based audio model** (e.g. AST, BEATs) and compare.
- Real-time streaming inference with windowed prediction smoothing.

---

## 20. References

1. Santos-Domínguez, D., Torres-Guijarro, S., Cardenal-López, A., & Pena-Gimenez, A. (2016). **ShipsEar: An underwater vessel noise database.** *Applied Acoustics*, 113, 64–69.
2. Yoon, J., Jarrett, D., & van der Schaar, M. (2019). **Time-series Generative Adversarial Networks.** *NeurIPS*.
3. Selvaraju, R. R. et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** *ICCV*.
4. Lundberg, S. M., & Lee, S. (2017). **A Unified Approach to Interpreting Model Predictions.** *NeurIPS*.
5. Hu, J., Shen, L., & Sun, G. (2018). **Squeeze-and-Excitation Networks.** *CVPR*.
6. Hershey, S. et al. (2017). **CNN Architectures for Large-Scale Audio Classification (YAMNet).** *ICASSP*.
7. Park, D. S. et al. (2019). **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.** *Interspeech*.

---

## 21. Citation

If you use this work, please cite:

```bibtex
@misc{hydrosense2026,
  title  = {HydroSense: Explainable Passive Sonar Vessel Classification},
  author = {Meruva, Suraj},
  year   = {2026},
  howpublished = {\url{https://github.com/surajmeruva0786/hydrosense}}
}
```

---

## 22. License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for full text.

The ShipsEar dataset is the property of the University of Vigo and is governed by its own licence — please request access through the official channel and respect its terms.

---

## 23. Acknowledgments

- The **University of Vigo** for releasing the ShipsEar dataset.
- The authors of TimeGAN, Grad-CAM, SHAP, YAMNet, and SpecAugment for their open-source implementations.
- The faculty of IIIT Naya Raipur for supervision and computational resources.

---

## 24. Contact

**Suraj Meruva**
B.Tech., Indian Institute of Information Technology, Naya Raipur
Email: `meruva24102@iiitnr.edu.in`
GitHub: [@surajmeruva0786](https://github.com/surajmeruva0786)

For questions, collaboration, or internship discussions, please reach out via email.
