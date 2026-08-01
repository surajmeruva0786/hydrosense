# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-08-01

Initial research-prototype release. Every module below is real, working
code, verified end-to-end against the synthetic hydrophone-audio generator
(`src/data/synthetic.py`) — see `data/README.md` for why the real ShipsEar
corpus isn't bundled, and `PROGRESS.md` for the full build log.

### Added

- **Core utilities**: seeding, metrics, YAML config loader, logging.
- **Data layer**: `ManifestDataset`, transforms, synthetic hydrophone-audio
  generator reproducing ShipsEar's 5-category class layout and imbalance.
- **Preprocessing pipeline**: signal conditioning (downmix/resample/
  bandpass), windowed segmentation with silence gating, four time-frequency
  representations (mel, log-mel, CQT, MFCC), recording-level stratified
  train/CV/test splitting, end-to-end `src.preprocessing.run` CLI.
- **Model architectures**: `HydroSense-Base` (4-block CNN), `HydroSense-SE`
  (adds Squeeze-and-Excitation attention), `HydroSense-TL` (frozen YAMNet
  backbone + dense head), behind a common model registry.
- **Training pipeline**: class-weighted cross-entropy + Mixup loss,
  SpecAugment/Gaussian-noise/Mixup online augmentation, cosine-warmup LR
  schedule, early stopping, checkpointing, 5-fold recording-level CV
  training CLI (`src.training.train`) with dispatch between the PyTorch
  (Base/SE) and Keras (TL) code paths.
- **Evaluation**: held-out test-set evaluation CLI producing
  `metrics.json` + a Markdown report.
- **TimeGAN augmentation**: embedder/recovery/generator/supervisor/
  discriminator sub-networks, 3-phase training CLI, synthetic-segment
  sampling CLI, and a validation CLI (PCA/t-SNE, discriminative score,
  predictive score).
- **Explainability (XAI)**: Grad-CAM on the final convolutional block,
  SHAP `DeepExplainer` per-frequency-band attribution, a spectral-peak
  acoustic sanity check, and an `src.xai.explain` CLI tying all three
  together for a single clip.
- **Streamlit demo app** (`app/streamlit_app.py`): spectrogram view, top-3
  predictions, Grad-CAM overlay toggle, SHAP attribution chart, audio
  playback.
- **Tests**: pytest suite covering dataset loading, transforms,
  preprocessing, model forward-passes, metrics, augmentation, and Grad-CAM.
- **CI/CD**: GitHub Actions workflow (lint, format-check, tests, end-to-end
  synthetic-data smoke test); `Dockerfile` + `docker-compose.yml` for the
  demo app; `Makefile` and `.pre-commit-config.yaml` for local dev
  ergonomics.
- **Notebooks**: EDA, representation comparison, TimeGAN validation
  walkthrough, and XAI deep-dive, all runnable against the synthetic
  dataset by default.
- **Docs**: `CONTRIBUTING.md`, `data/README.md` (ShipsEar access-request
  process + synthetic fallback), `DEPLOYMENT.md`, `environment.yml`.

### Fixed

Bugs found and fixed during the synthetic-data smoke-testing pass (see
`PROGRESS.md` for the running log):

- `ManifestDataset._resolve()` double-joining `segment_path` with the
  manifest's parent directory.
- `make_recording_splits()` forcing a minimum of 2 CV folds even when a
  class had fewer than 2 training recordings.
- `shap.DeepExplainer`'s backward hooks being incompatible with in-place
  ReLU ops — `HydroSenseBase`/`SE` now use `inplace=False`.
- `matplotlib.cm.get_cmap` removal in newer matplotlib — `gradcam.py` now
  tries `matplotlib.colormaps[name]` first.
- Self-contradicting CORS/XSRF settings in `.streamlit/config.toml`.

### Known limitations

- All reported benchmark numbers (README §16) are **targets from the
  literature**, not measured results — the real ShipsEar corpus has not
  been run through this pipeline yet. See `README.md` §16 and
  `data/README.md`.
- `HydroSense-TL` (the YAMNet transfer-learning path) is implemented but
  has not been exercised end-to-end locally (`tensorflow`/`tensorflow-hub`
  are heavy optional dependencies) — Base and SE have both been verified
  against synthetic data.
