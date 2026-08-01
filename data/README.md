# Data

This directory holds every stage of the HydroSense data pipeline. All
subdirectories except this file, `.gitkeep` placeholders, and small CSV
manifests are gitignored (see the root `.gitignore`) — audio and
spectrogram artifacts are regenerated locally, not committed.

```
data/
├── raw/         # Input .wav recordings + metadata.csv (gitignored)
├── processed/   # Preprocessed segments + <representation>/manifest.csv (gitignored)
├── synthetic/   # TimeGAN checkpoints + synthetic segments + manifest (gitignored)
└── splits/      # train/cv/test split CSVs (gitignored)
```

## The ShipsEar dataset (real data)

HydroSense is designed against **ShipsEar** (README §6): 90 recordings,
~3 hours, collected at the Port of Vigo, Spain, and released by the
University of Vigo.

ShipsEar is **access-gated** — it is not publicly downloadable and cannot
be redistributed with this repository. To use it:

1. Request access at
   [`atlanttic.uvigo.es/underwaternoise`](http://atlanttic.uvigo.es/underwaternoise/).
2. Once granted, place the `.wav` files under `data/raw/shipsear/`.
3. Build a `data/raw/shipsear/metadata.csv` with columns
   `recording_id, file_path, class_name, duration_s`, grouping ShipsEar's
   11 fine-grained classes into the 5 operational categories used
   throughout this project (README §6, "Class Taxonomy"). `scripts/
   verify_dataset.py` sanity-checks this layout before you run the
   pipeline.
4. Run the pipeline as documented in README §14, starting from
   `python -m src.preprocessing.run --input_dir data/raw/shipsear ...`.

Until you have access, every module in this repository is fully exercised
against synthetic data instead (below) — nothing is blocked on ShipsEar
access.

## The synthetic fallback (default / CI)

`src/data/synthetic.py` generates a small, physically-motivated but
**entirely synthetic** hydrophone-audio corpus: harmonic series (propeller
blade-rate + shaft harmonics) plus coloured broadband noise, with a
per-class spectral profile loosely modelled on the five ShipsEar
categories (`CLASS_PROFILES` in that module), at the same class imbalance
ShipsEar exhibits.

Generate it with:

```bash
python scripts/generate_synthetic_dataset.py \
    --output_dir data/raw/shipsear_synthetic \
    --seed 42
```

This is what every notebook, the CI pipeline (`.github/workflows/ci.yml`),
and local development use by default. It exists so the full pipeline —
preprocessing, training, TimeGAN augmentation, evaluation, XAI, the
Streamlit demo — is runnable and testable without waiting on dataset
access.

**Important:** metrics, TimeGAN validation scores, and XAI explanations
produced from synthetic data are *not* meaningful benchmarks — they verify
that the code runs correctly, not that the model has learned anything
acoustically real. README §16's reported numbers are targets from the
published literature on ShipsEar, not measurements from this synthetic
generator. See `PROGRESS.md` for exactly which stages have been
smoke-tested this way.

## Directory contents once populated

- **`data/raw/<corpus_name>/`** — `.wav` files + `metadata.csv`
  (`recording_id, file_path, class_name, duration_s`). Input to
  `src.preprocessing.run`.
- **`data/processed/<representation>/`** — one `.npy` array per segment
  plus `manifest.csv` (`segment_path, recording_id, label, class_name,
  split, fold`). Consumed by `src.data.dataset.ManifestDataset`. Written by
  `src.preprocessing.run`.
- **`data/synthetic/`** — `timegan_<class>.pt` checkpoints (from
  `src.augmentation.train_timegan`) plus, after sampling,
  `<representation>/<class>/*.npy` and `<representation>/manifest.csv`
  (from `src.augmentation.sample_timegan`) — always `split=train,
  fold=-1`, since synthetic data is only ever added to the training set
  (README §9 step 5, §5.6).
- **`data/splits/`** — `train.csv`, `cv_fold_<n>.csv`, `test.csv` written
  by `src.preprocessing.splits.save_splits`, recording-level (not
  segment-level) to prevent leakage between splits.
