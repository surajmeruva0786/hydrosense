# HydroSense — Build Progress

Working log for the full-project build-out described in `README.md`. Updated
as work continues; read this first when resuming.

**Ground rule for this build:** the real ShipsEar dataset is access-gated
(request-only from University of Vigo) and isn't available in this
environment. Every module below is implemented as real, working code and is
verified end-to-end against a synthetic hydrophone-audio generator
(`src/data/synthetic.py`), not against real acoustic data. Reported
benchmark numbers in the README (§16) are targets, not measured results —
that stays true until someone runs training against the real corpus.

## Status: 14 of 15 phases complete (steps 1-52 committed and pushed)

| # | Phase | Status |
|---|---|---|
| 0 | Repo scaffolding (.gitignore, LICENSE, requirements, pyproject, dirs) | ✅ done |
| 1 | Core utils (seed, metrics, config loader, logging) + config YAMLs | ✅ done |
| 2 | Data layer + synthetic dataset generator | ✅ done |
| 3 | Preprocessing pipeline (conditioning, segmentation, representations, splits, run.py) | ✅ done |
| 4 | Model architectures (Base, SE, TL/YAMNet, registry) | ✅ done |
| 5 | Training pipeline (losses, augment, scheduler, checkpoint, train.py) | ✅ done |
| 6 | Evaluation (report.py, evaluate.py) | ✅ done |
| 7 | TimeGAN augmentation (networks, train, sample, validate) | ✅ done |
| 8 | XAI (Grad-CAM, SHAP DeepExplainer, spectral peaks, explain.py) | ✅ done |
| 9 | Streamlit demo app | ✅ done |
| 10 | Tests (39 pytest tests, all passing) | ✅ done |
| 11 | CI/CD (GitHub Actions, Dockerfile, docker-compose, Makefile, pre-commit) | ✅ done |
| 12 | Notebooks (EDA, representations, TimeGAN validation, XAI analysis) | ✅ done |
| 13 | Docs polish (CONTRIBUTING, CHANGELOG, data/README, DEPLOYMENT, env.yml, README status update) | ✅ done |
| 14 | Integration test + release tag (full reinstall from requirements-dev.txt, full pytest run, v0.1.0 tag) | ⬜ not started |

Every phase above (0-10) was smoke-tested end-to-end in a local `.venv`
against synthetic data before being committed — not just written and
assumed correct. Notable bugs found and fixed along the way:

- `ManifestDataset._resolve()` was double-joining `segment_path` with the
  manifest's parent dir (paths written by `preprocessing/run.py` are already
  cwd-relative). Fixed in step 30.
- `make_recording_splits()` forced a minimum of 2 CV folds even when a class
  had fewer than 2 training recordings, crashing `StratifiedKFold` on tiny
  corpora. Fixed in step 42 (surfaced by the new test fixtures).
- `shap.DeepExplainer`'s backward hooks are incompatible with in-place ReLU
  ops in the current PyTorch — switched `HydroSenseBase`/`SE` to
  `inplace=False` (step 39).
- `matplotlib.cm.get_cmap` is removed in newer matplotlib; `gradcam.py` now
  tries `matplotlib.colormaps[name]` first (step 37).
- `.streamlit/config.toml` had a self-contradicting CORS/XSRF config
  (step 43).
- The naive `Dockerfile` (`pip install -r requirements.txt` directly) built
  a 12.7GB image because PyPI's default `torch`/`torchaudio` wheels pull in
  the full CUDA toolkit even in a CPU-only container. Fixed in step 45 by
  installing `torch`/`torchaudio` from PyTorch's CPU-only wheel index
  first; final image is 5.78GB. Verified: `docker build` succeeds,
  container starts, reports `healthy`, `GET /_stcore/health` → 200.
- `train_timegan.py`'s `--metadata_csv` has no relationship to
  `--output_dir`/other paths — it defaults to
  `data/raw/shipsear/metadata.csv` and must be passed explicitly when
  pointing at a different corpus location (hit while writing
  `03_timegan_validation.ipynb`, step 48).

Local dev venv at `Z:\hydrosense\.venv` now also has
`nbformat`/`nbclient`/`ipykernel` installed (for executing the notebooks
added in step 48) on top of the stack noted above. `tensorflow`/
`tensorflow-hub` (needed only for HydroSense-TL / YAMNet) are **still not
installed** in the local venv, though they are present in the Docker image
(step 45) since `requirements.txt` pulls them in for the app.

## What's left (in order)

1. **Phase 14**: reinstall the venv strictly from `requirements-dev.txt`
   (currently a hand-rolled subset was installed incrementally), run the
   full pytest suite + `ruff check` + `black --check` clean, run one
   complete pipeline pass on synthetic data end-to-end (preprocess -> train
   -> evaluate -> explain -> streamlit boots), then tag `v0.1.0`.
2. Decide whether to also actually install/exercise `tensorflow` +
   `tensorflow-hub` for a HydroSense-TL smoke test (currently the TL code
   path — `src/models/hydrosense_tl.py`, `_train_tl` in `train.py` — is
   written but has not been executed locally, unlike Base/SE which have
   been run end-to-end; it *is* installed and importable inside the Docker
   image built in step 45).

## Working method (for continuity)

Each unit of work: implement → smoke-test against synthetic data in
`.venv` → clean up generated test artifacts (they're gitignored, but delete
them anyway to keep the working tree clean) → `git add` the source files
only → commit with a message noting what was verified → `git push origin
main`. Commit messages are numbered "(step N/~110)" — continue that
numbering (currently at step 48) rather than restarting it.

Notebooks under `notebooks/` are executed (not just written) before commit
via `nbclient`, so the committed `.ipynb` files carry real output —
plots, subprocess logs, prediction/score JSON — from a run against the
synthetic dataset. Re-run with `jupyter nbconvert --execute --inplace
notebooks/<name>.ipynb` (or equivalent) after any change to keep outputs
current. Each notebook's heavier pipeline steps use their own scratch
directory (`notebooks/.tmp_nb0N/`, gitignored) so notebooks don't
interfere with each other or with `data/`.
