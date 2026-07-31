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

## Status: 11 of 15 phases complete (steps 1-44 committed and pushed)

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
| 11 | CI/CD (GitHub Actions) | 🟡 in progress — ci.yml added; Dockerfile/docker-compose/Makefile/pre-commit still open |
| 12 | Notebooks (EDA, representations, TimeGAN validation, XAI analysis) | ⬜ not started |
| 13 | Docs polish (CONTRIBUTING, CHANGELOG, data/README, DEPLOYMENT, env.yml, README status update) | ⬜ not started |
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

Local dev venv at `Z:\hydrosense\.venv` has numpy/scipy/pandas/scikit-learn/
librosa/soundfile/torch(cpu)/shap/streamlit/pytest/ruff/black installed —
reuse it rather than reinstalling from scratch. `tensorflow`/
`tensorflow-hub` (needed only for HydroSense-TL / YAMNet) are **not yet
installed** there.

## What's left (in order)

1. **Finish Phase 11**: Dockerfile, docker-compose.yml, .dockerignore,
   Makefile, `.pre-commit-config.yaml`. Verify `docker build` succeeds.
2. **Phase 12**: four notebooks under `notebooks/` — EDA, representation
   comparison, TimeGAN validation walkthrough, XAI deep dive. Should import
   from `src/` rather than duplicating logic, and run against the synthetic
   dataset by default.
3. **Phase 13**: `CONTRIBUTING.md`, `CHANGELOG.md`, `data/README.md`
   (explains the ShipsEar access-request process + synthetic fallback),
   `DEPLOYMENT.md` (Streamlit Community Cloud / HF Spaces / Docker), and
   `environment.yml`. Update the main `README.md` "Results" section (§16) to
   be explicit that the numbers are targets pending a real training run, and
   correct the HydroSense-Base param count (README says "~480k"; actual
   `HydroSenseBase(num_classes=5)` is 422,341 params, `HydroSenseSE` is
   433,221 — see step 26).
4. **Phase 14**: reinstall the venv strictly from `requirements-dev.txt`
   (currently a hand-rolled subset was installed incrementally), run the
   full pytest suite + `ruff check` + `black --check` clean, run one
   complete pipeline pass on synthetic data end-to-end (preprocess -> train
   all 3 models -> TimeGAN -> evaluate -> explain -> streamlit boots), then
   tag `v0.1.0`.
5. Decide whether to also actually install/exercise `tensorflow` +
   `tensorflow-hub` for a HydroSense-TL smoke test (currently the TL code
   path — `src/models/hydrosense_tl.py`, `_train_tl` in `train.py` — is
   written but has not been executed, unlike Base/SE which have been run
   end-to-end).

## Working method (for continuity)

Each unit of work: implement → smoke-test against synthetic data in
`.venv` → clean up generated test artifacts (they're gitignored, but delete
them anyway to keep the working tree clean) → `git add` the source files
only → commit with a message noting what was verified → `git push origin
main`. Commit messages are numbered "(step N/~110)" — continue that
numbering (currently at step 44) rather than restarting it.
