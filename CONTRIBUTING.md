# Contributing to HydroSense

Thanks for your interest in improving HydroSense. This document covers the
practical mechanics of contributing; for the project's scope and design
rationale see `README.md`, and for current build status see `PROGRESS.md`.

## Getting set up

```bash
git clone https://github.com/<your-username>/hydrosense.git
cd hydrosense
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

`requirements-dev.txt` installs `requirements.txt` plus lint/test/notebook
tooling (`ruff`, `black`, `mypy`, `pytest`, `pre-commit`, `ipykernel`,
`jupyterlab`).

Since the real ShipsEar dataset is access-gated (see `data/README.md`),
generate the synthetic corpus to develop and test against:

```bash
python scripts/generate_synthetic_dataset.py --output_dir data/raw/shipsear_synthetic
```

## Development workflow

1. **Branch** from `main`.
2. **Make focused changes.** Prefer several small, reviewable commits over one
   large one — see the commit message convention below.
3. **Run the checks locally** before pushing (the same ones CI runs, see
   `.github/workflows/ci.yml`):

   ```bash
   make lint       # ruff check
   make format     # black + ruff --fix
   make typecheck  # mypy src
   make test       # pytest
   ```

   Or run the underlying commands directly if you don't have `make`
   available (e.g. native Windows without WSL/Git Bash):

   ```bash
   ruff check src tests app scripts
   black --check src tests app scripts
   mypy src
   pytest
   ```

4. **Smoke-test end-to-end** against the synthetic dataset if you touched
   preprocessing, training, augmentation, or XAI code — the CI pipeline does
   this too (`.github/workflows/ci.yml`), but running it locally first saves
   a round trip:

   ```bash
   python scripts/generate_synthetic_dataset.py --output_dir data/raw/shipsear_ci --counts "A=3,B=3,C=3,D=3,E=3" --min_duration 15 --max_duration 20
   python -m src.preprocessing.run --input_dir data/raw/shipsear_ci --output_dir data/processed --sr 16000 --segment_length 10.0 --overlap 0.5 --representation mel --n_folds 2
   python -m src.training.train --model hydrosense_base --representation mel --folds 2 --epochs 1 --batch_size 4 --output_dir runs/ci_smoke --max_folds_to_run 1
   ```

5. **Open a pull request** against `main`. Describe *why* the change is
   needed, not just what changed — the diff already shows the what.

## Commit messages

Use a short imperative subject line with a scope prefix, e.g.:

```
feat(models): add dropout schedule to HydroSense-SE
fix(preprocessing): handle mono/stereo downmix edge case
docs: clarify TimeGAN checkpoint format
test(xai): cover Grad-CAM on empty batch
```

Keep commits atomic — one logical change per commit. This keeps the history
bisectable when a regression needs to be tracked down.

## Code style

- Formatting and linting are enforced by `black` and `ruff` (config in
  `pyproject.toml`); `pre-commit` runs both automatically on `git commit`.
- Type hints are expected on new public functions; `mypy` is advisory
  (`ignore_missing_imports = true`) rather than strict, since several
  dependencies (librosa, shap, tensorflow_hub) lack complete stubs.
- Follow the existing module layout: `src/<area>/<thing>.py` with a matching
  `tests/test_<thing>.py`. CLI entry points (`run.py`, `train.py`,
  `evaluate.py`, `explain.py`, `train_timegan.py`, `sample_timegan.py`,
  `validate_synthetic.py`) each expose a `main()` guarded by
  `if __name__ == "__main__":` and are invoked as `python -m src.<area>.<thing>`.

## Tests

New functionality should come with tests under `tests/`. The existing suite
(`pytest`) covers dataset loading, transforms, preprocessing, model
forward-passes, metrics, augmentation, and Grad-CAM — follow the pattern of
the closest existing test file. Tests run against small synthetic fixtures
(see `tests/conftest.py`), not the real ShipsEar corpus, and should stay
fast (seconds, not minutes) so CI remains usable.

## Reporting issues

Open a GitHub issue with:

- What you expected to happen vs. what happened.
- Steps to reproduce, including whether you were using the synthetic
  dataset or a real ShipsEar corpus.
- The output of `pip freeze` if the issue looks environment-related.

## Questions

For anything not covered here, open an issue or see the contact details in
`README.md` §24.
