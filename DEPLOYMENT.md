# Deployment

This covers deploying the interactive demo (`app/streamlit_app.py`,
README §12). The demo needs a trained checkpoint on disk — either bake one
into the deployment target or generate one at startup against the
synthetic dataset (README §14 usage examples).

## Option 1: Streamlit Community Cloud (simplest)

1. Push this repository to GitHub (public, or a private repo you connect
   your Streamlit Cloud account to).
2. At [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at your fork/repo, branch `main`, main file path
   `app/streamlit_app.py`.
3. Streamlit Cloud installs from `requirements.txt` automatically. No
   `Dockerfile` involved on this path.
4. **Provide a checkpoint.** The app looks for trained checkpoints under
   `runs/` (`app/components.py:find_available_checkpoints`). Either:
   - Commit a small pre-trained checkpoint under `runs/<name>/best.ckpt`
     (outside `.gitignore`'s `runs/*` — you'll need to `git add -f` it, or
     adjust `.gitignore` for that one path) and reference it in the app, or
   - Add a `packages.txt` / startup step that runs the synthetic-data
     smoke pipeline (README §14 "Usage") before the app starts, so a demo
     checkpoint exists at boot.
5. Community Cloud's free tier is CPU-only and has resource limits —
   `tensorflow`/`tensorflow-hub` (only needed for `HydroSense-TL`) can push
   past the free tier's memory ceiling. If you hit that, deploy with only
   `HydroSense-Base`/`HydroSense-SE` checkpoints (pure PyTorch, no
   TensorFlow import needed at inference time for those two models).

## Option 2: Hugging Face Spaces

1. Create a new Space, SDK = **Streamlit**.
2. Push this repository's contents to the Space's git remote (Spaces are
   themselves git repos):
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space-name>
   git push hf main
   ```
3. Spaces auto-detects `app/streamlit_app.py`? No — Spaces expects the
   entry point at the repo root by default. Either:
   - Set `app_file: app/streamlit_app.py` in the Space's `README.md`
     front-matter (the YAML block Spaces reads for config), or
   - Add a root-level `streamlit_app.py` that does
     `exec(open("app/streamlit_app.py").read())`.
4. Same checkpoint-availability caveat as Option 1 applies.
5. For anything beyond the free CPU tier (e.g. exercising `HydroSense-TL`
   with TensorFlow/YAMNet), select a paid hardware tier in the Space
   settings.

## Option 3: Docker (self-hosted / any container platform)

The repository ships a `Dockerfile` and `docker-compose.yml` that build and
serve the Streamlit app.

```bash
# Build
docker build -t hydrosense:latest .

# Run (maps ./data, ./runs, ./results as volumes so checkpoints and
# datasets outside the image are visible to the container)
docker compose up --build
```

The app is then served at `http://localhost:8501`.

To push to a container registry and deploy on any platform that runs
Docker images (Fly.io, Render, AWS ECS/Fargate, Google Cloud Run, Azure
Container Apps, a bare VM, etc.):

```bash
docker tag hydrosense:latest <registry>/<repo>:<tag>
docker push <registry>/<repo>:<tag>
```

Then point the target platform at that image, exposing container port
`8501`. Mount or bake in a `runs/` directory containing at least one
trained checkpoint — without one, the app starts but has nothing to
classify with.

### Producing a checkpoint to ship

For a real deployment, train against the real ShipsEar corpus (README §6,
`data/README.md`). For a demo/smoke deployment without dataset access:

```bash
python scripts/generate_synthetic_dataset.py --output_dir data/raw/shipsear_synthetic
python -m src.preprocessing.run --input_dir data/raw/shipsear_synthetic --output_dir data/processed --sr 16000 --segment_length 10.0 --overlap 0.5
python -m src.training.train --model hydrosense_base --representation mel --folds 2 --epochs 10 --batch_size 16 --output_dir runs/hydrosense_base_mel
```

This produces `runs/hydrosense_base_mel/best.ckpt` — but remember: a model
trained only on the synthetic generator has learned synthetic acoustic
patterns, not real vessel acoustics (`data/README.md`). Label any such
deployment clearly as a **pipeline demo**, not a working classifier.

## Environment variables / secrets

The app itself takes no required environment variables. `.streamlit/
config.toml` sets the theme and disables usage-stats collection;
`.streamlit/secrets.toml` (gitignored) is where you'd add any secrets if
you extend the app to call an external service — none are needed for the
base functionality.

## Health checks

Both the `Dockerfile` and `docker-compose.yml` define a health check against
Streamlit's built-in `/_stcore/health` endpoint. If you're fronting the
container with a load balancer or orchestrator, point its health check at
`GET /_stcore/health` on port `8501` (expects HTTP 200).
