# HydroSense — Streamlit demo image (README §12, §13)
#
# Builds a container that serves the interactive demo app. Training/
# preprocessing CLIs are also available inside the image via `docker run
# --entrypoint python hydrosense -m src.preprocessing.run ...` etc.
FROM python:3.11-slim AS base

# libsndfile1: required by soundfile/librosa for audio I/O.
# libgomp1: OpenMP runtime required by torch/scikit-learn at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# torch/torchaudio from PyPI pull in full CUDA toolkits (~8GB) that are dead
# weight in this CPU-only serving image; install the CPU-only wheels from
# PyTorch's own index first, then the rest of requirements.txt from PyPI
# (pip sees torch/torchaudio already satisfied and skips them).
RUN pip install --no-cache-dir torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 hydrosense \
    && chown -R hydrosense:hydrosense /app
USER hydrosense

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
