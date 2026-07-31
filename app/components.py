"""Reusable Streamlit widgets/plot helpers for `app/streamlit_app.py` (README §12)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def find_available_checkpoints(runs_dir: str | Path = "runs") -> list[str]:
    """List every `runs/**/best.ckpt` (torch) or `.../best.weights.h5` (Keras TL) found."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    torch_ckpts = [str(p) for p in runs_dir.glob("*/best.ckpt")]
    tl_ckpts = [str(p) for p in runs_dir.glob("*/best.weights.h5")]
    return sorted(torch_ckpts + tl_ckpts)


def plot_spectrogram(spectrogram: np.ndarray, title: str = "Mel-Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("time frame")
    ax.set_ylabel("frequency bin")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def plot_gradcam_overlay(overlay_rgb: np.ndarray, class_name: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.imshow(overlay_rgb, aspect="auto", origin="lower")
    ax.set_title(f"Grad-CAM — predicted class {class_name}")
    ax.set_xlabel("time frame")
    ax.set_ylabel("frequency bin")
    fig.tight_layout()
    return fig


def plot_shap_bar(profile: np.ndarray, class_name: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(profile)), profile, color="#3b7dd8")
    ax.set_title(f"SHAP per-mel-band attribution — {class_name}")
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("mean |SHAP value|")
    fig.tight_layout()
    return fig


def render_confidence_bars(top3: list[dict]) -> None:
    st.subheader("Top-3 Predictions")
    for entry in top3:
        st.write(f"**{entry['class_name']}** — {entry['confidence'] * 100:.1f}%")
        st.progress(min(1.0, float(entry["confidence"])))
