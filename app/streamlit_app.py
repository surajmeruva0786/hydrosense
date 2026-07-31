#!/usr/bin/env python
"""HydroSense interactive demo (README §12).

    streamlit run app/streamlit_app.py

Upload a `.wav` clip, pick a trained checkpoint, and get: the mel-spectrogram,
top-3 class predictions with confidence bars, a toggleable Grad-CAM overlay,
a SHAP per-band attribution chart, and audio playback — the artifact intended
for live demonstration to NSTL scientists (README §12).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components import (  # noqa: E402
    find_available_checkpoints,
    plot_gradcam_overlay,
    plot_shap_bar,
    plot_spectrogram,
    render_confidence_bars,
)
from src.data.dataset import ManifestDataset  # noqa: E402
from src.data.transforms import build_default_transform  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.preprocessing.representations import (  # noqa: E402
    compute_representation,
    expected_num_frames,
)
from src.preprocessing.segmentation import segment_and_filter_silence  # noqa: E402
from src.preprocessing.signal_conditioning import condition_signal  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.xai.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from src.xai.shap_explainer import (  # noqa: E402
    build_background,
    compute_shap_values,
    per_band_attribution,
)
from src.xai.spectral_peaks import acoustic_sanity_check  # noqa: E402

st.set_page_config(page_title="HydroSense", page_icon="🌊", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def _load_model(checkpoint_path: str):
    run_dir = Path(checkpoint_path).parent
    if not (run_dir / "config.yaml").exists():
        run_dir = run_dir.parent
    config = load_config(run_dir / "config.yaml")
    model = build_model(config)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()
    return model, config


def _prepare_clip(audio_bytes: bytes, config: dict):
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    waveform, orig_sr = sf.read(tmp_path, dtype="float32", always_2d=False)
    conditioned, sr = condition_signal(np.asarray(waveform), orig_sr, config["sample_rate"])
    segments = segment_and_filter_silence(
        conditioned, sr, segment_length_s=config["segment_length"], overlap=0.0
    )

    if not segments:
        return None, tmp_path

    rep = compute_representation(segments[0].waveform, sr, config["representation"])
    return rep, tmp_path


def main() -> None:
    st.title("🌊 HydroSense")
    st.caption(
        "Explainable passive sonar vessel classification — Grad-CAM + SHAP on every prediction."
    )

    with st.sidebar:
        st.header("Model")
        checkpoints = find_available_checkpoints()
        if not checkpoints:
            st.warning(
                "No trained checkpoints found under `runs/`. Train one first, e.g.:\n\n"
                "```\npython -m src.training.train --model hydrosense_se "
                "--representation mel --folds 5 --output_dir runs/hydrosense_se_mel\n```"
            )
            checkpoint_path = st.text_input("...or enter a checkpoint path manually", value="")
        else:
            checkpoint_path = st.selectbox("Checkpoint", checkpoints)

        compute_shap = st.checkbox("Compute SHAP attribution (slower)", value=False)
        show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)

    if not checkpoint_path or not Path(checkpoint_path).exists():
        st.info("Select or provide a valid checkpoint in the sidebar to begin.")
        return

    model, config = _load_model(checkpoint_path)
    st.success(
        f"Loaded **{config['model']}** ({config['representation']} representation, {config['num_classes']} classes)"
    )

    uploaded = st.file_uploader("Upload a hydrophone recording (.wav)", type=["wav"])
    if uploaded is None:
        st.stop()

    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    rep, tmp_path = _prepare_clip(audio_bytes, config)
    if rep is None:
        st.error(f"No usable (non-silent) {config['segment_length']}s segment found in this clip.")
        return

    num_frames = expected_num_frames(int(config["sample_rate"] * config["segment_length"]))
    transform = build_default_transform(target_frames=num_frames)
    x = torch.as_tensor(transform(rep), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()

    class_names = config["class_names"]
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [{"class_name": class_names[int(i)], "confidence": float(probs[i])} for i in top3_idx]
    predicted_class = top3[0]["class_name"]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(plot_spectrogram(rep, title=f"{config['representation']} spectrogram"))
    with col2:
        render_confidence_bars(top3)
        st.json({"predicted_class": predicted_class, "top3": top3})

    if show_gradcam:
        x_grad = x.clone().requires_grad_(True)
        cam = GradCAM(model, model.target_layer)
        heatmap, explained_class = cam(x_grad, class_idx=int(top3_idx[0]))
        cam.remove()

        overlay = overlay_heatmap(x[0, 0].numpy(), heatmap)
        st.pyplot(plot_gradcam_overlay(overlay, class_names[explained_class]))

        sanity = acoustic_sanity_check(x[0, 0].numpy(), heatmap, sample_rate=config["sample_rate"])
        with st.expander("Acoustic sanity check (README §11.3)"):
            st.json(sanity)

    if compute_shap:
        manifest_path = Path("data/processed") / config["representation"] / "manifest.csv"
        if not manifest_path.exists():
            st.warning(
                f"No manifest found at {manifest_path} to draw a SHAP background sample from."
            )
        else:
            with st.spinner("Computing SHAP attribution..."):
                background_ds = ManifestDataset(manifest_path, transform=transform, split="train")
                background = build_background(
                    background_ds, n_background=min(30, len(background_ds))
                )
                shap_values = compute_shap_values(model, background, x.detach())
                profile = per_band_attribution(shap_values, int(top3_idx[0]))
            st.pyplot(plot_shap_bar(profile, predicted_class))


if __name__ == "__main__":
    main()
