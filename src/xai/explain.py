#!/usr/bin/env python
"""Generate a full XAI report for a single audio clip (README §11, §14).

Usage:
    python -m src.xai.explain \\
        --checkpoint runs/hydrosense_se_mel/best.ckpt \\
        --audio_file data/raw/shipsear/sample_001.wav \\
        --output_dir results/explanations/sample_001

Writes, into `--output_dir`:
    - `prediction.json`      top-3 classes + confidences
    - `gradcam_overlay.png`  README §11.1 Grad-CAM heatmap over the spectrogram
    - `spectral_peaks.json`  README §11.3 acoustic sanity check
    - `shap_bar.png` + `shap_values.json`  README §11.2 (skipped with `--skip_shap`,
      or automatically if no training manifest is available to draw a SHAP background from)

This is the same computation the Streamlit demo (README §12) calls per upload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.transforms import build_default_transform  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.preprocessing.representations import compute_representation, expected_num_frames  # noqa: E402
from src.preprocessing.segmentation import segment_and_filter_silence  # noqa: E402
from src.preprocessing.signal_conditioning import condition_signal  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.xai.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from src.xai.spectral_peaks import acoustic_sanity_check  # noqa: E402

logger = get_logger("xai.explain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Used to draw a SHAP background sample")
    parser.add_argument("--skip_shap", action="store_true")
    parser.add_argument("--n_shap_background", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_and_prepare_clip(audio_file: str, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (representation_array, raw_conditioned_segment_waveform) for the clip's
    first usable segment."""
    import soundfile as sf

    waveform, orig_sr = sf.read(audio_file, dtype="float32", always_2d=False)
    conditioned, sr = condition_signal(np.asarray(waveform), orig_sr, config["sample_rate"])

    segments = segment_and_filter_silence(
        conditioned, sr, segment_length_s=config["segment_length"], overlap=0.0
    )
    if not segments:
        raise ValueError(f"No usable (non-silent) {config['segment_length']}s segment found in {audio_file}")

    seg = segments[0].waveform
    rep = compute_representation(seg, sr, config["representation"])
    return rep, seg


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent if (checkpoint_path.parent / "config.yaml").exists() else checkpoint_path.parent.parent
    config = load_config(run_dir / "config.yaml")

    model = build_model(config).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    rep, _raw_segment = load_and_prepare_clip(args.audio_file, config)
    num_frames = expected_num_frames(int(config["sample_rate"] * config["segment_length"]))
    transform = build_default_transform(target_frames=num_frames)
    x = torch.as_tensor(transform(rep), dtype=torch.float32).unsqueeze(0).to(args.device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    class_names = config["class_names"]
    top3_idx = probs.argsort()[::-1][:3]
    prediction = {
        "predicted_class": class_names[int(top3_idx[0])],
        "top3": [{"class_name": class_names[int(i)], "confidence": float(probs[i])} for i in top3_idx],
    }
    with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)
    logger.info("Prediction: %s", prediction)

    x_grad = x.clone().requires_grad_(True)
    cam = GradCAM(model, model.target_layer)
    heatmap, explained_class = cam(x_grad, class_idx=int(top3_idx[0]))
    cam.remove()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = overlay_heatmap(x[0, 0].cpu().numpy(), heatmap)
    plt.figure(figsize=(8, 4))
    plt.imshow(overlay, aspect="auto", origin="lower")
    plt.title(f"Grad-CAM: predicted class {class_names[explained_class]}")
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_overlay.png", dpi=120)
    plt.close()

    sanity = acoustic_sanity_check(x[0, 0].detach().cpu().numpy(), heatmap, sample_rate=config["sample_rate"])
    with (output_dir / "spectral_peaks.json").open("w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)
    logger.info("Acoustic sanity check: %s", sanity)

    manifest_path = Path(args.processed_dir) / config["representation"] / "manifest.csv"
    if args.skip_shap or not manifest_path.exists():
        logger.info("Skipping SHAP (skip_shap=%s, manifest exists=%s)", args.skip_shap, manifest_path.exists())
        return

    from src.data.dataset import ManifestDataset
    from src.xai.shap_explainer import build_background, compute_shap_values, per_band_attribution

    background_ds = ManifestDataset(manifest_path, transform=transform, split="train")
    if len(background_ds) < 2:
        logger.warning("Not enough training segments (%d) to build a SHAP background; skipping SHAP.", len(background_ds))
        return

    background = build_background(background_ds, n_background=min(args.n_shap_background, len(background_ds)))
    shap_values = compute_shap_values(model, background, x.detach())
    profile = per_band_attribution(shap_values, explained_class)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(profile)), profile)
    plt.title(f"SHAP per-mel-band attribution: {class_names[explained_class]}")
    plt.xlabel("frequency bin")
    plt.ylabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=120)
    plt.close()

    with (output_dir / "shap_values.json").open("w", encoding="utf-8") as f:
        json.dump({"class_name": class_names[explained_class], "per_band_importance": profile.tolist()}, f, indent=2)

    logger.info("Wrote full XAI report to %s", output_dir)


if __name__ == "__main__":
    main()
