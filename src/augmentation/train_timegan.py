#!/usr/bin/env python
"""Train a per-class TimeGAN on raw waveform segments (README §9).

Usage:
    python -m src.augmentation.train_timegan \\
        --classes D E \\
        --epochs 5000 \\
        --output_dir data/synthetic

Trains one TimeGAN per class listed in `--classes` (typically the
under-represented categories, README §6) on 4-second raw waveform segments
extracted from that class's recordings, following the paper's 3-phase
recipe: (1) embedder/recovery autoencoder pretraining, (2) supervisor
pretraining on real latent dynamics, (3) joint adversarial training.
Checkpoints are written to `<output_dir>/timegan_<class>.pt` for
`sample_timegan.py` to load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.augmentation.timegan_networks import TimeGAN  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("augmentation.train_timegan")

DEFAULT_FRAME_SAMPLES = 160  # 10 ms @ 16 kHz
DEFAULT_CLIP_SECONDS = 4.0


def frame_waveform(waveform: np.ndarray, frame_samples: int = DEFAULT_FRAME_SAMPLES) -> np.ndarray:
    """Chop a 1D waveform into non-overlapping (seq_len, frame_samples) frames.

    Keeps GRU sequence length tractable (e.g. 400 steps for a 4s/16kHz clip
    with 160-sample frames) instead of feeding ~64000 raw samples directly.
    """
    n_frames = len(waveform) // frame_samples
    trimmed = waveform[: n_frames * frame_samples]
    return trimmed.reshape(n_frames, frame_samples)


def unframe_waveform(frames: np.ndarray) -> np.ndarray:
    """Inverse of `frame_waveform`: (seq_len, frame_samples) -> 1D waveform."""
    return frames.reshape(-1)


def extract_class_clips(
    metadata_csv: str | Path,
    class_name: str,
    sample_rate: int = 16000,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    seed: int = 42,
) -> np.ndarray:
    """Extract non-overlapping `clip_seconds` raw waveform clips for one class's recordings.

    Returns an array of shape (n_clips, clip_samples) in [-1, 1] (waveforms
    are already peak-normalised by `src.data.synthetic` / real recordings).
    """
    import pandas as pd
    import soundfile as sf

    from src.preprocessing.signal_conditioning import condition_signal

    metadata_csv = Path(metadata_csv)
    metadata = pd.read_csv(metadata_csv)
    rows = metadata[metadata["class_name"] == class_name]
    if rows.empty:
        raise ValueError(f"No recordings found for class '{class_name}' in {metadata_csv}")

    clip_samples = int(clip_seconds * sample_rate)
    clips = []
    for _, row in rows.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.is_absolute() and not file_path.exists():
            file_path = metadata_csv.parent / file_path.name
        waveform, orig_sr = sf.read(str(file_path), dtype="float32", always_2d=False)
        conditioned, _ = condition_signal(np.asarray(waveform), orig_sr, sample_rate)

        n_clips = len(conditioned) // clip_samples
        for i in range(n_clips):
            clips.append(conditioned[i * clip_samples : (i + 1) * clip_samples])

    if not clips:
        raise ValueError(f"No {clip_seconds}s clips could be extracted for class '{class_name}'")
    return np.stack(clips)


def _train_one_class(
    clips: np.ndarray,
    frame_samples: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> tuple[TimeGAN, dict]:
    set_seed(seed)

    data_min, data_max = float(clips.min()), float(clips.max())
    norm = (clips - data_min) / (data_max - data_min + 1e-8)  # -> [0, 1] for the sigmoid-activated networks

    framed = np.stack([frame_waveform(c, frame_samples) for c in norm])  # (N, seq_len, frame_samples)
    x = torch.as_tensor(framed, dtype=torch.float32, device=device)
    n_samples, seq_len, feature_dim = x.shape

    model = TimeGAN(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    opt_er = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr)
    opt_gs = torch.optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    def batches():
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            yield x[idx]

    # --- Phase 1: embedder/recovery autoencoder pretraining ---
    phase1_epochs = max(1, epochs // 3)
    for epoch in range(phase1_epochs):
        for x_real in batches():
            h = model.embedder(x_real)
            x_tilde = model.recovery(h)
            loss = mse(x_tilde, x_real)
            opt_er.zero_grad()
            loss.backward()
            opt_er.step()
        if epoch % max(1, phase1_epochs // 5) == 0:
            logger.info("  [phase 1/3] epoch %d/%d recon_loss=%.5f", epoch + 1, phase1_epochs, loss.item())

    # --- Phase 2: supervisor pretraining on real latent dynamics ---
    phase2_epochs = max(1, epochs // 3)
    for epoch in range(phase2_epochs):
        for x_real in batches():
            with torch.no_grad():
                h = model.embedder(x_real)
            h_hat_supervise = model.supervisor(h[:, :-1, :])
            loss = mse(h_hat_supervise, h[:, 1:, :])
            opt_gs.zero_grad()
            loss.backward()
            opt_gs.step()
        if epoch % max(1, phase2_epochs // 5) == 0:
            logger.info("  [phase 2/3] epoch %d/%d supervised_loss=%.5f", epoch + 1, phase2_epochs, loss.item())

    # --- Phase 3: joint adversarial training ---
    phase3_epochs = max(1, epochs - phase1_epochs - phase2_epochs)
    for epoch in range(phase3_epochs):
        for x_real in batches():
            b, s, _ = x_real.shape
            z = model.sample_noise(b, s, device)

            # Generator + supervisor step
            e_hat = model.generator(z)
            h_hat = model.supervisor(e_hat)
            x_hat = model.recovery(h_hat)
            d_fake = model.discriminator(h_hat)

            with torch.no_grad():
                h_real = model.embedder(x_real)
            h_hat_supervise = model.supervisor(h_real[:, :-1, :])

            g_loss_adv = bce(d_fake, torch.ones_like(d_fake))
            g_loss_supervised = mse(h_hat_supervise, h_real[:, 1:, :])
            g_loss_moment = (
                torch.abs(x_hat.mean(dim=(0, 1)) - x_real.mean(dim=(0, 1))).mean()
                + torch.abs(x_hat.std(dim=(0, 1)) - x_real.std(dim=(0, 1))).mean()
            )
            g_loss = g_loss_adv + 100 * torch.sqrt(g_loss_supervised) + 100 * g_loss_moment

            opt_gs.zero_grad()
            g_loss.backward()
            opt_gs.step()

            # Embedder/recovery joint fine-tune (reconstruction + small supervised term)
            h = model.embedder(x_real)
            x_tilde = model.recovery(h)
            h_hat_supervise = model.supervisor(h[:, :-1, :].detach())
            er_loss = mse(x_tilde, x_real) + 0.1 * mse(h_hat_supervise, h[:, 1:, :].detach())
            opt_er.zero_grad()
            er_loss.backward()
            opt_er.step()

            # Discriminator step
            with torch.no_grad():
                h_real = model.embedder(x_real)
                e_hat = model.generator(z)
                h_hat = model.supervisor(e_hat)
            d_real = model.discriminator(h_real)
            d_fake = model.discriminator(h_hat)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))

            # Only update D when it isn't already dominating (standard TimeGAN heuristic)
            if d_loss.item() > 0.15:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

        if epoch % max(1, phase3_epochs // 10) == 0:
            logger.info(
                "  [phase 3/3] epoch %d/%d g_loss=%.4f d_loss=%.4f", epoch + 1, phase3_epochs, g_loss.item(), d_loss.item()
            )

    norm_stats = {"data_min": data_min, "data_max": data_max, "frame_samples": frame_samples, "seq_len": seq_len}
    return model, norm_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", default=["D", "E"])
    parser.add_argument("--metadata_csv", type=str, default="data/raw/shipsear/metadata.csv")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--frame_samples", type=int, default=DEFAULT_FRAME_SAMPLES)
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in args.classes:
        logger.info("=== Training TimeGAN for class '%s' ===", class_name)
        clips = extract_class_clips(args.metadata_csv, class_name, args.sample_rate, args.clip_seconds, args.seed)
        logger.info("Extracted %d clips of %.1fs for class '%s'", len(clips), args.clip_seconds, class_name)

        model, norm_stats = _train_one_class(
            clips,
            frame_samples=args.frame_samples,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=min(args.batch_size, max(2, len(clips))),
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

        ckpt_path = output_dir / f"timegan_{class_name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_dim": model.feature_dim,
                "hidden_dim": model.hidden_dim,
                "noise_dim": model.noise_dim,
                "norm_stats": norm_stats,
                "class_name": class_name,
                "sample_rate": args.sample_rate,
                "clip_seconds": args.clip_seconds,
            },
            ckpt_path,
        )
        logger.info("Saved TimeGAN checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
