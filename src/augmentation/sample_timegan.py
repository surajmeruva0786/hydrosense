#!/usr/bin/env python
"""Sample synthetic segments from trained TimeGANs and add them to the training manifest (README §9 step 5).

Usage:
    python -m src.augmentation.sample_timegan \\
        --checkpoints_dir data/synthetic \\
        --classes D E \\
        --n_samples 500 \\
        --representation mel \\
        --segment_length 10.0 \\
        --output_dir data/synthetic

For each class, samples `n_samples` synthetic 4-second clips
(`--clip_seconds`, matching training), concatenates as many as needed to
fill one `--segment_length` window, computes the same time-frequency
representation used for the real manifest, and writes
`<output_dir>/<representation>/manifest.csv` with `split=train, fold=-1` —
synthetic data is added to the training set only, never validation/test
(README §9 step 5, README §5.6).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.augmentation.timegan_networks import TimeGAN  # noqa: E402
from src.augmentation.train_timegan import unframe_waveform  # noqa: E402
from src.preprocessing.representations import compute_representation  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("augmentation.sample_timegan")


def load_timegan(checkpoint_path: str | Path, device: str = "cpu") -> tuple[TimeGAN, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TimeGAN(feature_dim=ckpt["feature_dim"], hidden_dim=ckpt["hidden_dim"], noise_dim=ckpt["noise_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def sample_waveforms(model: TimeGAN, ckpt: dict, n_samples: int, device: str = "cpu") -> np.ndarray:
    """Generate `n_samples` synthetic clips of `ckpt['clip_seconds']`, denormalised to waveform range."""
    norm_stats = ckpt["norm_stats"]
    seq_len = norm_stats["seq_len"]
    frames = model.generate(n_samples, seq_len, device=device).cpu().numpy()  # (N, seq_len, frame_samples) in [0,1]

    data_min, data_max = norm_stats["data_min"], norm_stats["data_max"]
    frames = frames * (data_max - data_min + 1e-8) + data_min
    return np.stack([unframe_waveform(f) for f in frames])


def assemble_segments(clips: np.ndarray, sample_rate: int, segment_length: float, seed: int = 42) -> np.ndarray:
    """Concatenate randomly-ordered synthetic clips end-to-end into fixed-length segments."""
    rng = np.random.default_rng(seed)
    target_samples = int(segment_length * sample_rate)
    clip_samples = clips.shape[1]
    clips_per_segment = math.ceil(target_samples / clip_samples)

    n_segments = max(1, len(clips) // clips_per_segment)
    segments = []
    order = rng.permutation(len(clips))
    for i in range(n_segments):
        idx = order[i * clips_per_segment : (i + 1) * clips_per_segment]
        if len(idx) < clips_per_segment:
            break
        concatenated = np.concatenate(clips[idx])
        segments.append(concatenated[:target_samples])
    return np.stack(segments) if segments else np.zeros((0, target_samples), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints_dir", type=str, default="data/synthetic")
    parser.add_argument("--classes", nargs="+", default=["D", "E"])
    parser.add_argument("--class_names", type=str, default="A,B,C,D,E", help="Full ordered class list -> label index")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of synthetic clips to draw per class")
    parser.add_argument("--representation", type=str, default="mel", choices=["mel", "log_mel", "cqt", "mfcc", "waveform"])
    parser.add_argument("--segment_length", type=float, default=10.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    class_to_idx = {name: idx for idx, name in enumerate(args.class_names.split(","))}

    output_dir = Path(args.output_dir)
    rows = []

    for class_name in args.classes:
        ckpt_path = Path(args.checkpoints_dir) / f"timegan_{class_name}.pt"
        if not ckpt_path.exists():
            logger.warning("No TimeGAN checkpoint for class '%s' at %s; skipping. Run train_timegan.py first.", class_name, ckpt_path)
            continue

        model, ckpt = load_timegan(ckpt_path, device=args.device)
        clips = sample_waveforms(model, ckpt, args.n_samples, device=args.device)
        segments = assemble_segments(clips, ckpt.get("sample_rate", args.sample_rate), args.segment_length, args.seed)
        logger.info("Class '%s': sampled %d clips -> %d synthetic %.1fs segments", class_name, len(clips), len(segments), args.segment_length)

        class_dir = output_dir / args.representation / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segments):
            rep = compute_representation(seg, args.sample_rate, args.representation)
            segment_path = class_dir / f"timegan_{class_name}_{i:04d}.npy"
            np.save(segment_path, rep)
            rows.append(
                {
                    "segment_path": str(segment_path),
                    "recording_id": f"timegan_{class_name}_{i:04d}",
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                    "split": "train",
                    "fold": -1,
                }
            )

    if not rows:
        logger.warning("No synthetic segments were generated; nothing written.")
        return

    manifest_dir = output_dir / args.representation
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info("Wrote %d synthetic segments to %s", len(rows), manifest_path)


if __name__ == "__main__":
    main()
