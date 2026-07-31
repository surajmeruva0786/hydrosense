#!/usr/bin/env python
"""Validate TimeGAN-synthesised segments against real data (README §9 step 4).

Usage:
    python -m src.augmentation.validate_synthetic \\
        --real_manifest data/processed/mel/manifest.csv \\
        --synthetic_manifest data/synthetic/mel/manifest.csv \\
        --class_name D \\
        --output_dir results/timegan_validation/D

Produces:
    - `pca.png`, `tsne.png` — 2D projections of real vs. synthetic segments
    - `scores.json` — discriminative score (post-hoc classifier accuracy,
      README §9) and predictive score (one-step-ahead RMSE, README §9)

Expects a 2D (freq, time) time-frequency representation (mel/log_mel/cqt/mfcc);
the raw `waveform` representation used by HydroSense-TL is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("augmentation.validate_synthetic")


def load_class_segments(
    manifest_path: str | Path,
    class_name: str,
    split: str | None = None,
    max_samples: int | None = None,
) -> np.ndarray:
    df = pd.read_csv(manifest_path)
    df = df[df["class_name"] == class_name]
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    if max_samples is not None:
        df = df.head(max_samples)
    return np.stack([np.load(p) for p in df["segment_path"]])


def flatten_for_embedding(segments: np.ndarray) -> np.ndarray:
    """(N, F, T) -> (N, F*T) time-averaged-and-flattened feature vector for PCA/t-SNE/discriminative scoring.

    Uses the per-frequency-bin time-mean and time-std (2*F features) rather
    than the full flattened array, which keeps the discriminative
    classifier's input dimensionality reasonable regardless of segment length.
    """
    mean_over_time = segments.mean(axis=2)
    std_over_time = segments.std(axis=2)
    return np.concatenate([mean_over_time, std_over_time], axis=1)


def pca_projection(real: np.ndarray, synthetic: np.ndarray, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    features = np.concatenate(
        [flatten_for_embedding(real), flatten_for_embedding(synthetic)], axis=0
    )
    proj = PCA(n_components=2, random_state=42).fit_transform(features)
    n_real = len(real)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:n_real, 0], proj[:n_real, 1], label="real", alpha=0.6, s=15)
    plt.scatter(proj[n_real:, 0], proj[n_real:, 1], label="synthetic (TimeGAN)", alpha=0.6, s=15)
    plt.legend()
    plt.title("PCA: real vs. synthetic")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def tsne_projection(real: np.ndarray, synthetic: np.ndarray, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    features = np.concatenate(
        [flatten_for_embedding(real), flatten_for_embedding(synthetic)], axis=0
    )
    n_samples = features.shape[0]
    perplexity = min(30, max(2, n_samples // 4))
    proj = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(
        features
    )
    n_real = len(real)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:n_real, 0], proj[:n_real, 1], label="real", alpha=0.6, s=15)
    plt.scatter(proj[n_real:, 0], proj[n_real:, 1], label="synthetic (TimeGAN)", alpha=0.6, s=15)
    plt.legend()
    plt.title("t-SNE: real vs. synthetic")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def discriminative_score(real: np.ndarray, synthetic: np.ndarray, seed: int = 42) -> float:
    """|test accuracy - 0.5| of a classifier trained to tell real from synthetic apart.

    0.0 = indistinguishable (ideal); 0.5 = trivially separable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    x = np.concatenate([flatten_for_embedding(real), flatten_for_embedding(synthetic)], axis=0)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synthetic))])

    if len(x) < 4:
        logger.warning("Too few samples (%d) for a reliable discriminative score.", len(x))
        return float("nan")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    return abs(accuracy - 0.5)


def predictive_score(real: np.ndarray, synthetic: np.ndarray, seed: int = 42) -> float:
    """One-step-ahead RMSE (README §9): fit a next-frame predictor on synthetic
    sequences, evaluate it on real sequences (train-on-synthetic, test-on-real,
    a.k.a. TSTR — lower is better)."""
    from sklearn.linear_model import Ridge

    def to_pairs(segments: np.ndarray):
        # (N, F, T) -> X: frame_t (N*(T-1), F), y: frame_{t+1} (N*(T-1), F)
        seqs = segments.transpose(0, 2, 1)  # (N, T, F)
        x = seqs[:, :-1, :].reshape(-1, seqs.shape[-1])
        y = seqs[:, 1:, :].reshape(-1, seqs.shape[-1])
        return x, y

    x_synth, y_synth = to_pairs(synthetic)
    x_real, y_real = to_pairs(real)

    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(x_synth, y_synth)
    y_pred = model.predict(x_real)
    rmse = float(np.sqrt(np.mean((y_pred - y_real) ** 2)))
    return rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real_manifest", type=str, required=True)
    parser.add_argument("--synthetic_manifest", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real = load_class_segments(
        args.real_manifest, args.class_name, split="train", max_samples=args.max_samples
    )
    synthetic = load_class_segments(
        args.synthetic_manifest, args.class_name, max_samples=args.max_samples
    )
    logger.info(
        "Loaded %d real and %d synthetic segments for class '%s'",
        len(real),
        len(synthetic),
        args.class_name,
    )

    pca_projection(real, synthetic, output_dir / "pca.png")
    tsne_projection(real, synthetic, output_dir / "tsne.png")

    scores = {
        "class_name": args.class_name,
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "discriminative_score": discriminative_score(real, synthetic, args.seed),
        "predictive_score_rmse": predictive_score(real, synthetic, args.seed),
    }
    logger.info("Scores: %s", scores)

    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
