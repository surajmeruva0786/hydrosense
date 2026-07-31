#!/usr/bin/env python
"""Held-out test-set evaluation CLI (README §10, §14).

Usage:
    python -m src.evaluation.evaluate \\
        --checkpoint runs/hydrosense_se_mel/best.ckpt \\
        --test_split data/splits/test.csv \\
        --output_dir results/hydrosense_se_mel

Evaluates once, only after model selection is complete (README §10) — this
script does not touch training or validation data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ManifestDataset  # noqa: E402
from src.data.transforms import build_default_transform  # noqa: E402
from src.evaluation.report import write_report  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.preprocessing.representations import expected_num_frames  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402

logger = get_logger("evaluation.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_split", type=str, default="data/splits/test.csv")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent if checkpoint_path.parent.name != f"fold_{0}" else checkpoint_path.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected a config.yaml alongside the checkpoint at {config_path}")

    config = load_config(config_path)
    logger.info("Evaluating %s (%s / %s) using checkpoint %s", config["model"], config["representation"], args.checkpoint, args.checkpoint)

    manifest_path = Path(args.processed_dir) / config["representation"] / "manifest.csv"
    dataset = ManifestDataset(
        manifest_path,
        transform=build_default_transform(expected_num_frames(int(config["sample_rate"] * config["segment_length"]))),
        split="test",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No test segments found in {manifest_path}. Did preprocessing run with test recordings?")

    if Path(args.test_split).exists():
        expected_recordings = set(pd.read_csv(args.test_split)["recording_id"])
        actual_recordings = set(dataset.recording_ids.tolist())
        if expected_recordings and expected_recordings != actual_recordings:
            logger.warning(
                "Test-split recording IDs differ from manifest test segments (expected=%d, actual=%d)",
                len(expected_recordings), len(actual_recordings),
            )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(config).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, config["class_names"])
    logger.info(
        "Test results: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f top2_acc=%.4f",
        metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["top2_accuracy"],
    )

    paths = write_report(metrics, args.output_dir)
    logger.info("Wrote %s and %s", paths["metrics"], paths["report"])


if __name__ == "__main__":
    main()
