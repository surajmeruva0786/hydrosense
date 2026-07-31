#!/usr/bin/env python
"""Generate a small synthetic ShipsEar-shaped corpus for local development, CI, and demos.

The real ShipsEar dataset must be requested from the University of Vigo
(README §6) and cannot be redistributed here. This script produces
physically-motivated but fully synthetic recordings with the same 5-class
layout (`src.data.synthetic`), so every stage of the pipeline —
preprocessing, training, TimeGAN, evaluation, XAI, the Streamlit demo — can
be exercised without the real corpus.

Usage:
    python scripts/generate_synthetic_dataset.py \
        --output_dir data/raw/shipsear_synthetic \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_dataset  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("generate_synthetic_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default="data/raw/shipsear_synthetic")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--min_duration", type=float, default=20.0)
    parser.add_argument("--max_duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--counts",
        type=str,
        default="A=6,B=8,C=10,D=5,E=4",
        help="Comma-separated CLASS=count pairs, e.g. 'A=6,B=8,C=10,D=5,E=4'.",
    )
    return parser.parse_args()


def parse_counts(spec: str) -> dict[str, int]:
    counts = {}
    for pair in spec.split(","):
        cls, n = pair.split("=")
        counts[cls.strip()] = int(n)
    return counts


def main() -> None:
    args = parse_args()
    counts = parse_counts(args.counts)

    logger.info("Generating synthetic corpus: %s -> %s", counts, args.output_dir)
    records = generate_synthetic_dataset(
        output_dir=args.output_dir,
        recordings_per_class=counts,
        duration_range_s=(args.min_duration, args.max_duration),
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    metadata_path = Path(args.output_dir) / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["recording_id", "file_path", "class_name", "duration_s"]
        )
        writer.writeheader()
        writer.writerows(records)

    total_duration = sum(r["duration_s"] for r in records)
    logger.info(
        "Wrote %d recordings (%.1f min total) to %s; metadata: %s",
        len(records),
        total_duration / 60.0,
        args.output_dir,
        metadata_path,
    )


if __name__ == "__main__":
    main()
